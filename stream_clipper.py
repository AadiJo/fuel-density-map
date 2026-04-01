from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytesseract
from yt_dlp import YoutubeDL


COUNTDOWN_RE = re.compile(r"(?<!\d)([0-3]):([0-5]\d)(?!\d)")
YOUTUBE_HLS_RE = re.compile(r'"hlsManifestUrl":"([^"]+)"')
# Default InnerTube clients (e.g. android_vr) may return UNPLAYABLE for some public uploads; android works.
YOUTUBE_YT_DLP_EXTRACTOR_ARGS: dict[str, Any] = {"youtube": {"player_client": ["android"]}}
BUN_CANDIDATE_PATHS = (
    Path.home() / ".bun" / "bin" / "bun",
    Path("/usr/local/bin/bun"),
)


@dataclass
class Config:
    stream_url: str
    work_dir: Path
    videos_dir: Path
    status_file: Path
    ffmpeg_bin: str
    ffprobe_bin: str
    pre_roll_sec: float
    post_roll_sec: float
    rolling_buffer_sec: float
    segment_time_sec: float
    start_threshold_sec: int
    max_match_sec: int
    stop_grace_sec: float
    min_timer_confidence: float
    ocr_region: tuple[float, float, float, float]


@dataclass
class SegmentInfo:
    path: Path
    index: int
    start_ts: float
    end_ts: float


@dataclass
class TimerReading:
    seconds: int
    text: str
    confidence: float
    detected_at: str
    region: str


class StreamClipper:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.stop_requested = False
        self.ffmpeg_proc: subprocess.Popen[str] | None = None
        self.segments: list[SegmentInfo] = []
        self.seen_segment_indexes: set[int] = set()
        self.analyzed_segment_indexes: set[int] = set()
        self.timer_history: deque[TimerReading] = deque(maxlen=8)
        self.recent_clips: deque[str] = deque(maxlen=8)
        self.active_match_started_at: float | None = None
        self.low_timer_seen_at: float | None = None
        self.zero_timer_seen_at: float | None = None
        self.last_timer_wallclock: float | None = None
        self.last_detection: TimerReading | None = None
        self.phase = "stopped"
        self.last_error: str | None = None
        self.stream_source_url: str | None = None
        self.match_state = "idle"
        self.updated_at = self.now_iso()

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def write_status(self) -> None:
        payload = {
            "running": self.phase not in {"stopped", "error"},
            "phase": self.phase,
            "matchState": self.match_state,
            "streamUrl": self.config.stream_url,
            "resolvedStreamUrl": self.stream_source_url,
            "startedAt": None,
            "updatedAt": self.updated_at,
            "lastError": self.last_error,
            "workDir": str(self.config.work_dir),
            "videosDir": str(self.config.videos_dir),
            "config": {
                "preRollSec": self.config.pre_roll_sec,
                "postRollSec": self.config.post_roll_sec,
                "rollingBufferSec": self.config.rolling_buffer_sec,
                "segmentTimeSec": self.config.segment_time_sec,
                "startThresholdSec": self.config.start_threshold_sec,
                "maxMatchSec": self.config.max_match_sec,
                "stopGraceSec": self.config.stop_grace_sec,
                "minTimerConfidence": self.config.min_timer_confidence,
                "ocrRegion": {
                    "x": self.config.ocr_region[0],
                    "y": self.config.ocr_region[1],
                    "width": self.config.ocr_region[2],
                    "height": self.config.ocr_region[3],
                },
            },
            "bufferedSeconds": round(len(self.segments) * self.config.segment_time_sec, 1),
            "bufferedSegments": len(self.segments),
            "latestDetection": asdict(self.last_detection) if self.last_detection else None,
            "recentClips": list(self.recent_clips),
            "activeMatch": {
                "startedAt": datetime.fromtimestamp(self.active_match_started_at, tz=timezone.utc).isoformat()
                if self.active_match_started_at
                else None,
                "secondsLeft": self.last_detection.seconds if self.match_state == "active" and self.last_detection else None,
                "statusText": self.status_text(),
            },
        }
        self.config.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.status_file.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    def status_text(self) -> str:
        if self.phase == "error":
            return "Stream error"
        if self.match_state == "active" and self.last_detection:
            return f"Match playing ({self.last_detection.text} left)"
        if self.phase == "saving":
            return "Saving clip"
        if self.phase in {"monitoring", "buffering"}:
            return "Watching stream"
        return "Stopped"

    def set_phase(self, phase: str, error: str | None = None) -> None:
        self.phase = phase
        self.last_error = error
        self.updated_at = self.now_iso()
        self.write_status()

    def resolve_stream_source(self) -> str:
        js_runtimes: dict[str, dict[str, str]] = {}
        for candidate in BUN_CANDIDATE_PATHS:
            if candidate.exists():
                js_runtimes["bun"] = {"path": str(candidate)}
                break
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
            "format": "best",
            "extractor_args": YOUTUBE_YT_DLP_EXTRACTOR_ARGS,
            "remote_components": ["ejs:github"],
            **({"js_runtimes": js_runtimes} if js_runtimes else {}),
        }
        with YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(self.config.stream_url, download=False)
            except Exception:  # noqa: BLE001
                info = None
        if not info:
            scraped = self.resolve_stream_source_from_watch_page(self.config.stream_url)
            if scraped:
                return scraped
            raise RuntimeError("Could not resolve the livestream URL.")
        requested_downloads = info.get("requested_downloads") or []
        for item in requested_downloads:
            url = item.get("url")
            if isinstance(url, str) and url:
                return url
        formats = info.get("formats") or []
        scored_formats: list[tuple[int, str]] = []
        for fmt in formats:
            url = fmt.get("url")
            protocol = fmt.get("protocol") or ""
            if not isinstance(url, str) or not url:
                continue
            if protocol not in {"m3u8", "m3u8_native", "https", "http_dash_segments", "dash"}:
                continue
            height = int(fmt.get("height") or 0)
            scored_formats.append((height, url))
        if scored_formats:
            scored_formats.sort(reverse=True)
            return scored_formats[0][1]
        direct_url = info.get("url")
        if isinstance(direct_url, str) and direct_url:
            return direct_url
        scraped = self.resolve_stream_source_from_watch_page(self.config.stream_url)
        if scraped:
            return scraped
        raise RuntimeError("yt-dlp resolved the page but did not return a playable stream URL.")

    def resolve_stream_source_from_watch_page(self, watch_url: str) -> str | None:
        if "youtube.com" not in watch_url and "youtu.be" not in watch_url:
            return None
        request = Request(
            watch_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(request, timeout=20) as response:
            html = response.read().decode("utf-8", errors="replace")
        match = YOUTUBE_HLS_RE.search(html)
        if not match:
            return None
        return json.loads(f'"{unescape(match.group(1))}"')

    def start_ffmpeg(self, input_url: str) -> None:
        segments_dir = self.config.work_dir / "segments"
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
        segments_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_url,
            "-map",
            "0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.config.segment_time_sec),
            "-segment_format",
            "mpegts",
            "-reset_timestamps",
            "1",
            str(segments_dir / "segment-%06d.ts"),
        ]
        self.ffmpeg_proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    def scan_segments(self) -> None:
        segments_dir = self.config.work_dir / "segments"
        segment_paths = sorted(segments_dir.glob("segment-*.ts"))
        fresh_segments: list[SegmentInfo] = []
        for segment_path in segment_paths:
            try:
                stat = segment_path.stat()
            except FileNotFoundError:
                continue
            match = re.search(r"segment-(\d+)\.ts$", segment_path.name)
            if not match:
                continue
            index = int(match.group(1))
            end_ts = stat.st_mtime
            start_ts = max(0.0, end_ts - self.config.segment_time_sec)
            fresh_segments.append(SegmentInfo(path=segment_path, index=index, start_ts=start_ts, end_ts=end_ts))
            self.seen_segment_indexes.add(index)
        self.segments = fresh_segments

    def prune_segments(self) -> None:
        keep_from = time.time() - self.config.rolling_buffer_sec
        if self.active_match_started_at is not None:
            keep_from = min(keep_from, self.active_match_started_at - 2)
        for segment in list(self.segments):
            if segment.end_ts < keep_from and segment.path.exists():
                try:
                    segment.path.unlink()
                except FileNotFoundError:
                    pass
        self.scan_segments()

    def capture_frame(self, segment_path: Path) -> Any | None:
        command = [
            self.config.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(max(self.config.segment_time_sec / 2.0, 0.1)),
            "-i",
            str(segment_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ]
        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            return None
        frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def ocr_timer(self, frame: Any) -> TimerReading | None:
        height, width = frame.shape[:2]
        regions = [("configured", self.config.ocr_region)]
        best: TimerReading | None = None
        for region_name, (x, y, w, h) in regions:
            x0 = int(width * x)
            y0 = int(height * y)
            x1 = int(width * (x + w))
            y1 = int(height * (y + h))
            crop = frame[y0:y1, x0:x1]
            reading = self.ocr_timer_from_crop(crop, region_name)
            if reading and (best is None or reading.confidence > best.confidence):
                best = reading
        return best

    def candidate_timer_crops(self, crop: Any) -> list[tuple[str, Any]]:
        if crop.size == 0:
            return []

        candidates: list[tuple[str, Any]] = [("full", crop)]
        height, width = crop.shape[:2]

        def add_relative(name: str, x: float, y: float, w: float, h: float) -> None:
            x0 = max(0, min(width - 1, int(width * x)))
            y0 = max(0, min(height - 1, int(height * y)))
            x1 = max(x0 + 1, min(width, int(width * (x + w))))
            y1 = max(y0 + 1, min(height, int(height * (y + h))))
            sub = crop[y0:y1, x0:x1]
            if sub.size:
                candidates.append((name, sub))

        add_relative("mid-band", 0.14, 0.08, 0.72, 0.84)
        add_relative("timer-center", 0.34, 0.0, 0.32, 1.0)
        add_relative("timer-tight", 0.39, 0.05, 0.22, 0.9)
        add_relative("timer-wide", 0.28, 0.0, 0.44, 1.0)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_x = width / 2.0
        center_y = height / 2.0

        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < width * height * 0.01:
                continue
            if w < width * 0.08 or w > width * 0.7:
                continue
            if h < height * 0.2 or h > height * 0.95:
                continue
            dist = abs((x + w / 2) - center_x) + abs((y + h / 2) - center_y) * 0.5
            score = area - dist * 20
            scored.append((score, (x, y, w, h)))

        if scored:
            scored.sort(reverse=True)
            x, y, w, h = scored[0][1]
            pad_x = max(4, int(w * 0.08))
            pad_y = max(4, int(h * 0.08))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(width, x + w + pad_x)
            y1 = min(height, y + h + pad_y)
            bright_crop = crop[y0:y1, x0:x1]
            if bright_crop.size:
                candidates.append(("bright-box", bright_crop))

        return candidates

    def ocr_timer_from_crop(self, crop: Any, region_name: str) -> TimerReading | None:
        if crop.size == 0:
            return None

        best: TimerReading | None = None
        for sub_name, sub_crop in self.candidate_timer_crops(crop):
            gray = cv2.cvtColor(sub_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            variants = [
                resized,
                binary,
                cv2.bitwise_not(binary),
                cv2.adaptiveThreshold(
                    resized,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    9,
                ),
                cv2.bitwise_not(
                    cv2.adaptiveThreshold(
                        resized,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        31,
                        9,
                    )
                ),
            ]

            for image in variants:
                for psm in (8, 7, 6):
                    data = pytesseract.image_to_data(
                        image,
                        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789:",
                        output_type=pytesseract.Output.DICT,
                    )
                    text = " ".join(part.strip() for part in data.get("text", []) if part and part.strip())
                    normalized = text.replace(" ", "").replace(".", ":").replace(";", ":")
                    match = COUNTDOWN_RE.search(normalized)
                    if not match:
                        raw_text = pytesseract.image_to_string(
                            image,
                            config=f"--psm {psm} -c tessedit_char_whitelist=0123456789:",
                        )
                        normalized_raw = raw_text.strip().replace(" ", "").replace(".", ":").replace(";", ":")
                        match = COUNTDOWN_RE.search(normalized_raw)
                        if not match:
                            continue
                        confidence = 0.24 if "tight" in sub_name or "center" in sub_name else 0.16
                    else:
                        confidences = [float(value) for value in data.get("conf", []) if value not in {"-1", "", None}]
                        confidence = max(confidences) / 100.0 if confidences else 0.12

                    seconds = int(match.group(1)) * 60 + int(match.group(2))
                    if seconds > self.config.max_match_sec:
                        continue
                    reading = TimerReading(
                        seconds=seconds,
                        text=f"{match.group(1)}:{match.group(2)}",
                        confidence=confidence,
                        detected_at=self.now_iso(),
                        region=f"{region_name}:{sub_name}",
                    )
                    if best is None or reading.confidence > best.confidence:
                        best = reading
        return best

    def stable_countdown(self) -> bool:
        if len(self.timer_history) < 3:
            return False
        readings = list(self.timer_history)[-3:]
        seconds = [reading.seconds for reading in readings]
        if max(seconds) > self.config.max_match_sec:
            return False
        if min(seconds) < 0:
            return False
        if seconds[0] < self.config.start_threshold_sec:
            return False
        deltas = [seconds[index] - seconds[index + 1] for index in range(len(seconds) - 1)]
        return all(delta >= 0 for delta in deltas) and all(delta <= 6 for delta in deltas)

    def maybe_start_match(self, reading: TimerReading, segment: SegmentInfo) -> None:
        if self.match_state != "idle":
            return
        if not self.stable_countdown():
            return
        self.active_match_started_at = max(segment.start_ts - self.config.pre_roll_sec, 0.0)
        self.low_timer_seen_at = None
        self.zero_timer_seen_at = None
        self.match_state = "active"
        self.phase = "monitoring"
        self.updated_at = self.now_iso()
        self.write_status()

    def update_match_from_reading(self, reading: TimerReading, segment: SegmentInfo) -> None:
        self.last_detection = reading
        self.last_timer_wallclock = segment.end_ts
        self.timer_history.append(reading)
        self.updated_at = self.now_iso()
        self.maybe_start_match(reading, segment)
        if self.match_state != "active":
            self.write_status()
            return
        if reading.seconds <= 3 and self.low_timer_seen_at is None:
            self.low_timer_seen_at = segment.end_ts
        if reading.seconds <= 0 and self.zero_timer_seen_at is None:
            self.zero_timer_seen_at = segment.end_ts
        self.write_status()

    def maybe_finish_match(self) -> None:
        if self.match_state != "active" or self.active_match_started_at is None:
            return
        now_ts = time.time()
        if self.zero_timer_seen_at and now_ts - self.zero_timer_seen_at >= self.config.post_roll_sec:
            self.finish_match(self.zero_timer_seen_at + self.config.post_roll_sec)
            return
        if self.low_timer_seen_at and self.last_timer_wallclock and now_ts - self.last_timer_wallclock >= self.config.stop_grace_sec:
            self.finish_match(self.last_timer_wallclock + self.config.post_roll_sec)

    def finish_match(self, clip_end_ts: float) -> None:
        if self.active_match_started_at is None:
            return
        self.phase = "saving"
        self.updated_at = self.now_iso()
        self.write_status()
        clip_file = self.export_clip(self.active_match_started_at, clip_end_ts)
        if clip_file:
            self.recent_clips.appendleft(clip_file.name)
        self.active_match_started_at = None
        self.low_timer_seen_at = None
        self.zero_timer_seen_at = None
        self.match_state = "idle"
        self.phase = "monitoring"
        self.updated_at = self.now_iso()
        self.write_status()

    def export_clip(self, clip_start_ts: float, clip_end_ts: float) -> Path | None:
        selected = [segment for segment in self.segments if segment.end_ts >= clip_start_ts and segment.start_ts <= clip_end_ts]
        if not selected:
            return None
        concat_file = self.config.work_dir / "clip-inputs.txt"
        concat_lines = []
        for segment in selected:
            escaped_path = segment.path.as_posix().replace("'", "'\\''")
            concat_lines.append(f"file '{escaped_path}'")
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        clip_started = datetime.fromtimestamp(clip_start_ts)
        output_path = self.config.videos_dir / f"match-{clip_started.strftime('%Y%m%d-%H%M%S')}.mp4"
        trim_start = max(0.0, clip_start_ts - selected[0].start_ts)
        trim_duration = max(1.0, clip_end_ts - clip_start_ts)
        copy_command = [
            self.config.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-ss",
            f"{trim_start:.3f}",
            "-t",
            f"{trim_duration:.3f}",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(copy_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            transcode_command = [
                self.config.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-ss",
                f"{trim_start:.3f}",
                "-t",
                f"{trim_duration:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            subprocess.run(transcode_command, check=True, capture_output=True, text=True)
        return output_path

    def analyze_latest_segment(self) -> None:
        if not self.segments:
            return
        candidate = self.segments[-2] if len(self.segments) > 1 else self.segments[-1]
        if candidate.index in self.analyzed_segment_indexes:
            return
        frame = self.capture_frame(candidate.path)
        self.analyzed_segment_indexes.add(candidate.index)
        if frame is None:
            return
        reading = self.ocr_timer(frame)
        if reading is None:
            self.updated_at = self.now_iso()
            self.write_status()
            return
        self.update_match_from_reading(reading, candidate)

    def stop_ffmpeg(self) -> None:
        if self.ffmpeg_proc is None:
            return
        if self.ffmpeg_proc.poll() is not None:
            return
        self.ffmpeg_proc.send_signal(signal.SIGINT)
        try:
            self.ffmpeg_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.ffmpeg_proc.kill()
            self.ffmpeg_proc.wait(timeout=3)

    def run(self) -> int:
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self.config.videos_dir.mkdir(parents=True, exist_ok=True)
        self.set_phase("starting")
        try:
            self.stream_source_url = self.resolve_stream_source()
            self.start_ffmpeg(self.stream_source_url)
            self.set_phase("buffering")
            while not self.stop_requested:
                if self.ffmpeg_proc and self.ffmpeg_proc.poll() is not None:
                    stderr = ""
                    if self.ffmpeg_proc.stderr:
                        stderr = self.ffmpeg_proc.stderr.read().strip()
                    raise RuntimeError(stderr or "ffmpeg stopped unexpectedly while recording the stream.")
                self.scan_segments()
                if self.segments and self.phase == "buffering":
                    self.phase = "monitoring"
                self.analyze_latest_segment()
                self.maybe_finish_match()
                self.prune_segments()
                self.updated_at = self.now_iso()
                self.write_status()
                time.sleep(0.8)
            if self.match_state == "active" and self.last_timer_wallclock:
                self.finish_match(self.last_timer_wallclock + self.config.post_roll_sec)
            self.stop_ffmpeg()
            self.set_phase("stopped")
            return 0
        except Exception as exc:  # noqa: BLE001
            self.stop_ffmpeg()
            self.set_phase("error", str(exc))
            return 1


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Record an FRC livestream into per-match clips.")
    parser.add_argument("--stream-url", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--pre-roll-sec", type=float, default=8.0)
    parser.add_argument("--post-roll-sec", type=float, default=8.0)
    parser.add_argument("--rolling-buffer-sec", type=float, default=45.0)
    parser.add_argument("--segment-time-sec", type=float, default=2.0)
    parser.add_argument("--start-threshold-sec", type=int, default=75)
    parser.add_argument("--max-match-sec", type=int, default=180)
    parser.add_argument("--stop-grace-sec", type=float, default=5.0)
    parser.add_argument("--min-timer-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-region", default="0.34,0.84,0.32,0.13")
    args = parser.parse_args()
    ocr_region = tuple(float(part) for part in args.ocr_region.split(",", 3))
    if len(ocr_region) != 4:
        raise ValueError("ocr-region must be x,y,width,height")
    return Config(
        stream_url=args.stream_url,
        work_dir=Path(args.work_dir),
        videos_dir=Path(args.videos_dir),
        status_file=Path(args.status_file),
        ffmpeg_bin=args.ffmpeg_bin,
        ffprobe_bin=args.ffprobe_bin,
        pre_roll_sec=args.pre_roll_sec,
        post_roll_sec=args.post_roll_sec,
        rolling_buffer_sec=args.rolling_buffer_sec,
        segment_time_sec=args.segment_time_sec,
        start_threshold_sec=args.start_threshold_sec,
        max_match_sec=args.max_match_sec,
        stop_grace_sec=args.stop_grace_sec,
        min_timer_confidence=args.min_timer_confidence,
        ocr_region=ocr_region,
    )


def main() -> int:
    config = parse_args()
    clipper = StreamClipper(config)

    def request_stop(_signum: int, _frame: Any) -> None:
        clipper.stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return clipper.run()


if __name__ == "__main__":
    raise SystemExit(main())
