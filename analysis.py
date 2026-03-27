import json
import shutil
import subprocess

import cv2
import numpy as np


def yellow_pixel_mask(frame):
    return (
        (frame[:, :, 2] > 200) &
        (frame[:, :, 1] > 200) &
        (frame[:, :, 0] < 100)
    )


def yellow_pixel_mask_hsv(frame_bgr):
    """HSV-based yellow detection, tolerant of shadows and compression."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([15, 70, 110]), np.array([45, 255, 255]))


def _frame_mask_for_region(frame, bbox=None):
    """Return a uint32 mask for the requested region without allocating a full-frame canvas."""
    if bbox is None:
        return yellow_pixel_mask(frame).astype(np.uint32)

    x, y, width, height = bbox
    frame_slice = frame[y : y + height, x : x + width]
    return yellow_pixel_mask(frame_slice).astype(np.uint32)


def clamp_bbox(bbox, frame_width, frame_height):
    if bbox is None:
        return None

    x, y, width, height = bbox
    x = max(0, min(int(x), frame_width))
    y = max(0, min(int(y), frame_height))
    width = max(0, min(int(width), frame_width - x))
    height = max(0, min(int(height), frame_height - y))

    if width == 0 or height == 0:
        return None
    return x, y, width, height


def _ffmpeg_binary():
    return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_binary():
    return shutil.which("ffprobe") or "ffprobe"


def _parse_ffprobe_fps(value):
    if not value or value in {"0/0", "0"}:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator or 0)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    return float(value)


def _probe_video_metadata(vid_path):
    command = [
        _ffprobe_binary(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        vid_path,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout or "{}")
    stream = (payload.get("streams") or [{}])[0]
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": _parse_ffprobe_fps(stream.get("avg_frame_rate")) or 30.0,
        "frame_count": int(stream.get("nb_frames") or 0),
    }


def _iter_ffmpeg_frames(vid_path, width, height):
    frame_size = int(width) * int(height) * 3
    command = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        vid_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if process.stdout is None:
            raise RuntimeError("FFmpeg did not provide a readable stdout pipe.")
        while True:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) != frame_size:
                raise RuntimeError("FFmpeg returned a truncated frame.")
            yield np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3)).copy()
    finally:
        stderr_text = ""
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(stderr_text or "FFmpeg failed while decoding the video.")


def get_total_yellow_pixels_from_video(
    vid_path, start_time=0, end_time=None, bbox=None, progress_callback=None
):
    if not hasattr(cv2, "VideoCapture"):
        metadata = _probe_video_metadata(vid_path)
        fps = metadata["fps"] or 30.0
        frame_width = int(metadata["width"])
        frame_height = int(metadata["height"])
        frame_count_total = int(metadata["frame_count"])
        if end_time is None:
            end_time = frame_count_total / fps if frame_count_total > 0 else 0.0

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time is not None else frame_count_total

        print(f"Video FPS: {fps}")
        print(f"Start Frame: {start_frame}")
        print(f"End Frame: {end_frame}")

        bbox = clamp_bbox(bbox, frame_width, frame_height)
        video_frame_totals = np.zeros((frame_height, frame_width), dtype=np.uint32)
        if bbox is None:
            accumulator_view = video_frame_totals
        else:
            x, y, width, height = bbox
            accumulator_view = video_frame_totals[y : y + height, x : x + width]

        phase_total = max(1, end_frame - start_frame)
        processed = 0
        progress_stride = max(1, phase_total // 150)

        for frame_index, frame in enumerate(_iter_ffmpeg_frames(vid_path, frame_width, frame_height)):
            if frame_index < start_frame:
                continue
            if frame_index >= end_frame:
                break
            if frame_index == start_frame or frame_index % max(int(fps), 1) == 0:
                print(f"Processing frame {frame_index}/{end_frame}", end="\r")

            accumulator_view += _frame_mask_for_region(frame, bbox=bbox)
            processed += 1
            if progress_callback and (
                processed == 1
                or processed % progress_stride == 0
                or processed >= phase_total
            ):
                progress_callback(processed, phase_total)
        print()
        return video_frame_totals

    cap = cv2.VideoCapture(vid_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    if fps <= 0:
        fps = 30.0

    if end_time is None:
        end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    print(f"Video FPS: {fps}")
    print(f"Start Frame: {start_frame}")
    print(f"End Frame: {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bbox = clamp_bbox(bbox, frame_width, frame_height)
    video_frame_totals = np.zeros((frame_height, frame_width), dtype=np.uint32)
    if bbox is None:
        accumulator_view = video_frame_totals
    else:
        x, y, width, height = bbox
        accumulator_view = video_frame_totals[y : y + height, x : x + width]

    phase_total = max(1, end_frame - start_frame)
    processed = 0
    progress_stride = max(1, phase_total // 150)

    # Go frame by frame while only touching the ROI accumulator when a bbox is provided.
    while frame_count < end_frame:
        if frame_count == start_frame or frame_count % max(int(fps), 1) == 0:
            print(f"Processing frame {frame_count}/{end_frame}", end="\r")
        ret, frame = cap.read()
        if not ret:
            break

        accumulator_view += _frame_mask_for_region(frame, bbox=bbox)

        frame_count += 1
        processed += 1
        if progress_callback and (
            processed == 1
            or processed % progress_stride == 0
            or processed >= phase_total
        ):
            progress_callback(processed, phase_total)
    cap.release()
    print()
    return video_frame_totals


def analyze_frame(frame, bbox=None):
    # OpenCV frames are BGR. Match the original RGB thresholds without converting.
    if bbox is None:
        return _frame_mask_for_region(frame)

    x, y, width, height = bbox
    mask = np.zeros(frame.shape[:2], dtype=np.uint32)
    mask[y : y + height, x : x + width] = _frame_mask_for_region(frame, bbox=bbox)
    return mask
