import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image

import analysis

DEFAULT_FIELD_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__),
    "webui",
    "public",
    "assets",
    "rebuilt-field.png",
)
DEFAULT_TARGET_PROCESS_FPS = 15.0
DEFAULT_WORKING_SCALE = 0.75
DEFAULT_MAX_ACTIVE_TRACKS = 900
DEFAULT_TRACK_MAX_AGE = 18

PROGRESS_PREFIX = "PROGRESS_JSON:"

FIELD_DESTINATION_BOUNDS = {
    "top_left": (0.133, 0.053),
    "top_right": (0.866, 0.053),
    "bottom_right": (0.866, 0.946),
    "bottom_left": (0.133, 0.946),
}

# Normalized polygons on the field asset for the two side hex goals. Any projected fuel point
# that lands inside these shapes is excluded from the field-map export.
FIELD_FUEL_EXCLUSION_ZONES = (
    np.array(
        [
            [0.2812, 0.4991],
            [0.2966, 0.4302],
            [0.3302, 0.4302],
            [0.3461, 0.4991],
            [0.3302, 0.5660],
            [0.2966, 0.5660],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0.6178, 0.4991],
            [0.6337, 0.4302],
            [0.6675, 0.4302],
            [0.6834, 0.4991],
            [0.6675, 0.5660],
            [0.6337, 0.5660],
        ],
        dtype=np.float32,
    ),
)

# Third JSON field kept for compatibility; UI draws a fixed size.
FIELD_MAP_FUEL_RADIUS = 10

OVERLAY_FUEL_DOT_RADIUS_PX = 5

FUEL_TRACK_MATCH_RADIUS_PX = 14.0
FUEL_TRACK_MAX_MISSES = 2
BAD_FRAME_HISTORY_SIZE = 6
BAD_FRAME_MIN_BASELINE_COUNT = 6
BAD_FRAME_MIN_COUNT_DELTA = 8
BAD_FRAME_COUNT_DELTA_RATIO = 0.45

_MORPH_KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
LEGACY_MAX_CENTERS = 500


def emit_progress(phase, current, total):
    """Machine-readable progress for the Node server (stderr, line-buffered)."""
    line = f'{PROGRESS_PREFIX}{json.dumps({"phase": phase, "current": current, "total": total})}\n'
    sys.stderr.write(line)
    sys.stderr.flush()


def compute_total_work_units(video_path):
    """Single metadata probe over the source video."""
    return max(1, int(probe_video_metadata(video_path)["frame_count"]))


def _odd_kernel_size(value):
    size = max(1, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _safe_percentile(values, percentile):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def summarize_counts(values):
    if not values:
        return {"min": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": int(arr.min()),
        "p50": _safe_percentile(values, 50),
        "p90": _safe_percentile(values, 90),
        "p95": _safe_percentile(values, 95),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def create_color_array(raw_data, max_value, average_of_non_zero_values, average_display_color):
    color_array = np.zeros((*raw_data.shape, 3), dtype=np.uint8)

    if max_value <= 0:
        return color_array

    average_color = np.array(average_display_color, dtype=np.float32)
    non_zero_mask = raw_data > 0

    if average_of_non_zero_values > 0:
        lower_mask = non_zero_mask & (raw_data <= average_of_non_zero_values)
        if np.any(lower_mask):
            lower_ratio = (raw_data[lower_mask] / average_of_non_zero_values).astype(np.float32)
            color_array[lower_mask] = np.clip(lower_ratio[:, None] * average_color, 0, 255).astype(np.uint8)

    upper_mask = non_zero_mask & (raw_data > average_of_non_zero_values)
    upper_range = max_value - average_of_non_zero_values
    if np.any(upper_mask):
        if upper_range <= 0:
            color_array[upper_mask] = 255
        else:
            upper_ratio = ((raw_data[upper_mask] - average_of_non_zero_values) / upper_range).astype(np.float32)
            color_array[upper_mask] = np.clip(
                average_color + upper_ratio[:, None] * (255 - average_color),
                0,
                255,
            ).astype(np.uint8)

    color_array[raw_data >= max_value] = 255
    return color_array


def parse_bbox(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Bounding box must be x,y,width,height")
    return tuple(parts)


def parse_quad(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 8:
        raise ValueError("Quad must be x1,y1,x2,y2,x3,y3,x4,y4")
    return np.array(
        [
            [parts[0], parts[1]],
            [parts[2], parts[3]],
            [parts[4], parts[5]],
            [parts[6], parts[7]],
        ],
        dtype=np.float32,
    )


def build_destination_quad(width, height):
    return np.array(
        [
            [width * FIELD_DESTINATION_BOUNDS["top_left"][0], height * FIELD_DESTINATION_BOUNDS["top_left"][1]],
            [width * FIELD_DESTINATION_BOUNDS["top_right"][0], height * FIELD_DESTINATION_BOUNDS["top_right"][1]],
            [width * FIELD_DESTINATION_BOUNDS["bottom_right"][0], height * FIELD_DESTINATION_BOUNDS["bottom_right"][1]],
            [width * FIELD_DESTINATION_BOUNDS["bottom_left"][0], height * FIELD_DESTINATION_BOUNDS["bottom_left"][1]],
        ],
        dtype=np.float32,
    )


def quad_mask(bbox, field_quad):
    x, y, _, _ = bbox
    local_quad = np.round(field_quad - np.array([x, y], dtype=np.float32)).astype(np.int32)
    local_quad[:, 0] = np.clip(local_quad[:, 0], 0, bbox[2] - 1)
    local_quad[:, 1] = np.clip(local_quad[:, 1], 0, bbox[3] - 1)
    mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, local_quad, 255)
    return mask


def precompute_roi_quad_mask(bbox, field_quad):
    """Quad intersection mask depends only on bbox + quad, not on video frame."""
    if field_quad is None:
        return None
    return quad_mask(bbox, field_quad)


def roi_yellow_mask_binary(frame, bbox, field_quad, quad_mask_roi=None):
    """HSV yellow mask cropped to the ROI and field quad, with light noise removal."""
    x, y, width, height = bbox
    frame_slice = frame[y : y + height, x : x + width]
    if frame_slice.size == 0:
        return None
    mask = analysis.yellow_pixel_mask_hsv(frame_slice)
    if field_quad is not None:
        qm = quad_mask_roi if quad_mask_roi is not None else quad_mask(bbox, field_quad)
        mask = cv2.bitwise_and(mask, qm)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL_OPEN)
    return mask


def compute_frame_stride(source_fps, target_fps):
    if source_fps <= 0:
        source_fps = 30.0
    if target_fps is None or target_fps <= 0 or target_fps >= source_fps:
        return 1
    return max(1, int(round(float(source_fps) / float(target_fps))))


def draw_fuel_dots_full_frame(height, width, bbox, centers, overlay_color):
    """RGB canvas; LINE_8 is much faster than LINE_AA for hundreds of small circles."""
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if not centers:
        return out
    color = tuple(int(c) for c in overlay_color)
    dot_r = OVERLAY_FUEL_DOT_RADIUS_PX
    x0, y0 = bbox[0], bbox[1]
    for cx, cy in centers:
        cv2.circle(out, (int(x0 + cx), int(y0 + cy)), dot_r, color, -1, lineType=cv2.LINE_8)
    return out


class ProcessingMetrics:
    def __init__(self):
        self.stage_seconds = {
            "decode": 0.0,
            "mask": 0.0,
            "detect": 0.0,
            "stabilize": 0.0,
            "project": 0.0,
            "encode": 0.0,
            "total": 0.0,
        }
        self.raw_center_counts = []
        self.stable_track_counts = []
        self.detector_budget_hits = 0
        self.saturated_frame_count = 0

    def add_time(self, stage_name, seconds):
        self.stage_seconds[stage_name] = self.stage_seconds.get(stage_name, 0.0) + float(seconds)

    def add_counts(self, raw_count, stable_count, saturated):
        self.raw_center_counts.append(int(raw_count))
        self.stable_track_counts.append(int(stable_count))
        if saturated:
            self.detector_budget_hits += 1
            self.saturated_frame_count += 1

    def to_stats(self):
        return {
            "timings": {name: round(value, 6) for name, value in self.stage_seconds.items()},
            "rawCenterCountSummary": summarize_counts(self.raw_center_counts),
            "stableTrackCountSummary": summarize_counts(self.stable_track_counts),
            "detectorBudgetHits": int(self.detector_budget_hits),
            "saturatedFrameCount": int(self.saturated_frame_count),
        }


class PeakBallDetector:
    """Fast peak detector designed to be portable to a CUDA response-map implementation later."""

    def __init__(self, working_scale=DEFAULT_WORKING_SCALE, detector_budget=None, warmup_frames=12, ema_alpha=0.2):
        self.working_scale = float(np.clip(working_scale, 0.2, 1.0))
        self.requested_budget = int(detector_budget) if detector_budget else None
        self.warmup_frames = max(1, int(warmup_frames))
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 0.95))
        self._estimated_ball_diameter = None
        self._calibration_frames = 0

    @property
    def estimated_ball_diameter(self):
        return float(self._estimated_ball_diameter or 8.0)

    def _scaled_mask(self, mask_binary):
        if self.working_scale >= 0.999:
            return mask_binary, 1.0
        scaled = cv2.resize(
            mask_binary,
            dsize=None,
            fx=self.working_scale,
            fy=self.working_scale,
            interpolation=cv2.INTER_AREA,
        )
        return scaled, self.working_scale

    def _update_ball_size_estimate(self, mask_binary):
        should_calibrate = self._calibration_frames < self.warmup_frames or self._calibration_frames % 30 == 0
        self._calibration_frames += 1
        if not should_calibrate:
            return

        mask_work = (mask_binary > 0).astype(np.uint8) * 255
        if not np.any(mask_work):
            return

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_work)
        if n_labels <= 1:
            return

        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        candidates = areas[(areas >= 4) & (areas <= 800)]
        if candidates.size == 0:
            return

        if candidates.size > 16:
            cutoff = max(1, int(math.ceil(candidates.size * 0.65)))
            candidates = np.sort(candidates)[:cutoff]

        diameter = float(np.median(2.0 * np.sqrt(candidates / np.pi)))
        diameter = float(np.clip(diameter, 3.0, 24.0))
        if self._estimated_ball_diameter is None:
            self._estimated_ball_diameter = diameter
            return
        self._estimated_ball_diameter = (
            (1.0 - self.ema_alpha) * float(self._estimated_ball_diameter)
            + self.ema_alpha * diameter
        )

    def resolve_budget(self, mask_shape):
        if self.requested_budget is not None:
            return max(1, int(self.requested_budget))
        ball_diameter = self.estimated_ball_diameter / max(self.working_scale, 1e-6)
        single_ball_area = max(math.pi * (ball_diameter * 0.5) ** 2, 20.0)
        roi_area = float(mask_shape[0] * mask_shape[1])
        derived = int(roi_area / max(single_ball_area * 2.0, 48.0))
        return int(np.clip(derived, 128, DEFAULT_MAX_ACTIVE_TRACKS))

    def _grid_nms(self, points_xy, scores, nms_radius, budget):
        selected = []
        selected_grid = {}
        if len(points_xy) == 0:
            return selected, False

        order = np.argsort(-scores)
        cell_size = max(1.0, float(nms_radius))
        radius_sq = float(nms_radius) * float(nms_radius)
        saturated = False

        for idx in order:
            if len(selected) >= budget:
                saturated = True
                break

            px, py = float(points_xy[idx][0]), float(points_xy[idx][1])
            cx = int(px // cell_size)
            cy = int(py // cell_size)
            keep = True
            for gx in range(cx - 1, cx + 2):
                for gy in range(cy - 1, cy + 2):
                    for qx, qy in selected_grid.get((gx, gy), ()):
                        dx = px - qx
                        dy = py - qy
                        if dx * dx + dy * dy < radius_sq:
                            keep = False
                            break
                    if not keep:
                        break
                if not keep:
                    break

            if not keep:
                continue

            selected.append((px, py))
            selected_grid.setdefault((cx, cy), []).append((px, py))

        return selected, saturated

    def detect(self, mask_binary):
        if mask_binary is None or not np.any(mask_binary):
            return [], False

        working_mask, scale = self._scaled_mask(mask_binary)
        self._update_ball_size_estimate(working_mask)

        ball_diameter = self.estimated_ball_diameter
        blur_size = _odd_kernel_size(max(3.0, ball_diameter * 1.1))
        response = cv2.boxFilter(
            working_mask.astype(np.float32),
            ddepth=cv2.CV_32F,
            ksize=(blur_size, blur_size),
            normalize=True,
        )
        response_max = float(response.max()) if response.size else 0.0
        if response_max <= 0:
            return [], False

        local_max_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                _odd_kernel_size(max(3.0, ball_diameter * 0.9)),
                _odd_kernel_size(max(3.0, ball_diameter * 0.9)),
            ),
        )
        local_max = cv2.dilate(response, local_max_kernel)
        peak_cutoff = max(22.0, min(170.0, response_max * 0.38))
        candidate_mask = (response >= (local_max - 1e-3)) & (response >= peak_cutoff) & (working_mask > 0)
        ys, xs = np.where(candidate_mask)
        if len(xs) == 0:
            return [], False

        scores = response[ys, xs]
        candidate_points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        budget = self.resolve_budget(mask_binary.shape)
        nms_radius = max(2.0, ball_diameter * 0.8)
        selected_scaled, saturated = self._grid_nms(candidate_points, scores, nms_radius, budget)

        refine_radius = max(2, int(round(ball_diameter)))
        centers = []
        inv_scale = 1.0 / scale
        for sx, sy in selected_scaled:
            cx = int(round(sx))
            cy = int(round(sy))
            x0 = max(0, cx - refine_radius)
            x1 = min(working_mask.shape[1], cx + refine_radius + 1)
            y0 = max(0, cy - refine_radius)
            y1 = min(working_mask.shape[0], cy + refine_radius + 1)
            patch = working_mask[y0:y1, x0:x1]
            if patch.size == 0 or not np.any(patch):
                refined_x = float(cx)
                refined_y = float(cy)
            else:
                moments = cv2.moments(patch, binaryImage=True)
                if moments["m00"] > 0:
                    refined_x = x0 + (moments["m10"] / moments["m00"])
                    refined_y = y0 + (moments["m01"] / moments["m00"])
                else:
                    refined_x = float(cx)
                    refined_y = float(cy)

            centers.append((int(round(refined_x * inv_scale)), int(round(refined_y * inv_scale))))

        return centers, saturated


def _dt_peaks_in_component(component_mask, budget, min_sep):
    """Distance-transform local maxima within a single connected component."""
    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5).astype(np.float32)
    peak_val = dist.max()
    if peak_val < 0.5:
        return []

    dil = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    lm = np.isclose(dist, dil, rtol=0, atol=1e-3) & (dist >= 0.5) & (component_mask > 0)
    ys, xs = np.where(lm)
    if len(xs) == 0:
        return []

    strengths = dist[lm]
    order = np.argsort(-strengths)
    sep2 = min_sep * min_sep
    peaks = []
    for idx in order:
        if len(peaks) >= budget:
            break
        py, px = int(ys[idx]), int(xs[idx])
        if all((px - ux) ** 2 + (py - uy) ** 2 >= sep2 for ux, uy in peaks):
            peaks.append((px, py))
    return peaks


def legacy_ball_centers_from_mask(mask_binary, max_centers=LEGACY_MAX_CENTERS):
    mask_work = (mask_binary > 0).astype(np.uint8) * 255
    if not np.any(mask_work):
        return [], False

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_work)
    if n_labels <= 1:
        return [], False

    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)])
    sorted_areas = np.sort(areas)
    bottom_n = max(1, len(sorted_areas) * 40 // 100)
    single_ball_area = float(np.clip(np.median(sorted_areas[:bottom_n]), 6, 200))
    min_sep = max(1.5, np.sqrt(single_ball_area) * 0.35)

    centers = []
    saturated = False
    for i in range(1, n_labels):
        if len(centers) >= max_centers:
            saturated = True
            break

        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        estimated_balls = max(1, round(area / single_ball_area))

        if estimated_balls == 1:
            centers.append((int(cx), int(cy)))
            continue

        component_mask = (labels == i).astype(np.uint8) * 255
        peaks = _dt_peaks_in_component(component_mask, estimated_balls, min_sep)

        if peaks:
            centers.extend(peaks)
        else:
            centers.append((int(cx), int(cy)))

        shortfall = estimated_balls - len(peaks)
        if shortfall > 0 and peaks:
            comp_ys, comp_xs = np.where(component_mask > 0)
            if len(comp_xs) > 0:
                existing = set(peaks)
                indices = np.linspace(0, len(comp_xs) - 1, shortfall + 2, dtype=int)[1:-1]
                for idx in indices:
                    pt = (int(comp_xs[idx]), int(comp_ys[idx]))
                    if pt not in existing:
                        centers.append(pt)
                        if len(centers) >= max_centers:
                            saturated = True
                            break
        if len(centers) >= max_centers:
            saturated = True
            break

    return centers[:max_centers], saturated


def ball_centers_from_mask(
    mask_binary,
    max_centers=None,
    working_scale=DEFAULT_WORKING_SCALE,
    detector_budget=None,
    detector_mode="legacy",
):
    if detector_mode == "peak":
        detector = PeakBallDetector(working_scale=working_scale, detector_budget=detector_budget)
        centers, _ = detector.detect(mask_binary)
        if max_centers is not None:
            return centers[: int(max_centers)]
        return centers

    legacy_centers, _ = legacy_ball_centers_from_mask(
        mask_binary,
        max_centers=int(max_centers or LEGACY_MAX_CENTERS),
    )
    return legacy_centers


class FuelTemporalStabilizer:
    """Hold stationary fuel through brief dropouts, reject obvious spikes, and keep track counts bounded."""

    def __init__(
        self,
        match_radius_px=FUEL_TRACK_MATCH_RADIUS_PX,
        max_misses=FUEL_TRACK_MAX_MISSES,
        history_size=BAD_FRAME_HISTORY_SIZE,
        min_baseline_count=BAD_FRAME_MIN_BASELINE_COUNT,
        min_count_delta=BAD_FRAME_MIN_COUNT_DELTA,
        count_delta_ratio=BAD_FRAME_COUNT_DELTA_RATIO,
        max_active_tracks=DEFAULT_MAX_ACTIVE_TRACKS,
        max_track_age=DEFAULT_TRACK_MAX_AGE,
        dedupe_radius_px=None,
    ):
        self.match_radius = float(match_radius_px)
        self.match_radius_sq = self.match_radius * self.match_radius
        self.max_misses = int(max_misses)
        self.min_baseline_count = int(min_baseline_count)
        self.min_count_delta = int(min_count_delta)
        self.count_delta_ratio = float(count_delta_ratio)
        self.count_history = deque(maxlen=max(1, int(history_size)))
        self.max_active_tracks = max(1, int(max_active_tracks))
        self.max_track_age = max(1, int(max_track_age))
        if dedupe_radius_px is None:
            self.dedupe_radius = 0.0
        else:
            self.dedupe_radius = max(0.0, float(dedupe_radius_px))
        self._next_track_id = 1
        self.tracks = []

    def _baseline_count(self):
        if not self.count_history:
            return None
        return float(np.median(np.array(self.count_history, dtype=np.float32)))

    def _is_bad_frame(self, raw_count):
        baseline = self._baseline_count()
        if baseline is None or baseline < self.min_baseline_count:
            return False
        allowed_delta = max(self.min_count_delta, int(round(baseline * self.count_delta_ratio)))
        return abs(int(raw_count) - int(round(baseline))) >= allowed_delta

    def _current_centers(self):
        return [(int(track["x"]), int(track["y"])) for track in self.tracks]

    def _dedupe_centers(self, centers, radius):
        if not centers:
            return []
        if radius <= 0:
            return [(int(cx), int(cy)) for cx, cy in centers]
        cell_size = max(1.0, float(radius))
        radius_sq = float(radius) * float(radius)
        grid = {}
        deduped = []
        for cx, cy in centers:
            fx = float(cx)
            fy = float(cy)
            gx = int(fx // cell_size)
            gy = int(fy // cell_size)
            keep = True
            for nx in range(gx - 1, gx + 2):
                for ny in range(gy - 1, gy + 2):
                    for ox, oy in grid.get((nx, ny), ()):
                        dx = fx - ox
                        dy = fy - oy
                        if dx * dx + dy * dy < radius_sq:
                            keep = False
                            break
                    if not keep:
                        break
                if not keep:
                    break
            if not keep:
                continue
            deduped.append((int(round(fx)), int(round(fy))))
            grid.setdefault((gx, gy), []).append((fx, fy))
        return deduped

    def _prune_tracks(self):
        if len(self.tracks) <= self.max_active_tracks:
            return
        self.tracks.sort(
            key=lambda track: (
                int(track["missed"]),
                -int(track["age"]),
                int(track["id"]),
            )
        )
        self.tracks = self.tracks[: self.max_active_tracks]

    def _match_and_update_tracks(self, centers, saturated=False):
        deduped_centers = self._dedupe_centers(centers, self.dedupe_radius)
        if not self.tracks:
            self.tracks = [
                {"id": self._next_track_id + idx, "x": int(cx), "y": int(cy), "missed": 0, "age": 0}
                for idx, (cx, cy) in enumerate(deduped_centers[: self.max_active_tracks])
            ]
            self._next_track_id += len(self.tracks)
            return

        cell_size = max(1.0, self.match_radius)
        track_grid = {}
        for track_index, track in enumerate(self.tracks):
            gx = int(float(track["x"]) // cell_size)
            gy = int(float(track["y"]) // cell_size)
            track_grid.setdefault((gx, gy), []).append(track_index)

        unmatched_tracks = set(range(len(self.tracks)))
        matched_tracks = set()
        next_tracks = []

        for cx, cy in deduped_centers:
            fx = float(cx)
            fy = float(cy)
            gx = int(fx // cell_size)
            gy = int(fy // cell_size)
            best_track_index = None
            best_distance_sq = None
            for nx in range(gx - 1, gx + 2):
                for ny in range(gy - 1, gy + 2):
                    for track_index in track_grid.get((nx, ny), ()):
                        if track_index in matched_tracks:
                            continue
                        track = self.tracks[track_index]
                        dx = fx - float(track["x"])
                        dy = fy - float(track["y"])
                        distance_sq = dx * dx + dy * dy
                        if distance_sq > self.match_radius_sq:
                            continue
                        if best_distance_sq is None or distance_sq < best_distance_sq:
                            best_distance_sq = distance_sq
                            best_track_index = track_index

            if best_track_index is not None:
                base_track = self.tracks[best_track_index]
                next_tracks.append(
                    {
                        "id": int(base_track["id"]),
                        "x": int(cx),
                        "y": int(cy),
                        "missed": 0,
                        "age": int(base_track["age"]) + 1,
                    }
                )
                unmatched_tracks.discard(best_track_index)
                matched_tracks.add(best_track_index)
            elif len(next_tracks) < self.max_active_tracks:
                next_tracks.append(
                    {
                        "id": int(self._next_track_id),
                        "x": int(cx),
                        "y": int(cy),
                        "missed": 0,
                        "age": 0,
                    }
                )
                self._next_track_id += 1

        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            missed = int(track["missed"]) + 1
            age = int(track["age"]) + 1
            if missed > self.max_misses or age > self.max_track_age:
                continue
            next_tracks.append(
                {
                    "id": int(track["id"]),
                    "x": int(track["x"]),
                    "y": int(track["y"]),
                    "missed": missed,
                    "age": age,
                }
            )

        self.tracks = self._dedupe_tracks(next_tracks)
        if saturated:
            self._prune_tracks()

    def _dedupe_tracks(self, tracks):
        if not tracks:
            return []
        if self.dedupe_radius <= 0:
            return tracks[: self.max_active_tracks]
        tracks_sorted = sorted(
            tracks,
            key=lambda track: (
                int(track["missed"]),
                -int(track["age"]),
                int(track["id"]),
            )
        )
        kept = []
        grid = {}
        radius_sq = self.dedupe_radius * self.dedupe_radius
        cell_size = max(1.0, self.dedupe_radius)
        for track in tracks_sorted:
            fx = float(track["x"])
            fy = float(track["y"])
            gx = int(fx // cell_size)
            gy = int(fy // cell_size)
            keep = True
            for nx in range(gx - 1, gx + 2):
                for ny in range(gy - 1, gy + 2):
                    for ox, oy in grid.get((nx, ny), ()):
                        dx = fx - ox
                        dy = fy - oy
                        if dx * dx + dy * dy < radius_sq:
                            keep = False
                            break
                    if not keep:
                        break
                if not keep:
                    break
            if not keep:
                continue
            kept.append(track)
            grid.setdefault((gx, gy), []).append((fx, fy))
            if len(kept) >= self.max_active_tracks:
                break
        return kept

    def stabilize(self, centers, saturated=False):
        raw_centers = self._dedupe_centers([(int(cx), int(cy)) for cx, cy in centers], self.dedupe_radius)
        raw_count = len(raw_centers)

        if self._is_bad_frame(raw_count):
            stable = self._current_centers()
            self.count_history.append(len(stable))
            return stable, True

        self._match_and_update_tracks(raw_centers, saturated=saturated)
        self._prune_tracks()
        stabilized = self._current_centers()
        self.count_history.append(len(stabilized))
        return stabilized, False


class CpuFrameProcessor:
    def __init__(
        self,
        bbox,
        field_quad,
        working_scale=DEFAULT_WORKING_SCALE,
        detector_budget=None,
        detector_mode="legacy",
    ):
        self.bbox = bbox
        self.field_quad = field_quad
        self.roi_quad_mask = precompute_roi_quad_mask(bbox, field_quad)
        self.detector_mode = detector_mode
        self.detector_budget = detector_budget
        self.working_scale = working_scale
        self.detector = None
        if self.detector_mode == "peak":
            self.detector = PeakBallDetector(
                working_scale=working_scale,
                detector_budget=detector_budget,
            )

    @property
    def backend_name(self):
        return "cpu"

    def mask_for_frame(self, frame):
        return roi_yellow_mask_binary(frame, self.bbox, self.field_quad, quad_mask_roi=self.roi_quad_mask)

    def detect_from_mask(self, mask):
        if mask is None or not np.any(mask):
            return [], False
        if self.detector_mode == "peak" and self.detector is not None:
            return self.detector.detect(mask)
        return legacy_ball_centers_from_mask(mask, max_centers=LEGACY_MAX_CENTERS)


class CudaFrameProcessor(CpuFrameProcessor):
    """Guarded CUDA backend. Falls back to the CPU detector until a CUDA OpenCV build is installed."""

    def __init__(
        self,
        bbox,
        field_quad,
        working_scale=DEFAULT_WORKING_SCALE,
        detector_budget=None,
        detector_mode="legacy",
    ):
        if not cuda_backend_available():
            raise RuntimeError(
                "CUDA backend requested, but this OpenCV runtime does not expose a CUDA device. "
                "Install a CUDA-enabled OpenCV build or run with --backend cpu."
            )
        super().__init__(
            bbox,
            field_quad,
            working_scale=working_scale,
            detector_budget=detector_budget,
            detector_mode=detector_mode,
        )

    @property
    def backend_name(self):
        return "cuda"


def default_backend_name():
    return "cuda" if cuda_backend_available() else "cpu"


def cuda_backend_available():
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount()) > 0
    except Exception:
        return False


def create_frame_processor(backend_name, bbox, field_quad, working_scale, detector_budget, detector_mode):
    if backend_name == "cuda":
        return CudaFrameProcessor(
            bbox,
            field_quad,
            working_scale=working_scale,
            detector_budget=detector_budget,
            detector_mode=detector_mode,
        )
    return CpuFrameProcessor(
        bbox,
        field_quad,
        working_scale=working_scale,
        detector_budget=detector_budget,
        detector_mode=detector_mode,
    )


def component_radius(area):
    return int(np.clip(4.5 + np.sqrt(max(area, 1)) / 2.0, 5, 16))


def point_is_in_field_fuel_exclusion_zone(normalized_x, normalized_y):
    point = (float(normalized_x), float(normalized_y))
    for polygon in FIELD_FUEL_EXCLUSION_ZONES:
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return True
    return False


def project_fuel_points_from_centers(centers, bbox, projection_matrix, field_width, field_height):
    if not centers:
        return []
    x0, y0 = bbox[0], bbox[1]
    src = np.array([[[x0 + cx, y0 + cy]] for cx, cy in centers], dtype=np.float32)
    projected = cv2.perspectiveTransform(src, projection_matrix).reshape(-1, 2)
    points = []
    inv_fw = 1.0 / float(field_width)
    inv_fh = 1.0 / float(field_height)
    min_x = field_width * FIELD_DESTINATION_BOUNDS["top_left"][0]
    max_x = field_width * FIELD_DESTINATION_BOUNDS["top_right"][0]
    min_y = field_height * FIELD_DESTINATION_BOUNDS["top_left"][1]
    max_y = field_height * FIELD_DESTINATION_BOUNDS["bottom_left"][1]
    for px, py in projected:
        px = float(np.clip(px, min_x, max_x))
        py = float(np.clip(py, min_y, max_y))
        normalized_fx = float(np.clip(px * inv_fw, 0.0, 1.0))
        normalized_fy = float(np.clip(py * inv_fh, 0.0, 1.0))
        if point_is_in_field_fuel_exclusion_zone(normalized_fx, normalized_fy):
            continue
        normalized_x = int(round(normalized_fx * 10000))
        normalized_y = int(round(normalized_fy * 10000))
        points.append([normalized_x, normalized_y, FIELD_MAP_FUEL_RADIUS])
    return points


def project_fuel_points(frame, bbox, field_quad, projection_matrix, field_width, field_height):
    mask = roi_yellow_mask_binary(frame, bbox, field_quad)
    if mask is None or not np.any(mask):
        return []
    peaks = ball_centers_from_mask(mask)
    return project_fuel_points_from_centers(peaks, bbox, projection_matrix, field_width, field_height)


def ffmpeg_binary():
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe_binary():
    return os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"


def _parse_ffprobe_fps(value):
    if not value or value == "0/0":
        return 30.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator or 0)
        if denominator_value == 0:
            return 30.0
        return float(numerator) / denominator_value
    return float(value)


def probe_video_metadata(video_path):
    result = subprocess.run(
        [
            ffprobe_binary(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("Unable to read video stream metadata.")
    stream = streams[0]
    fps = _parse_ffprobe_fps(stream.get("avg_frame_rate"))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration = float(stream.get("duration") or 0.0)
    frame_count_raw = stream.get("nb_frames")
    if frame_count_raw and str(frame_count_raw).isdigit():
        frame_count = int(frame_count_raw)
    else:
        frame_count = int(round(duration * fps)) if duration > 0 and fps > 0 else 0
    return {
        "width": width,
        "height": height,
        "fps": fps if fps > 0 else 30.0,
        "duration": duration,
        "frame_count": max(1, frame_count),
    }


class FFmpegFrameReader:
    def __init__(self, video_path, width, height, stride=1):
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.stride = max(1, int(stride))
        self.frame_size = self.width * self.height * 3
        self.last_read_seconds = 0.0
        self._closed = False

        command = [
            ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
        ]
        if self.stride > 1:
            command.extend(["-vf", f"select=not(mod(n\\,{self.stride}))"])
        command.extend(
            [
                "-vsync",
                "0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ]
        )
        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self._proc.stdout is None:
            raise StopIteration
        started = time.perf_counter()
        chunk = self._proc.stdout.read(self.frame_size)
        self.last_read_seconds = time.perf_counter() - started
        if len(chunk) == 0:
            self.close()
            raise StopIteration
        if len(chunk) != self.frame_size:
            self.close()
            raise RuntimeError("FFmpeg returned an incomplete frame.")
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape((self.height, self.width, 3))
        return frame

    def close(self):
        if self._closed:
            return
        self._closed = True
        stderr_text = ""
        if self._proc.stdout is not None:
            self._proc.stdout.close()
        if self._proc.stderr is not None:
            stderr_text = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = self._proc.wait()
        if return_code not in (0, 255):
            raise RuntimeError(stderr_text or "FFmpeg failed while decoding video frames.")


def ffmpeg_has_encoder(name):
    try:
        result = subprocess.run(
            [ffmpeg_binary(), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False
    return name in result.stdout


def has_accessible_nvidia_gpu():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return False
    return "GPU " in result.stdout


class OverlayFrameWriter:
    def write(self, frame_rgb):
        raise NotImplementedError

    def close(self):
        return None


class OverlayFramesSink(OverlayFrameWriter):
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.frame_count = 0
        os.makedirs(self.frames_dir, exist_ok=True)

    def write(self, frame_rgb):
        success, encoded = cv2.imencode(
            ".webp",
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_WEBP_QUALITY, 80],
        )
        if not success:
            raise RuntimeError("Unable to encode overlay frame.")
        frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.webp")
        encoded.tofile(frame_path)
        self.frame_count += 1


class OverlayVideoSink(OverlayFrameWriter):
    def __init__(self, output_path, width, height, fps, prefer_nvenc=True):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(fps, 1.0))
        self.frame_count = 0

        encoder = "libx264"
        if prefer_nvenc and has_accessible_nvidia_gpu() and ffmpeg_has_encoder("h264_nvenc"):
            encoder = "h264_nvenc"

        command = [
            ffmpeg_binary(),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            encoder,
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_rgb):
        if self._proc.stdin is None:
            raise RuntimeError("Overlay video encoder is not writable.")
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
        self.frame_count += 1

    def close(self):
        stderr_text = ""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        if self._proc.stderr is not None:
            stderr_text = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = self._proc.wait()
        if return_code != 0:
            raise RuntimeError(stderr_text or "FFmpeg failed while encoding overlay video.")


def create_overlay_sink(overlay_output, session_dir, frame_width, frame_height, export_fps):
    os.makedirs(session_dir, exist_ok=True)
    if overlay_output == "frames":
        overlay_frames_dir = os.path.join(session_dir, "overlay-frames")
        return OverlayFramesSink(overlay_frames_dir), {
            "framesDirName": "overlay-frames",
            "overlayVideoFileName": None,
            "playbackMode": "frames",
        }

    overlay_video_path = os.path.join(session_dir, "overlay-video.mp4")
    return OverlayVideoSink(overlay_video_path, frame_width, frame_height, export_fps), {
        "framesDirName": None,
        "overlayVideoFileName": "overlay-video.mp4",
        "playbackMode": "video",
    }


def create_live_overlay_frame(frame, bbox, field_quad, overlay_color):
    """
    One fixed-size dot per detected fuel center. Uses the same ROI mask as the field map.
    """
    h, w = frame.shape[:2]
    mask = roi_yellow_mask_binary(frame, bbox, field_quad)
    if mask is None or not np.any(mask):
        return np.zeros((h, w, 3), dtype=np.uint8)
    centers = ball_centers_from_mask(mask, detector_mode="legacy")
    return draw_fuel_dots_full_frame(h, w, bbox, centers, overlay_color)


def write_dynamic_assets(
    video_path,
    session_dir,
    field_map_path,
    raw_data_path,
    bbox,
    field_quad,
    average_display_color,
    field_image_path,
    target_process_fps=DEFAULT_TARGET_PROCESS_FPS,
    progress_callback=None,
    progress_total_frames=None,
    progress_offset=0,
    backend_name=None,
    overlay_output="frames",
    working_scale=DEFAULT_WORKING_SCALE,
    detector_budget=None,
    max_active_tracks=DEFAULT_MAX_ACTIVE_TRACKS,
    detector_mode="legacy",
):
    metadata = probe_video_metadata(video_path)
    fps = metadata["fps"]
    frame_width = int(metadata["width"])
    frame_height = int(metadata["height"])
    bbox = analysis.clamp_bbox(bbox, frame_width, frame_height)
    if bbox is None:
        raise RuntimeError("Bounding box is outside the video frame.")
    x, y, width, height = bbox
    raw_data = np.zeros((frame_height, frame_width), dtype=np.uint32)
    raw_data_view = raw_data[y : y + height, x : x + width]

    field_image = Image.open(field_image_path).convert("RGB")
    field_width, field_height = field_image.size
    projection_matrix = cv2.getPerspectiveTransform(
        field_quad if field_quad is not None else np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[0] + bbox[2], bbox[1]],
                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                [bbox[0], bbox[1] + bbox[3]],
            ],
            dtype=np.float32,
        ),
        build_destination_quad(field_width, field_height),
    )

    frame_processor = create_frame_processor(
        backend_name or default_backend_name(),
        bbox,
        field_quad,
        working_scale=working_scale,
        detector_budget=detector_budget,
        detector_mode=detector_mode,
    )

    frame_count = 0
    field_frames = []
    total_frames = max(1, int(metadata["frame_count"]))
    progress_stride = max(1, total_frames // 150)
    stride = compute_frame_stride(fps, target_process_fps)
    export_fps = float(fps) / float(stride)
    overlay_sink, overlay_output_meta = create_overlay_sink(
        overlay_output,
        session_dir,
        frame_width,
        frame_height,
        export_fps,
    )
    temporal_stabilizer = FuelTemporalStabilizer(max_active_tracks=max_active_tracks)
    sample_weight = np.uint32(stride)
    metrics = ProcessingMetrics()

    print(f"Video FPS: {fps}")
    print(f"Target processing FPS: {target_process_fps}")
    print(f"Frame stride: {stride}")
    print(f"Effective exported FPS: {export_fps}")
    print(f"Backend: {frame_processor.backend_name}")
    print(f"Overlay output: {overlay_output}")

    if progress_callback and progress_total_frames is not None:
        progress_callback(progress_offset, progress_total_frames)

    video_read_count = 0
    frame_reader = None
    total_start = time.perf_counter()
    try:
        frame_reader = FFmpegFrameReader(video_path, frame_width, frame_height, stride=stride)
        for frame in frame_reader:
            metrics.add_time("decode", frame_reader.last_read_seconds)
            if progress_callback and progress_total_frames is not None:
                if (
                    video_read_count == 0
                    or video_read_count % progress_stride == 0
                    or video_read_count + 1 >= total_frames
                ):
                    progress_callback(progress_offset + video_read_count + 1, progress_total_frames)

            if video_read_count == 0 or video_read_count % max(int(export_fps), 1) == 0:
                print(f"Processing sampled frame {video_read_count}/{total_frames}", end="\r")

            raw_data_view += analysis.yellow_pixel_mask(frame[y : y + height, x : x + width]).astype(np.uint32) * sample_weight

            mask_start = time.perf_counter()
            mask = frame_processor.mask_for_frame(frame)
            metrics.add_time("mask", time.perf_counter() - mask_start)

            detect_start = time.perf_counter()
            raw_centers, saturated = frame_processor.detect_from_mask(mask)
            metrics.add_time("detect", time.perf_counter() - detect_start)

            stabilize_start = time.perf_counter()
            stable_centers, _bad_frame = temporal_stabilizer.stabilize(raw_centers, saturated=saturated)
            metrics.add_time("stabilize", time.perf_counter() - stabilize_start)
            metrics.add_counts(len(raw_centers), len(stable_centers), saturated)

            live_overlay = draw_fuel_dots_full_frame(
                frame_height, frame_width, bbox, stable_centers, average_display_color
            )

            project_start = time.perf_counter()
            field_frames.append(
                project_fuel_points_from_centers(
                    stable_centers, bbox, projection_matrix, field_width, field_height
                )
            )
            metrics.add_time("project", time.perf_counter() - project_start)

            encode_start = time.perf_counter()
            overlay_sink.write(live_overlay)
            metrics.add_time("encode", time.perf_counter() - encode_start)

            frame_count += 1
            video_read_count += stride
    finally:
        if frame_reader is not None:
            frame_reader.close()
        metrics.add_time("total", time.perf_counter() - total_start)
        overlay_sink.close()
    print()

    print("Saving raw data...")
    np.savetxt(raw_data_path, raw_data, fmt="%d")

    with open(field_map_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "imageWidth": field_width,
                "imageHeight": field_height,
                "fps": export_fps,
                "frameCount": frame_count,
                "frames": field_frames,
            },
            handle,
            separators=(",", ":"),
        )

    return {
        "raw_data": raw_data,
        "fps": export_fps,
        "frameCount": frame_count,
        "backend": frame_processor.backend_name,
        "overlayOutput": overlay_output,
        "overlayOutputMeta": overlay_output_meta,
        "metrics": metrics.to_stats(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a fuel-density overlay from a video.")
    parser.add_argument("--video", required=True, help="Path to the local video file")
    parser.add_argument("--session-dir", required=True, help="Directory to store generated files")
    parser.add_argument("--bbox", required=True, help="Bounding box in x,y,width,height pixels")
    parser.add_argument("--quad", help="Field quad in x1,y1,x2,y2,x3,y3,x4,y4 pixels")
    parser.add_argument("--average-display-color", default="255,0,255", help="RGB color for the average intensity point")
    parser.add_argument("--pct-from-average-to-max", type=float, default=0.5)
    parser.add_argument("--target-process-fps", type=float, default=DEFAULT_TARGET_PROCESS_FPS)
    parser.add_argument("--field-image", default=DEFAULT_FIELD_IMAGE_PATH, help="Top-down field asset for field-map projection")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default=default_backend_name())
    parser.add_argument("--overlay-output", choices=("video", "frames"), default="frames")
    parser.add_argument("--working-scale", type=float, default=DEFAULT_WORKING_SCALE)
    parser.add_argument("--detector-budget", type=int, default=0)
    parser.add_argument("--max-active-tracks", type=int, default=DEFAULT_MAX_ACTIVE_TRACKS)
    parser.add_argument("--detector-mode", choices=("legacy", "peak"), default="legacy")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    field_quad = parse_quad(args.quad) if args.quad else None
    average_display_color = tuple(int(part.strip()) for part in args.average_display_color.split(","))

    total_work = compute_total_work_units(args.video)
    emit_progress("frames", 0, total_work)

    os.makedirs(args.session_dir, exist_ok=True)

    raw_data_path = os.path.join(args.session_dir, "raw_data.txt")
    overlay_path = os.path.join(args.session_dir, "overlay.png")
    transparent_overlay_path = os.path.join(args.session_dir, "overlay-transparent.png")
    field_map_path = os.path.join(args.session_dir, "field-map.json")
    stats_path = os.path.join(args.session_dir, "stats.json")

    print("Processing video...")
    overlay_timing = write_dynamic_assets(
        args.video,
        args.session_dir,
        field_map_path,
        raw_data_path,
        bbox,
        field_quad,
        average_display_color,
        args.field_image,
        target_process_fps=args.target_process_fps,
        progress_callback=lambda processed, total: emit_progress("frames", min(processed, total_work), total_work),
        progress_total_frames=total_work,
        backend_name=args.backend,
        overlay_output=args.overlay_output,
        working_scale=args.working_scale,
        detector_budget=(args.detector_budget or None),
        max_active_tracks=args.max_active_tracks,
        detector_mode=args.detector_mode,
    )
    raw_data = overlay_timing.pop("raw_data")

    emit_progress("encode", total_work, total_work)

    max_value = int(raw_data.max())
    non_zero_values = raw_data[raw_data > 0]
    actual_average = float(non_zero_values.mean()) if non_zero_values.size else 0.0
    average_of_non_zero_values = actual_average + args.pct_from_average_to_max * (max_value - actual_average)

    print("Creating overlay images...")
    color_array = create_color_array(raw_data, max_value, average_of_non_zero_values, average_display_color)
    Image.fromarray(color_array).save(overlay_path)

    alpha_channel = np.where(np.any(color_array != 0, axis=2), 220, 0).astype(np.uint8)
    transparent_image = np.dstack((color_array, alpha_channel))
    Image.fromarray(transparent_image).save(transparent_overlay_path)

    stats = {
        "backend": overlay_timing["backend"],
        "bbox": {
            "x": bbox[0],
            "y": bbox[1],
            "width": bbox[2],
            "height": bbox[3],
        },
        "maxValue": max_value,
        "actualAverage": actual_average,
        "weightedAverage": average_of_non_zero_values,
        "nonZeroPixels": int(non_zero_values.size),
        "overlayFps": overlay_timing["fps"],
        "overlayFrameCount": overlay_timing["frameCount"],
        **overlay_timing["metrics"],
    }

    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print("Finished.")


if __name__ == "__main__":
    main()
