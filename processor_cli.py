import argparse
import builtins
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
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
DEFAULT_FUEL_BASE_COLOR_RGB = analysis.DEFAULT_FUEL_BASE_COLOR_RGB

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

AIR_PROFILE_EXPORT_SIZE = 1000
AIR_PROFILE_GROUND_MARGIN = 0.02
AIR_PROFILE_HEIGHT_HEADROOM = 3.5
AIR_PROFILE_MAX_SUPPORTED_DEPTH = 0.62
AIR_PROFILE_DEPTH_SUPPORT_POWER = 1.75
AIR_PROFILE_FRAME_MIN_ACTIVE_HEIGHT = 500
AIR_PROFILE_FRAME_MIN_STRONG_HEIGHT = 1500
AIR_PROFILE_FRAME_MIN_ACTIVE_POINTS = 2
AIR_PROFILE_POINT_MIN_HEIGHT = 300
AIR_PROFILE_HEIGHT_GAIN = 3.0
AIR_PROFILE_DEFAULT_CONTACT_RADIUS = 6.0
AIR_PROFILE_CONTACT_RADIUS_MIN = 3.0
AIR_PROFILE_CONTACT_RADIUS_MAX = 18.0
AIR_PROFILE_WALL_VALIDATION_SECONDS = 20.0
AIR_PROFILE_WALL_VALIDATION_MIN_FRAMES = 120
AIR_PROFILE_WALL_INVALID_FRACTION = 0.9
AIR_PROFILE_WALL_INVALID_MEDIAN_ACTIVE = 12

FUEL_TRACK_MATCH_RADIUS_PX = 14.0
FUEL_TRACK_MAX_MISSES = 2
FUEL_TRACK_SMOOTHING_ALPHA = 0.45
FUEL_TRACK_VELOCITY_ALPHA = 0.35
FUEL_TRACK_VELOCITY_DECAY = 0.7
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


def mask_has_signal(mask):
    return mask is not None and mask.size > 0 and int(np.count_nonzero(mask)) > 0


def parse_rgb_color(value):
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("RGB color must be r,g,b")
    red, green, blue = parts
    return (
        int(np.clip(red, 0, 255)),
        int(np.clip(green, 0, 255)),
        int(np.clip(blue, 0, 255)),
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


def quad_mask(bbox, quads):
    x, y, _, _ = bbox
    mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
    if isinstance(quads, np.ndarray):
        quads = [quads]
    for quad in quads:
        if quad is None:
            continue
        local_quad = np.round(quad - np.array([x, y], dtype=np.float32)).astype(np.int32)
        local_quad[:, 0] = np.clip(local_quad[:, 0], 0, bbox[2] - 1)
        local_quad[:, 1] = np.clip(local_quad[:, 1], 0, bbox[3] - 1)
        cv2.fillConvexPoly(mask, local_quad, 255)
    return mask


def roi_quads_for_detection(field_quad, wall_quads=None):
    quads = []
    if field_quad is not None:
        quads.append(field_quad)
    if wall_quads:
        quads.extend(quad for quad in wall_quads.values() if quad is not None)
    return quads


def bbox_from_quads_pixels(quads, frame_width, frame_height, padding_px=6):
    valid_quads = [quad for quad in quads if quad is not None and getattr(quad, "size", 0) > 0]
    if not valid_quads:
        return None
    stacked = np.vstack(valid_quads).astype(np.float32)
    min_x = max(0, int(np.floor(float(np.min(stacked[:, 0]))) - padding_px))
    min_y = max(0, int(np.floor(float(np.min(stacked[:, 1]))) - padding_px))
    max_x = min(int(frame_width), int(np.ceil(float(np.max(stacked[:, 0]))) + padding_px))
    max_y = min(int(frame_height), int(np.ceil(float(np.max(stacked[:, 1]))) + padding_px))
    if max_x <= min_x or max_y <= min_y:
        return None
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def precompute_roi_quad_mask(bbox, field_quad, wall_quads=None):
    """Quad intersection mask depends only on bbox + quad, not on video frame."""
    quads = roi_quads_for_detection(field_quad, wall_quads)
    if not quads:
        return None
    return quad_mask(bbox, quads)


def roi_yellow_mask_binary(frame, bbox, field_quad, quad_mask_roi=None, fuel_base_color_rgb=None, wall_quads=None):
    """HSV yellow mask cropped to the ROI and field quad, with light noise removal."""
    x, y, width, height = bbox
    frame_slice = frame[y : y + height, x : x + width]
    if frame_slice.size == 0:
        return None
    mask = analysis.fuel_pixel_mask_hsv(frame_slice, base_color_rgb=fuel_base_color_rgb)
    roi_quads = roi_quads_for_detection(field_quad, wall_quads)
    if roi_quads:
        qm = quad_mask_roi if quad_mask_roi is not None else quad_mask(bbox, roi_quads)
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
        if not mask_has_signal(mask_work):
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
        if not mask_has_signal(mask_binary):
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
            if not mask_has_signal(patch):
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
    lm = (np.isclose(dist, dil, rtol=0, atol=1e-3) & (dist >= 0.5) & (component_mask > 0)).astype(np.uint8)
    n_labels, labels, _, centroids = cv2.connectedComponentsWithStats(lm)
    if n_labels <= 1:
        return []

    candidates = []
    for label_idx in range(1, n_labels):
        ys, xs = np.where(labels == label_idx)
        if len(xs) == 0:
            continue
        strengths = dist[ys, xs]
        peak_idx = int(np.argmax(strengths))
        peak_strength = float(strengths[peak_idx])
        centroid_x, centroid_y = centroids[label_idx]
        px = float(xs[peak_idx])
        py = float(ys[peak_idx])
        # For narrow plateaus, the connected-component centroid is more stable than any one pixel.
        if len(xs) > 1:
            px = float(centroid_x)
            py = float(centroid_y)
        candidates.append((peak_strength, px, py))

    if not candidates:
        return []

    candidates.sort(key=lambda item: -item[0])
    sep2 = min_sep * min_sep
    peaks = []
    for _strength, px, py in candidates:
        if len(peaks) >= budget:
            break
        if all((px - ux) ** 2 + (py - uy) ** 2 >= sep2 for ux, uy in peaks):
            peaks.append((px, py))
    return peaks


def _should_prefer_legacy_split(peak_centers, legacy_centers):
    peak_count = len(peak_centers)
    legacy_count = len(legacy_centers)
    if legacy_count <= 0:
        return False
    if peak_count <= 0:
        return True
    return legacy_count >= max(peak_count + 2, int(math.ceil(peak_count * 1.5)))


def legacy_ball_centers_from_mask(mask_binary, max_centers=LEGACY_MAX_CENTERS):
    mask_work = (mask_binary > 0).astype(np.uint8) * 255
    if not mask_has_signal(mask_work):
        return [], False

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_work)
    if n_labels <= 1:
        return [], False

    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)])
    sorted_areas = np.sort(areas)
    bottom_n = max(1, len(sorted_areas) * 40 // 100)
    single_ball_area = float(np.clip(np.median(sorted_areas[:bottom_n]), 6, 240))
    min_sep = max(1.5, np.sqrt(single_ball_area) * 0.28)

    centers = []
    saturated = False
    for i in range(1, n_labels):
        if len(centers) >= max_centers:
            saturated = True
            break

        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        component_mask = (labels == i).astype(np.uint8) * 255
        peak_candidates = _dt_peaks_in_component(component_mask, max_centers, min_sep)
        estimated_from_area = max(1, round(area / single_ball_area))
        estimated_balls = max(estimated_from_area, len(peak_candidates))

        if estimated_balls == 1:
            centers.append((int(cx), int(cy)))
            continue

        peaks = peak_candidates[:estimated_balls]

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
    detector_mode="hybrid",
):
    if detector_mode in {"peak", "hybrid"}:
        detector = PeakBallDetector(working_scale=working_scale, detector_budget=detector_budget)
        peak_centers, _ = detector.detect(mask_binary)
        if detector_mode == "peak":
            if max_centers is not None:
                return peak_centers[: int(max_centers)]
            return peak_centers
        legacy_centers, _ = legacy_ball_centers_from_mask(
            mask_binary,
            max_centers=int(max_centers or LEGACY_MAX_CENTERS),
        )
        if _should_prefer_legacy_split(peak_centers, legacy_centers):
            return legacy_centers
        if max_centers is not None:
            return peak_centers[: int(max_centers)]
        return peak_centers

    legacy_centers, _ = legacy_ball_centers_from_mask(mask_binary, max_centers=int(max_centers or LEGACY_MAX_CENTERS))
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
        smoothing_alpha=FUEL_TRACK_SMOOTHING_ALPHA,
        velocity_alpha=FUEL_TRACK_VELOCITY_ALPHA,
        velocity_decay=FUEL_TRACK_VELOCITY_DECAY,
    ):
        to_int = builtins.int
        to_float = builtins.float
        self.match_radius = to_float(match_radius_px)
        self.match_radius_sq = self.match_radius * self.match_radius
        self.max_misses = to_int(max_misses)
        self.min_baseline_count = to_int(min_baseline_count)
        self.min_count_delta = to_int(min_count_delta)
        self.count_delta_ratio = to_float(count_delta_ratio)
        self.count_history = deque(maxlen=max(1, to_int(history_size)))
        self.max_active_tracks = max(1, to_int(max_active_tracks))
        self.max_track_age = max(1, to_int(max_track_age))
        if dedupe_radius_px is None:
            self.dedupe_radius = 0.0
        else:
            self.dedupe_radius = max(0.0, to_float(dedupe_radius_px))
        self.smoothing_alpha = to_float(np.clip(smoothing_alpha, 0.01, 1.0))
        self.velocity_alpha = to_float(np.clip(velocity_alpha, 0.01, 1.0))
        self.velocity_decay = to_float(np.clip(velocity_decay, 0.0, 1.0))
        self._next_track_id = 1
        self.tracks = []

    def _coerce_point(self, x_value, y_value):
        to_float = builtins.float
        # Some detector outputs may carry metadata like ((x, y), score).
        if isinstance(x_value, (tuple, list, np.ndarray)) and len(x_value) == 2:
            x_value, y_value = x_value
        elif isinstance(y_value, (tuple, list, np.ndarray)) and len(y_value) == 2:
            x_value, y_value = y_value
        try:
            fx = to_float(x_value)
            fy = to_float(y_value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(fx) or not np.isfinite(fy):
            return None
        return fx, fy

    def _baseline_count(self):
        to_float = builtins.float
        if not self.count_history:
            return None
        return to_float(np.median(np.array(self.count_history, dtype=np.float32)))

    def _is_bad_frame(self, raw_count):
        to_int = builtins.int
        baseline = self._baseline_count()
        if baseline is None or baseline < self.min_baseline_count:
            return False
        allowed_delta = max(self.min_count_delta, to_int(round(baseline * self.count_delta_ratio)))
        return abs(to_int(raw_count) - to_int(round(baseline))) >= allowed_delta

    def _current_centers(self):
        to_int = builtins.int
        centers = []
        for track in self.tracks:
            point = self._coerce_point(track.get("x"), track.get("y"))
            if point is None:
                continue
            centers.append((to_int(point[0]), to_int(point[1])))
        return centers

    def _make_track(self, center, age=0):
        to_int = builtins.int
        to_float = builtins.float
        cx, cy = center
        return {
            "id": to_int(self._next_track_id),
            "x": to_float(cx),
            "y": to_float(cy),
            "vx": 0.0,
            "vy": 0.0,
            "missed": 0,
            "age": to_int(age),
        }

    def _predicted_position(self, track):
        to_float = builtins.float
        point = self._coerce_point(track.get("x"), track.get("y"))
        if point is None:
            return None
        base_x, base_y = point
        try:
            vx = to_float(track.get("vx", 0.0))
            vy = to_float(track.get("vy", 0.0))
        except (TypeError, ValueError):
            vx = 0.0
            vy = 0.0
        if not np.isfinite(vx):
            vx = 0.0
        if not np.isfinite(vy):
            vy = 0.0
        return (base_x + vx, base_y + vy)

    def _dedupe_centers(self, centers, radius):
        to_int = builtins.int
        to_float = builtins.float
        if not centers:
            return []
        normalized_points = []
        append_normalized_point = normalized_points.append
        for center in centers:
            try:
                cx, cy = center
            except (TypeError, ValueError):
                continue
            point = self._coerce_point(cx, cy)
            if point is None:
                continue
            fx, fy = point
            append_normalized_point((to_int(round(fx)), to_int(round(fy))))
        if not normalized_points:
            return []
        if radius <= 0:
            return normalized_points
        cell_size = max(1.0, to_float(radius))
        radius_sq = to_float(radius) * to_float(radius)
        grid = {}
        deduped_points = []
        append_deduped_point = deduped_points.append
        for cx, cy in normalized_points:
            fx = to_float(cx)
            fy = to_float(cy)
            gx = to_int(fx // cell_size)
            gy = to_int(fy // cell_size)
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
            append_deduped_point((to_int(round(fx)), to_int(round(fy))))
            grid.setdefault((gx, gy), []).append((fx, fy))
        return deduped_points

    def _prune_tracks(self):
        to_int = builtins.int
        if len(self.tracks) <= self.max_active_tracks:
            return
        self.tracks.sort(
            key=lambda track: (
                to_int(track["missed"]),
                -to_int(track["age"]),
                to_int(track["id"]),
            )
        )
        self.tracks = self.tracks[: self.max_active_tracks]

    def _match_and_update_tracks(self, centers, saturated=False):
        to_int = builtins.int
        to_float = builtins.float
        dedupe_radius = self.dedupe_radius
        tracks = self.tracks
        next_track_id = to_int(self._next_track_id)
        max_active_tracks = self.max_active_tracks
        max_misses = self.max_misses
        max_track_age = self.max_track_age
        match_radius = self.match_radius
        match_radius_sq = self.match_radius_sq
        smoothing_alpha = self.smoothing_alpha
        velocity_alpha = self.velocity_alpha
        velocity_decay = self.velocity_decay
        coerce_point = self._coerce_point
        predicted_position = self._predicted_position
        make_track = self._make_track

        deduped_centers = self._dedupe_centers(centers, dedupe_radius)
        if not tracks:
            tracks = []
            for cx, cy in deduped_centers[:max_active_tracks]:
                tracks.append(make_track((cx, cy)))
                next_track_id += 1
            self._next_track_id = next_track_id
            self.tracks = tracks
            return

        cell_size = max(1.0, match_radius)
        track_predictions = {}
        track_grid = {}
        for track_index, track in enumerate(tracks):
            predicted = predicted_position(track)
            if predicted is None:
                continue
            track_predictions[track_index] = predicted
            pred_x, pred_y = predicted
            gx = to_int(pred_x // cell_size)
            gy = to_int(pred_y // cell_size)
            track_grid.setdefault((gx, gy), []).append(track_index)

        unmatched_tracks = set(range(len(tracks)))
        matched_tracks = set()
        next_tracks = []

        for cx, cy in deduped_centers:
            center_point = coerce_point(cx, cy)
            if center_point is None:
                continue
            center_x, center_y = center_point
            gx = to_int(center_x // cell_size)
            gy = to_int(center_y // cell_size)
            best_track_index = None
            best_distance_sq = None
            for nx in range(gx - 1, gx + 2):
                for ny in range(gy - 1, gy + 2):
                    for track_index in track_grid.get((nx, ny), ()):
                        if track_index in matched_tracks:
                            continue
                        predicted_point = track_predictions.get(track_index)
                        if predicted_point is None:
                            continue
                        pred_x, pred_y = predicted_point
                        dx = center_x - pred_x
                        dy = center_y - pred_y
                        distance_sq = dx * dx + dy * dy
                        if distance_sq > match_radius_sq:
                            continue
                        if best_distance_sq is None or distance_sq < best_distance_sq:
                            best_distance_sq = distance_sq
                            best_track_index = track_index

            if best_track_index is not None:
                base_track = tracks[best_track_index]
                previous = coerce_point(base_track.get("x"), base_track.get("y"))
                if previous is None:
                    if len(next_tracks) < max_active_tracks:
                        next_tracks.append(make_track((cx, cy)))
                        next_track_id += 1
                    unmatched_tracks.discard(best_track_index)
                    matched_tracks.add(best_track_index)
                    continue
                prev_x, prev_y = previous
                measured_dx = center_x - prev_x
                measured_dy = center_y - prev_y
                vx = (1.0 - velocity_alpha) * to_float(base_track.get("vx", 0.0)) + velocity_alpha * measured_dx
                vy = (1.0 - velocity_alpha) * to_float(base_track.get("vy", 0.0)) + velocity_alpha * measured_dy
                next_x = prev_x + smoothing_alpha * measured_dx
                next_y = prev_y + smoothing_alpha * measured_dy
                next_tracks.append(
                    {
                        "id": to_int(base_track["id"]),
                        "x": next_x,
                        "y": next_y,
                        "vx": vx,
                        "vy": vy,
                        "missed": 0,
                        "age": to_int(base_track["age"]) + 1,
                    }
                )
                unmatched_tracks.discard(best_track_index)
                matched_tracks.add(best_track_index)
            elif len(next_tracks) < max_active_tracks:
                next_tracks.append(make_track((cx, cy)))
                next_track_id += 1

        for track_index in unmatched_tracks:
            track = tracks[track_index]
            missed = to_int(track["missed"]) + 1
            age = to_int(track["age"]) + 1
            if missed > max_misses or age > max_track_age:
                continue
            predicted = predicted_position(track)
            if predicted is None:
                continue
            next_x, next_y = predicted
            next_vx = to_float(track.get("vx", 0.0)) * velocity_decay
            next_vy = to_float(track.get("vy", 0.0)) * velocity_decay
            next_tracks.append(
                {
                    "id": to_int(track["id"]),
                    "x": next_x,
                    "y": next_y,
                    "vx": next_vx,
                    "vy": next_vy,
                    "missed": missed,
                    "age": age,
                }
            )

        self._next_track_id = next_track_id
        self.tracks = self._dedupe_tracks(next_tracks)
        if saturated:
            self._prune_tracks()

    def _dedupe_tracks(self, tracks):
        to_int = builtins.int
        to_float = builtins.float
        if not tracks:
            return []
        if self.dedupe_radius <= 0:
            return tracks[: self.max_active_tracks]
        tracks_sorted = sorted(
            tracks,
            key=lambda track: (
                to_int(track["missed"]),
                -to_int(track["age"]),
                to_int(track["id"]),
            )
        )
        kept = []
        grid = {}
        radius_sq = self.dedupe_radius * self.dedupe_radius
        cell_size = max(1.0, self.dedupe_radius)
        for track in tracks_sorted:
            fx = to_float(track["x"])
            fy = to_float(track["y"])
            gx = to_int(fx // cell_size)
            gy = to_int(fy // cell_size)
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
        raw_centers = self._dedupe_centers(centers, self.dedupe_radius)
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
        wall_quads=None,
        fuel_base_color_rgb=DEFAULT_FUEL_BASE_COLOR_RGB,
        working_scale=DEFAULT_WORKING_SCALE,
        detector_budget=None,
        detector_mode="hybrid",
    ):
        self.bbox = bbox
        self.field_quad = field_quad
        self.wall_quads = dict(wall_quads or {})
        self.fuel_base_color_rgb = fuel_base_color_rgb
        self.roi_quad_mask = precompute_roi_quad_mask(bbox, field_quad, self.wall_quads)
        self.detector_mode = detector_mode
        self.detector_budget = detector_budget
        self.working_scale = working_scale
        self.detector = None
        if self.detector_mode in {"peak", "hybrid"}:
            self.detector = PeakBallDetector(
                working_scale=working_scale,
                detector_budget=detector_budget,
            )

    @property
    def backend_name(self):
        return "cpu"

    def mask_for_frame(self, frame):
        return roi_yellow_mask_binary(
            frame,
            self.bbox,
            self.field_quad,
            quad_mask_roi=self.roi_quad_mask,
            fuel_base_color_rgb=self.fuel_base_color_rgb,
            wall_quads=self.wall_quads,
        )

    def detect_from_mask(self, mask):
        if not mask_has_signal(mask):
            return [], False
        if self.detector_mode == "peak" and self.detector is not None:
            peak_centers, peak_saturated = self.detector.detect(mask)
            return peak_centers, peak_saturated
        if self.detector_mode == "hybrid" and self.detector is not None:
            peak_centers, peak_saturated = self.detector.detect(mask)
            legacy_centers, legacy_saturated = legacy_ball_centers_from_mask(mask, max_centers=LEGACY_MAX_CENTERS)
            if _should_prefer_legacy_split(peak_centers, legacy_centers):
                return legacy_centers, legacy_saturated
            return peak_centers, peak_saturated
        return legacy_ball_centers_from_mask(mask, max_centers=LEGACY_MAX_CENTERS)


class CudaFrameProcessor(CpuFrameProcessor):
    """Guarded CUDA backend. Falls back to the CPU detector until a CUDA OpenCV build is installed."""

    def __init__(
        self,
        bbox,
        field_quad,
        wall_quads=None,
        fuel_base_color_rgb=DEFAULT_FUEL_BASE_COLOR_RGB,
        working_scale=DEFAULT_WORKING_SCALE,
        detector_budget=None,
        detector_mode="hybrid",
    ):
        if not cuda_backend_available():
            raise RuntimeError(
                "CUDA backend requested, but this OpenCV runtime does not expose a CUDA device. "
                "Install a CUDA-enabled OpenCV build or run with --backend cpu."
            )
        super().__init__(
            bbox,
            field_quad,
            wall_quads=wall_quads,
            fuel_base_color_rgb=fuel_base_color_rgb,
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


def create_frame_processor(
    backend_name,
    bbox,
    field_quad,
    wall_quads,
    fuel_base_color_rgb,
    working_scale,
    detector_budget,
    detector_mode,
):
    if backend_name == "cuda":
        return CudaFrameProcessor(
            bbox,
            field_quad,
            wall_quads=wall_quads,
            fuel_base_color_rgb=fuel_base_color_rgb,
            working_scale=working_scale,
            detector_budget=detector_budget,
            detector_mode=detector_mode,
        )
    return CpuFrameProcessor(
        bbox,
        field_quad,
        wall_quads=wall_quads,
        fuel_base_color_rgb=fuel_base_color_rgb,
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


def field_projection_bounds(field_width, field_height):
    min_x = field_width * FIELD_DESTINATION_BOUNDS["top_left"][0]
    max_x = field_width * FIELD_DESTINATION_BOUNDS["top_right"][0]
    min_y = field_height * FIELD_DESTINATION_BOUNDS["top_left"][1]
    max_y = field_height * FIELD_DESTINATION_BOUNDS["bottom_left"][1]
    return min_x, max_x, min_y, max_y


def normalize_between(value, min_value, max_value):
    span = max(float(max_value) - float(min_value), 1e-6)
    return float(np.clip((float(value) - float(min_value)) / span, 0.0, 1.0))


def infer_wall_side(field_quad, wall_quad):
    if field_quad is None or wall_quad is None:
        return None
    wall_baseline_a = np.asarray(wall_quad[2], dtype=np.float32)
    wall_baseline_b = np.asarray(wall_quad[3], dtype=np.float32)
    wall_midpoint = (wall_baseline_a + wall_baseline_b) * 0.5
    edges = (
        ("top", np.asarray(field_quad[0], dtype=np.float32), np.asarray(field_quad[1], dtype=np.float32)),
        ("right", np.asarray(field_quad[1], dtype=np.float32), np.asarray(field_quad[2], dtype=np.float32)),
        ("bottom", np.asarray(field_quad[2], dtype=np.float32), np.asarray(field_quad[3], dtype=np.float32)),
        ("left", np.asarray(field_quad[3], dtype=np.float32), np.asarray(field_quad[0], dtype=np.float32)),
    )
    best_name = None
    best_score = None
    for name, edge_a, edge_b in edges:
        direct = np.linalg.norm(wall_baseline_a - edge_a) + np.linalg.norm(wall_baseline_b - edge_b)
        swapped = np.linalg.norm(wall_baseline_a - edge_b) + np.linalg.norm(wall_baseline_b - edge_a)
        edge_midpoint = (edge_a + edge_b) * 0.5
        score = float(min(direct, swapped) + (0.5 * np.linalg.norm(wall_midpoint - edge_midpoint)))
        if best_score is None or score < best_score:
            best_score = score
            best_name = name
    return best_name


def build_wall_destination_quad(size=AIR_PROFILE_EXPORT_SIZE):
    size = float(size)
    return np.array(
        [
            [0.0, 0.0],
            [size, 0.0],
            [size, size],
            [0.0, size],
        ],
        dtype=np.float32,
    )


def estimate_contact_radius(distance_transform, cx, cy):
    if distance_transform is None or distance_transform.size == 0:
        return AIR_PROFILE_DEFAULT_CONTACT_RADIUS
    ix = int(round(cx))
    iy = int(round(cy))
    x0 = max(0, ix - 2)
    x1 = min(distance_transform.shape[1], ix + 3)
    y0 = max(0, iy - 2)
    y1 = min(distance_transform.shape[0], iy + 3)
    patch = distance_transform[y0:y1, x0:x1]
    if patch.size == 0:
        radius = AIR_PROFILE_DEFAULT_CONTACT_RADIUS
    else:
        radius = float(patch.max())
        if radius <= 0:
            radius = AIR_PROFILE_DEFAULT_CONTACT_RADIUS
    return float(np.clip(radius, AIR_PROFILE_CONTACT_RADIUS_MIN, AIR_PROFILE_CONTACT_RADIUS_MAX))


def project_points(points, projection_matrix):
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    src = np.array([[[float(px), float(py)]] for px, py in points], dtype=np.float32)
    return cv2.perspectiveTransform(src, projection_matrix).reshape(-1, 2)


def relative_height_from_wall_y(projected_y, wall_height, ground_margin=AIR_PROFILE_GROUND_MARGIN):
    wall_height = max(float(wall_height), 1.0)
    # Keep several wall-heights of headroom so airborne points spread across low/mid/high
    # instead of all saturating to "high" as soon as they project above the wall top.
    baseline_relative = (wall_height - float(projected_y)) / (wall_height * AIR_PROFILE_HEIGHT_HEADROOM)
    baseline_relative = float(np.clip(baseline_relative, 0.0, 1.0))
    if baseline_relative <= ground_margin:
        return 0.0
    return float(np.clip((baseline_relative - ground_margin) / max(1.0 - ground_margin, 1e-6), 0.0, 1.0))


def depth_from_field_projection(projected_x, projected_y, field_width, field_height, wall_side):
    min_x, max_x, min_y, max_y = field_projection_bounds(field_width, field_height)
    field_x = normalize_between(projected_x, min_x, max_x)
    field_y = normalize_between(projected_y, min_y, max_y)
    if wall_side == "bottom":
        return 1.0 - field_y
    if wall_side == "left":
        return field_x
    if wall_side == "right":
        return 1.0 - field_x
    return field_y


def wall_support_from_depth(depth_norm, max_supported_depth=AIR_PROFILE_MAX_SUPPORTED_DEPTH):
    max_supported_depth = max(float(max_supported_depth), 1e-6)
    closeness = float(np.clip(1.0 - (float(depth_norm) / max_supported_depth), 0.0, 1.0))
    return float(closeness ** AIR_PROFILE_DEPTH_SUPPORT_POWER)


def build_wall_references(field_quad, wall_quads=None, legacy_wall_quad=None):
    wall_references = []
    if wall_quads:
        for side, quad in wall_quads.items():
            if quad is None:
                continue
            wall_references.append(
                {
                    "side": side,
                    "projection_matrix": cv2.getPerspectiveTransform(quad, build_wall_destination_quad()),
                }
            )
    elif legacy_wall_quad is not None and field_quad is not None:
        legacy_side = infer_wall_side(field_quad, legacy_wall_quad)
        if legacy_side is not None:
            wall_references.append(
                {
                    "side": legacy_side,
                    "projection_matrix": cv2.getPerspectiveTransform(legacy_wall_quad, build_wall_destination_quad()),
                }
            )
    return wall_references


def project_synchronized_field_and_air_points(
    centers,
    bbox,
    field_projection_matrix,
    field_width,
    field_height,
    mask_binary=None,
    wall_references=None,
    wall_projection_matrix=None,
    wall_side=None,
    wall_height=AIR_PROFILE_EXPORT_SIZE,
):
    if not centers or field_projection_matrix is None:
        return [], []
    x0, y0 = bbox[0], bbox[1]
    center_points = [(x0 + float(cx), y0 + float(cy)) for cx, cy in centers]
    projected_field_points = project_points(center_points, field_projection_matrix)

    if wall_references is None:
        wall_references = []
        if wall_projection_matrix is not None and wall_side is not None:
            wall_references.append(
                {
                    "side": wall_side,
                    "projection_matrix": wall_projection_matrix,
                }
            )

    include_air = bool(wall_references) and mask_binary is not None and mask_binary.size > 0
    distance_transform = None
    bottom_points = []
    projected_wall_points_by_side = {}
    if include_air:
        distance_transform = cv2.distanceTransform((mask_binary > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        for cx, cy in centers:
            radius = estimate_contact_radius(distance_transform, cx, cy)
            bottom_points.append((x0 + float(cx), y0 + float(cy) + radius))
        for wall_reference in wall_references:
            projected_wall_points_by_side[wall_reference["side"]] = project_points(
                bottom_points,
                wall_reference["projection_matrix"],
            )

    field_points = []
    air_points = []
    wall_reference_by_side = {reference["side"]: reference for reference in wall_references}
    inv_fw = 1.0 / float(field_width)
    inv_fh = 1.0 / float(field_height)
    min_x, max_x, min_y, max_y = field_projection_bounds(field_width, field_height)
    for point_index, (field_px, field_py) in enumerate(projected_field_points):
        field_px = float(np.clip(field_px, min_x, max_x))
        field_py = float(np.clip(field_py, min_y, max_y))
        normalized_fx = float(np.clip(field_px * inv_fw, 0.0, 1.0))
        normalized_fy = float(np.clip(field_py * inv_fh, 0.0, 1.0))
        if point_is_in_field_fuel_exclusion_zone(normalized_fx, normalized_fy):
            continue
        field_points.append(
            [
                int(round(normalized_fx * 10000)),
                int(round(normalized_fy * 10000)),
                FIELD_MAP_FUEL_RADIUS,
            ]
        )
        if include_air:
            candidate_wall_references = wall_references
            if len(wall_references) > 1:
                if normalized_fx <= 0.5 and "left" in wall_reference_by_side:
                    candidate_wall_references = [wall_reference_by_side["left"]]
                elif normalized_fx > 0.5 and "right" in wall_reference_by_side:
                    candidate_wall_references = [wall_reference_by_side["right"]]
            nearest_wall_reference = None
            nearest_depth = None
            best_supported_reference = None
            best_supported_depth = None
            best_supported_weight = 0.0
            for wall_reference in candidate_wall_references:
                current_depth = depth_from_field_projection(
                    field_px,
                    field_py,
                    field_width,
                    field_height,
                    wall_reference["side"],
                )
                if nearest_depth is None or current_depth < nearest_depth:
                    nearest_depth = current_depth
                    nearest_wall_reference = wall_reference
                current_support_weight = wall_support_from_depth(current_depth)
                if current_support_weight <= 0.0:
                    continue
                if current_depth >= AIR_PROFILE_MAX_SUPPORTED_DEPTH:
                    continue
                if (
                    current_support_weight > best_supported_weight
                    or (
                        math.isclose(current_support_weight, best_supported_weight)
                        and (best_supported_depth is None or current_depth < best_supported_depth)
                    )
                ):
                    best_supported_reference = wall_reference
                    best_supported_depth = current_depth
                    best_supported_weight = current_support_weight

            chosen_reference = best_supported_reference or nearest_wall_reference
            chosen_depth = best_supported_depth if best_supported_reference is not None else nearest_depth
            relative_height = 0.0
            depth = float(np.clip(chosen_depth if chosen_depth is not None else 0.0, 0.0, 1.0))
            if best_supported_reference is not None and chosen_reference is not None:
                _wall_px, wall_py = projected_wall_points_by_side[chosen_reference["side"]][point_index]
                base_relative_height = relative_height_from_wall_y(wall_py, wall_height)
                relative_height = base_relative_height * float(np.clip(best_supported_weight, 0.0, 1.0))
                if base_relative_height <= 0.0:
                    relative_height = 0.0
            air_points.append(
                [
                    int(round(normalized_fx * 10000)),
                    int(round(float(np.clip(relative_height, 0.0, 1.0)) * 10000)),
                ]
            )
    return field_points, air_points


def build_air_profile_frame(
    centers,
    bbox,
    mask_binary,
    field_projection_matrix,
    wall_projection_matrix,
    field_width,
    field_height,
    wall_side,
    wall_height=AIR_PROFILE_EXPORT_SIZE,
):
    _, air_points = project_synchronized_field_and_air_points(
        centers,
        bbox,
        field_projection_matrix,
        field_width,
        field_height,
        mask_binary=mask_binary,
        wall_projection_matrix=wall_projection_matrix,
        wall_side=wall_side,
        wall_height=wall_height,
    )
    return air_points


def filter_air_profile_points(
    air_points,
    min_active_height=AIR_PROFILE_FRAME_MIN_ACTIVE_HEIGHT,
    min_strong_height=AIR_PROFILE_FRAME_MIN_STRONG_HEIGHT,
    min_active_points=AIR_PROFILE_FRAME_MIN_ACTIVE_POINTS,
    point_min_height=AIR_PROFILE_POINT_MIN_HEIGHT,
    height_gain=AIR_PROFILE_HEIGHT_GAIN,
):
    if not air_points:
        return []
    active_count = sum(1 for _depth, height in air_points if int(height) >= int(min_active_height))
    max_height = max(int(height) for _depth, height in air_points)
    if active_count < int(min_active_points) and max_height < int(min_strong_height):
        return [[int(depth), 0] for depth, _height in air_points]

    filtered = []
    for depth, height in air_points:
        height = int(height)
        if height < int(point_min_height):
            filtered.append([int(depth), 0])
            continue
        scaled_height = int(round(min(10000, height * float(height_gain))))
        filtered.append([int(depth), scaled_height])
    return filtered


def dedupe_air_centers(centers, radius_px=8.0):
    stabilizer = FuelTemporalStabilizer(dedupe_radius_px=radius_px)
    return stabilizer._dedupe_centers(centers, stabilizer.dedupe_radius)


def validate_wall_side(
    video_path,
    metadata,
    bbox,
    field_quad,
    side,
    quad,
    projection_matrix,
    field_width,
    field_height,
    fuel_base_color_rgb,
    working_scale,
    detector_budget,
    detector_mode,
    backend_name,
    validation_seconds=AIR_PROFILE_WALL_VALIDATION_SECONDS,
    min_frames=AIR_PROFILE_WALL_VALIDATION_MIN_FRAMES,
):
    if quad is None or field_quad is None:
        return True, {"reason": "missing-quad"}

    frame_width = int(metadata["width"])
    frame_height = int(metadata["height"])
    fps = float(metadata["fps"])
    stride = compute_frame_stride(fps, DEFAULT_TARGET_PROCESS_FPS)
    validation_frames = max(int(min_frames), int(round(validation_seconds * (fps / max(stride, 1)))))
    validation_bbox = bbox_from_quads_pixels(
        roi_quads_for_detection(field_quad, {side: quad}),
        frame_width,
        frame_height,
        padding_px=6,
    )
    validation_bbox = analysis.clamp_bbox(validation_bbox, frame_width, frame_height)
    if validation_bbox is None:
        return False, {"reason": "empty-bbox"}

    wall_reference = build_wall_references(field_quad, {side: quad})
    if not wall_reference:
        return False, {"reason": "no-reference"}

    frame_processor = create_frame_processor(
        backend_name or default_backend_name(),
        validation_bbox,
        field_quad,
        {side: quad},
        fuel_base_color_rgb,
        working_scale=working_scale,
        detector_budget=detector_budget,
        detector_mode=detector_mode,
    )

    active_counts = []
    reader = None
    try:
        reader = FFmpegFrameReader(video_path, frame_width, frame_height, stride=stride)
        for frame_index, frame in enumerate(reader):
            if frame_index >= validation_frames:
                break
            mask = frame_processor.mask_for_frame(frame)
            raw_centers, _saturated = frame_processor.detect_from_mask(mask)
            centers = dedupe_air_centers(raw_centers)
            _unused_field_points, air_points = project_synchronized_field_and_air_points(
                centers,
                validation_bbox,
                projection_matrix,
                field_width,
                field_height,
                mask_binary=mask,
                wall_references=wall_reference,
            )
            filtered_air_points = filter_air_profile_points(air_points)
            active_counts.append(sum(1 for _x, height in filtered_air_points if int(height) > 0))
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

    if not active_counts:
        return False, {"reason": "no-frames"}

    active_frame_fraction = float(sum(1 for count in active_counts if count > 0)) / float(len(active_counts))
    median_active = float(np.median(np.array(active_counts, dtype=np.float32)))
    is_valid = not (
        active_frame_fraction >= float(AIR_PROFILE_WALL_INVALID_FRACTION)
        and median_active >= float(AIR_PROFILE_WALL_INVALID_MEDIAN_ACTIVE)
    )
    return is_valid, {
        "reason": "validated" if is_valid else "suppressed-noisy-wall",
        "frames": len(active_counts),
        "activeFrameFraction": active_frame_fraction,
        "medianActiveCount": median_active,
    }


def project_fuel_points_from_centers(centers, bbox, projection_matrix, field_width, field_height):
    field_points, _ = project_synchronized_field_and_air_points(
        centers,
        bbox,
        projection_matrix,
        field_width,
        field_height,
    )
    return field_points


def project_fuel_points(frame, bbox, field_quad, projection_matrix, field_width, field_height, fuel_base_color_rgb=None):
    mask = roi_yellow_mask_binary(frame, bbox, field_quad, fuel_base_color_rgb=fuel_base_color_rgb)
    if not mask_has_signal(mask):
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


class JsonFrameStream:
    def __init__(self, directory, prefix):
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=prefix,
            suffix=".jsonpart",
            dir=directory,
            delete=False,
        )
        self.path = tmp.name
        self._handle = tmp
        self._needs_comma = False
        self._finalized = False

    def write_frame(self, frame_points):
        if self._needs_comma:
            self._handle.write(",")
        json.dump(frame_points, self._handle, separators=(",", ":"))
        self._needs_comma = True

    def finalize_to(self, output_path, payload_prefix, payload_suffix):
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
        with open(output_path, "w", encoding="utf-8") as out_handle:
            out_handle.write(payload_prefix)
            with open(self.path, "r", encoding="utf-8") as frames_handle:
                shutil.copyfileobj(frames_handle, out_handle)
            out_handle.write(payload_suffix)
        self._finalized = True
        self.cleanup()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def cleanup(self):
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


class OverlayFramesSink(OverlayFrameWriter):
    def __init__(self, frames_dir):
        self.frames_dir = frames_dir
        self.frame_count = 0
        os.makedirs(self.frames_dir, exist_ok=True)

    def write(self, frame_rgb):
        frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.png")
        # PNG writes are larger than WEBP, but they have been more reliable for long runs.
        Image.fromarray(np.ascontiguousarray(frame_rgb), mode="RGB").save(
            frame_path,
            format="PNG",
        )
        self.frame_count += 1


class OverlayVideoSink(OverlayFrameWriter):
    def __init__(self, output_path, width, height, fps, prefer_nvenc=True):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(fps, 1.0))
        self.frame_count = 0

        encoder = "libx264"
        # NVENC rejects very small frame sizes on some drivers; keep tiny test clips on libx264.
        nvenc_size_ok = self.width >= 145 and self.height >= 145
        if prefer_nvenc and nvenc_size_ok and has_accessible_nvidia_gpu() and ffmpeg_has_encoder("h264_nvenc"):
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


def create_live_overlay_frame(frame, bbox, field_quad, overlay_color, fuel_base_color_rgb=None):
    """
    One fixed-size dot per detected fuel center. Uses the same ROI mask as the field map.
    """
    h, w = frame.shape[:2]
    mask = roi_yellow_mask_binary(frame, bbox, field_quad, fuel_base_color_rgb=fuel_base_color_rgb)
    if not mask_has_signal(mask):
        return np.zeros((h, w, 3), dtype=np.uint8)
    centers = ball_centers_from_mask(mask, detector_mode="hybrid")
    return draw_fuel_dots_full_frame(h, w, bbox, centers, overlay_color)


def write_dynamic_assets(
    video_path,
    session_dir,
    field_map_path,
    raw_data_path,
    bbox,
    field_quad,
    average_display_color,
    fuel_base_color_rgb,
    field_image_path,
    wall_quad=None,
    wall_quads=None,
    target_process_fps=DEFAULT_TARGET_PROCESS_FPS,
    progress_callback=None,
    progress_total_frames=None,
    progress_offset=0,
    backend_name=None,
    overlay_output="frames",
    working_scale=DEFAULT_WORKING_SCALE,
    detector_budget=None,
    max_active_tracks=DEFAULT_MAX_ACTIVE_TRACKS,
    detector_mode="hybrid",
):
    metadata = probe_video_metadata(video_path)
    fps = metadata["fps"]
    frame_width = int(metadata["width"])
    frame_height = int(metadata["height"])
    os.makedirs(session_dir, exist_ok=True)
    field_bbox = analysis.clamp_bbox(bbox, frame_width, frame_height)
    if field_bbox is None:
        raise RuntimeError("Bounding box is outside the video frame.")
    x, y, width, height = field_bbox
    raw_data = np.zeros((frame_height, frame_width), dtype=np.uint32)
    raw_data_view = raw_data[y : y + height, x : x + width]

    field_image = Image.open(field_image_path).convert("RGB")
    field_width, field_height = field_image.size
    projection_matrix = cv2.getPerspectiveTransform(
        field_quad if field_quad is not None else np.array(
            [
                [field_bbox[0], field_bbox[1]],
                [field_bbox[0] + field_bbox[2], field_bbox[1]],
                [field_bbox[0] + field_bbox[2], field_bbox[1] + field_bbox[3]],
                [field_bbox[0], field_bbox[1] + field_bbox[3]],
            ],
            dtype=np.float32,
        ),
        build_destination_quad(field_width, field_height),
    )
    resolved_wall_quads = dict(wall_quads or {})
    if wall_quad is not None and "right" not in resolved_wall_quads:
        resolved_wall_quads["right"] = wall_quad
    if resolved_wall_quads:
        validated_wall_quads = {}
        for side, quad in resolved_wall_quads.items():
            is_valid, validation_stats = validate_wall_side(
                video_path,
                metadata,
                field_bbox,
                field_quad,
                side,
                quad,
                projection_matrix,
                field_width,
                field_height,
                fuel_base_color_rgb,
                working_scale,
                detector_budget,
                detector_mode,
                backend_name,
            )
            stats_payload = json.dumps(validation_stats, separators=(",", ":"))
            if is_valid:
                print(f"Wall {side} validation passed: {stats_payload}")
                validated_wall_quads[side] = quad
            else:
                print(f"Wall {side} validation suppressed: {stats_payload}")
        resolved_wall_quads = validated_wall_quads
    legacy_wall_for_fallback = wall_quad if (wall_quad is not None and not wall_quads) else None
    wall_references = build_wall_references(field_quad, resolved_wall_quads, legacy_wall_quad=legacy_wall_for_fallback)
    air_bbox = bbox_from_quads_pixels(
        roi_quads_for_detection(field_quad, resolved_wall_quads),
        frame_width,
        frame_height,
        padding_px=6,
    )
    air_bbox = analysis.clamp_bbox(air_bbox, frame_width, frame_height) if air_bbox is not None else None

    field_frame_processor = create_frame_processor(
        backend_name or default_backend_name(),
        field_bbox,
        field_quad,
        None,
        fuel_base_color_rgb,
        working_scale=working_scale,
        detector_budget=detector_budget,
        detector_mode=detector_mode,
    )
    air_frame_processor = None
    air_detector_mode = "peak"
    if "left" in resolved_wall_quads or len(wall_references) > 1:
        air_detector_mode = detector_mode
    if wall_references and air_bbox is not None:
        air_frame_processor = create_frame_processor(
            backend_name or default_backend_name(),
            air_bbox,
            field_quad,
            resolved_wall_quads,
            fuel_base_color_rgb,
            working_scale=working_scale,
            detector_budget=detector_budget,
            detector_mode=air_detector_mode,
        )

    frame_count = 0
    field_frame_stream = JsonFrameStream(session_dir, "field-frames-")
    air_profile_stream = JsonFrameStream(session_dir, "air-profile-") if air_frame_processor is not None else None
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
    field_temporal_stabilizer = FuelTemporalStabilizer(max_active_tracks=max_active_tracks)
    sample_weight = np.uint32(stride)
    metrics = ProcessingMetrics()

    print(f"Video FPS: {fps}")
    print(f"Target processing FPS: {target_process_fps}")
    print(f"Frame stride: {stride}")
    print(f"Effective exported FPS: {export_fps}")
    print(f"Backend: {field_frame_processor.backend_name}")
    print(f"Overlay output: {overlay_output}")
    print(f"Field detector mode: {detector_mode}")
    if air_frame_processor is not None:
        print(f"Air detector mode: {air_detector_mode}")
        print(f"Field bbox: {field_bbox}")
        print(f"Air bbox: {air_bbox}")

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

            raw_data_view += analysis.fuel_pixel_mask_hsv(
                frame[y : y + height, x : x + width],
                base_color_rgb=fuel_base_color_rgb,
            ).astype(np.uint32) * sample_weight

            mask_start = time.perf_counter()
            field_mask = field_frame_processor.mask_for_frame(frame)
            metrics.add_time("mask", time.perf_counter() - mask_start)

            detect_start = time.perf_counter()
            raw_centers, saturated = field_frame_processor.detect_from_mask(field_mask)
            metrics.add_time("detect", time.perf_counter() - detect_start)

            stabilize_start = time.perf_counter()
            stable_centers, _bad_frame = field_temporal_stabilizer.stabilize(raw_centers, saturated=saturated)
            metrics.add_time("stabilize", time.perf_counter() - stabilize_start)
            metrics.add_counts(len(raw_centers), len(stable_centers), saturated)

            live_overlay = draw_fuel_dots_full_frame(
                frame_height, frame_width, field_bbox, stable_centers, average_display_color
            )

            project_start = time.perf_counter()
            synchronized_field_points, _unused_air_points = project_synchronized_field_and_air_points(
                stable_centers,
                field_bbox,
                projection_matrix,
                field_width,
                field_height,
                mask_binary=field_mask,
            )
            field_frame_stream.write_frame(synchronized_field_points)
            if air_profile_stream is not None and air_frame_processor is not None:
                air_mask = air_frame_processor.mask_for_frame(frame)
                air_raw_centers, air_saturated = air_frame_processor.detect_from_mask(air_mask)
                air_centers = dedupe_air_centers(air_raw_centers)
                _unused_field_points, synchronized_air_points = project_synchronized_field_and_air_points(
                    air_centers,
                    air_bbox,
                    projection_matrix,
                    field_width,
                    field_height,
                    mask_binary=air_mask,
                    wall_references=wall_references,
                )
                synchronized_air_points = filter_air_profile_points(synchronized_air_points)
                air_profile_stream.write_frame(synchronized_air_points)
            metrics.add_time("project", time.perf_counter() - project_start)

            encode_start = time.perf_counter()
            overlay_sink.write(live_overlay)
            metrics.add_time("encode", time.perf_counter() - encode_start)

            frame_count += 1
            video_read_count += stride
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error = None
        if frame_reader is not None:
            try:
                frame_reader.close()
            except Exception as exc:
                cleanup_error = exc
        metrics.add_time("total", time.perf_counter() - total_start)
        try:
            overlay_sink.close()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        for stream in (field_frame_stream, air_profile_stream):
            if stream is None:
                continue
            try:
                stream.close()
                if active_error is not None:
                    stream.cleanup()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None:
            if active_error is None:
                raise cleanup_error
            print(f"Cleanup warning: {cleanup_error}", file=sys.stderr)
    print()

    print("Saving raw data...")
    np.savetxt(raw_data_path, raw_data, fmt="%d")

    field_frame_stream.finalize_to(
        field_map_path,
        (
            f'{{"imageWidth":{int(field_width)},"imageHeight":{int(field_height)},'
            f'"fps":{json.dumps(export_fps)},"frameCount":{int(frame_count)},"frames":['
        ),
        "]}",
    )

    if air_profile_stream is not None and wall_references:
        air_profile_path = os.path.join(session_dir, "air-profile.json")
        exported_wall_side = wall_references[0]["side"] if len(wall_references) == 1 else "mixed"
        air_profile_stream.finalize_to(
            air_profile_path,
            (
                f'{{"fps":{json.dumps(export_fps)},"frameCount":{int(frame_count)},'
                f'"wallSide":{json.dumps(exported_wall_side)},"frames":['
            ),
            "]}",
        )

    return {
        "raw_data": raw_data,
        "fps": export_fps,
        "frameCount": frame_count,
        "backend": field_frame_processor.backend_name,
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
    parser.add_argument("--wall-quad", help="Wall quad in x1,y1,x2,y2,x3,y3,x4,y4 pixels")
    parser.add_argument("--wall-quad-left", help="Left wall quad in x1,y1,x2,y2,x3,y3,x4,y4 pixels")
    parser.add_argument("--wall-quad-right", help="Right wall quad in x1,y1,x2,y2,x3,y3,x4,y4 pixels")
    parser.add_argument("--average-display-color", default="255,0,255", help="RGB color for the average intensity point")
    parser.add_argument(
        "--fuel-base-color",
        default=",".join(str(value) for value in DEFAULT_FUEL_BASE_COLOR_RGB),
        help="RGB color sampled from fuel for HSV-based detection",
    )
    parser.add_argument("--pct-from-average-to-max", type=float, default=0.5)
    parser.add_argument("--target-process-fps", type=float, default=DEFAULT_TARGET_PROCESS_FPS)
    parser.add_argument("--field-image", default=DEFAULT_FIELD_IMAGE_PATH, help="Top-down field asset for field-map projection")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default=default_backend_name())
    parser.add_argument("--overlay-output", choices=("video", "frames"), default="frames")
    parser.add_argument("--working-scale", type=float, default=DEFAULT_WORKING_SCALE)
    parser.add_argument("--detector-budget", type=int, default=0)
    parser.add_argument("--max-active-tracks", type=int, default=DEFAULT_MAX_ACTIVE_TRACKS)
    parser.add_argument("--detector-mode", choices=("legacy", "peak", "hybrid"), default="hybrid")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    field_quad = parse_quad(args.quad) if args.quad else None
    wall_quad = parse_quad(args.wall_quad) if args.wall_quad else None
    wall_quads = {}
    if args.wall_quad_left:
        wall_quads["left"] = parse_quad(args.wall_quad_left)
    if args.wall_quad_right:
        wall_quads["right"] = parse_quad(args.wall_quad_right)
    average_display_color = tuple(int(part.strip()) for part in args.average_display_color.split(","))
    fuel_base_color_rgb = parse_rgb_color(args.fuel_base_color)

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
        fuel_base_color_rgb,
        args.field_image,
        wall_quad=wall_quad,
        wall_quads=wall_quads,
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
        "fuelBaseColor": {
            "r": int(fuel_base_color_rgb[0]),
            "g": int(fuel_base_color_rgb[1]),
            "b": int(fuel_base_color_rgb[2]),
        },
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
