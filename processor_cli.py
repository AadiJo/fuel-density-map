import argparse
import json
import os
import sys
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

PROGRESS_PREFIX = "PROGRESS_JSON:"


def emit_progress(phase, current, total):
    """Machine-readable progress for the Node server (stderr, line-buffered)."""
    line = f'{PROGRESS_PREFIX}{json.dumps({"phase": phase, "current": current, "total": total})}\n'
    sys.stderr.write(line)
    sys.stderr.flush()


def compute_total_work_units(video_path):
    """Single sampled pass over the source video."""
    cap = cv2.VideoCapture(video_path)
    frame_count_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(1, frame_count_cap)


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


_MORPH_KERNEL_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))


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


def ball_centers_from_mask(mask_binary, max_centers=500, **_kwargs):
    """
    Area-calibrated ball detection.

    Uses the smallest detected blobs as a single-ball reference, then estimates
    how many balls each larger blob contains by dividing its area.  Distance-
    transform peaks place centers inside large blobs; any shortfall is filled by
    uniform sampling along the blob's pixels.
    """
    mask_work = (mask_binary > 0).astype(np.uint8) * 255
    if not np.any(mask_work):
        return []

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_work)
    if n_labels <= 1:
        return []

    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)])

    # Calibrate single-ball area from the smallest blobs.
    sorted_areas = np.sort(areas)
    bottom_n = max(1, len(sorted_areas) * 40 // 100)
    single_ball_area = float(np.clip(np.median(sorted_areas[:bottom_n]), 6, 200))

    min_sep = max(1.5, np.sqrt(single_ball_area) * 0.35)

    centers = []
    for i in range(1, n_labels):
        if len(centers) >= max_centers:
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

    return centers[:max_centers]


# Third JSON field kept for compatibility; UI draws a fixed size.
FIELD_MAP_FUEL_RADIUS = 10

OVERLAY_FUEL_DOT_RADIUS_PX = 5

FUEL_TRACK_MATCH_RADIUS_PX = 14.0
FUEL_TRACK_MAX_MISSES = 2
BAD_FRAME_HISTORY_SIZE = 6
BAD_FRAME_MIN_BASELINE_COUNT = 6
BAD_FRAME_MIN_COUNT_DELTA = 8
BAD_FRAME_COUNT_DELTA_RATIO = 0.45


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


class FuelTemporalStabilizer:
    """Hold stationary fuel through brief dropouts and reject obvious count-spike frames."""

    def __init__(
        self,
        match_radius_px=FUEL_TRACK_MATCH_RADIUS_PX,
        max_misses=FUEL_TRACK_MAX_MISSES,
        history_size=BAD_FRAME_HISTORY_SIZE,
        min_baseline_count=BAD_FRAME_MIN_BASELINE_COUNT,
        min_count_delta=BAD_FRAME_MIN_COUNT_DELTA,
        count_delta_ratio=BAD_FRAME_COUNT_DELTA_RATIO,
    ):
        self.match_radius_sq = float(match_radius_px) * float(match_radius_px)
        self.max_misses = int(max_misses)
        self.min_baseline_count = int(min_baseline_count)
        self.min_count_delta = int(min_count_delta)
        self.count_delta_ratio = float(count_delta_ratio)
        self.count_history = deque(maxlen=max(1, int(history_size)))
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

    def _match_and_update_tracks(self, centers):
        if not self.tracks:
            self.tracks = [{"x": int(cx), "y": int(cy), "missed": 0} for cx, cy in centers]
            return

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_centers = set(range(len(centers)))
        matches = []
        for center_index, (cx, cy) in enumerate(centers):
            for track_index, track in enumerate(self.tracks):
                dx = float(cx) - float(track["x"])
                dy = float(cy) - float(track["y"])
                dist_sq = dx * dx + dy * dy
                if dist_sq <= self.match_radius_sq:
                    matches.append((dist_sq, track_index, center_index))
        matches.sort(key=lambda item: item[0])

        next_tracks = [None] * len(self.tracks)
        for _, track_index, center_index in matches:
            if track_index not in unmatched_tracks or center_index not in unmatched_centers:
                continue
            cx, cy = centers[center_index]
            next_tracks[track_index] = {"x": int(cx), "y": int(cy), "missed": 0}
            unmatched_tracks.remove(track_index)
            unmatched_centers.remove(center_index)

        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            missed = int(track["missed"]) + 1
            if missed <= self.max_misses:
                next_tracks[track_index] = {"x": track["x"], "y": track["y"], "missed": missed}

        self.tracks = [track for track in next_tracks if track is not None]
        for center_index in sorted(unmatched_centers):
            cx, cy = centers[center_index]
            self.tracks.append({"x": int(cx), "y": int(cy), "missed": 0})

    def stabilize(self, centers):
        raw_centers = [(int(cx), int(cy)) for cx, cy in centers]
        raw_count = len(raw_centers)

        if self._is_bad_frame(raw_count):
            return self._current_centers(), True

        self._match_and_update_tracks(raw_centers)
        stabilized = self._current_centers()
        self.count_history.append(len(stabilized))
        return stabilized, False


def create_live_overlay_frame(frame, bbox, field_quad, overlay_color):
    """
    One fixed-size dot per detected fuel center. Uses the same ROI mask as the field map
    (not dilate+blur — that merged entire clusters into one pink smear).
    """
    h, w = frame.shape[:2]
    mask = roi_yellow_mask_binary(frame, bbox, field_quad)
    if mask is None or not np.any(mask):
        return np.zeros((h, w, 3), dtype=np.uint8)
    centers = ball_centers_from_mask(mask)
    return draw_fuel_dots_full_frame(h, w, bbox, centers, overlay_color)


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


def write_dynamic_assets(
    video_path,
    overlay_frames_dir,
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
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bbox = analysis.clamp_bbox(bbox, frame_width, frame_height)
    if bbox is None:
        cap.release()
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

    os.makedirs(overlay_frames_dir, exist_ok=True)
    frame_count = 0
    field_frames = []
    total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    progress_stride = max(1, total_frames // 150)
    roi_quad_mask = precompute_roi_quad_mask(bbox, field_quad)
    stride = compute_frame_stride(fps, target_process_fps)
    export_fps = float(fps) / float(stride)
    temporal_stabilizer = FuelTemporalStabilizer()
    sample_weight = np.uint32(stride)

    print(f"Video FPS: {fps}")
    print(f"Target processing FPS: {target_process_fps}")
    print(f"Frame stride: {stride}")
    print(f"Effective exported FPS: {export_fps}")

    if progress_callback and progress_total_frames is not None:
        progress_callback(progress_offset, progress_total_frames)

    video_read_count = 0
    while video_read_count < total_frames:
        should_process = (video_read_count % stride) == 0
        if should_process:
            ret, frame = cap.read()
        else:
            ret = cap.grab()
            frame = None
        if not ret:
            break

        if progress_callback and progress_total_frames is not None:
            if (
                video_read_count == 0
                or video_read_count % progress_stride == 0
                or video_read_count + 1 >= total_frames
            ):
                progress_callback(progress_offset + video_read_count + 1, progress_total_frames)

        if not should_process:
            video_read_count += 1
            continue

        if video_read_count == 0 or video_read_count % max(int(export_fps), 1) == 0:
            print(f"Processing sampled frame {video_read_count}/{total_frames}", end="\r")

        raw_data_view += analysis.yellow_pixel_mask(frame[y : y + height, x : x + width]).astype(np.uint32) * sample_weight

        # Single mask + center pass per frame (create_live_overlay + project_fuel duplicated this).
        mask = roi_yellow_mask_binary(frame, bbox, field_quad, quad_mask_roi=roi_quad_mask)
        centers = []
        if mask is not None and np.any(mask):
            centers = ball_centers_from_mask(mask)
        stable_centers, _bad_frame = temporal_stabilizer.stabilize(centers)
        live_overlay = draw_fuel_dots_full_frame(
            frame_height, frame_width, bbox, stable_centers, average_display_color
        )
        field_frames.append(
            project_fuel_points_from_centers(
                stable_centers, bbox, projection_matrix, field_width, field_height
            )
        )

        success, encoded = cv2.imencode(
            ".webp",
            cv2.cvtColor(live_overlay, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_WEBP_QUALITY, 80],
        )
        if not success:
            cap.release()
            raise RuntimeError("Unable to encode overlay frame.")

        frame_path = os.path.join(overlay_frames_dir, f"frame_{frame_count:06d}.webp")
        encoded.tofile(frame_path)
        frame_count += 1
        video_read_count += 1

    cap.release()
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
    overlay_frames_dir = os.path.join(args.session_dir, "overlay-frames")
    field_map_path = os.path.join(args.session_dir, "field-map.json")
    stats_path = os.path.join(args.session_dir, "stats.json")

    print("Processing video...")
    overlay_timing = write_dynamic_assets(
        args.video,
        overlay_frames_dir,
        field_map_path,
        raw_data_path,
        bbox,
        field_quad,
        average_display_color,
        args.field_image,
        target_process_fps=args.target_process_fps,
        progress_callback=lambda processed, total: emit_progress("frames", min(processed, total_work), total_work),
        progress_total_frames=total_work,
    )
    raw_data = overlay_timing.pop("raw_data")

    emit_progress("encode", total_work, total_work)

    max_value = int(raw_data.max())
    non_zero_values = raw_data[raw_data > 0]
    actual_average = float(non_zero_values.mean()) if non_zero_values.size else 0.0
    average_of_non_zero_values = actual_average + args.pct_from_average_to_max * (max_value - actual_average)

    print("Creating overlay images...")
    color_array = create_color_array(raw_data, max_value, average_of_non_zero_values, average_display_color)
    Image.fromarray(color_array, mode="RGB").save(overlay_path)

    alpha_channel = np.where(np.any(color_array != 0, axis=2), 220, 0).astype(np.uint8)
    transparent_image = np.dstack((color_array, alpha_channel))
    Image.fromarray(transparent_image, mode="RGBA").save(transparent_overlay_path)

    stats = {
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
    }

    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print("Finished.")


if __name__ == "__main__":
    main()
