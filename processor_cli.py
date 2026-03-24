import argparse
import json
import os

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

FIELD_DESTINATION_BOUNDS = {
    "top_left": (0.064, 0.053),
    "top_right": (0.866, 0.053),
    "bottom_right": (0.866, 0.945),
    "bottom_left": (0.064, 0.945),
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


def roi_yellow_mask_binary(frame, bbox, field_quad):
    """HSV yellow mask cropped to the ROI and field quad, with light noise removal."""
    x, y, width, height = bbox
    frame_slice = frame[y : y + height, x : x + width]
    if frame_slice.size == 0:
        return None
    mask = analysis.yellow_pixel_mask_hsv(frame_slice)
    if field_quad is not None:
        mask = cv2.bitwise_and(mask, quad_mask(bbox, field_quad))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )
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


def create_live_overlay_frame(frame, bbox, field_quad, overlay_color):
    """
    One fixed-size dot per detected fuel center. Uses the same ROI mask as the field map
    (not dilate+blur — that merged entire clusters into one pink smear).
    """
    h, w = frame.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    mask = roi_yellow_mask_binary(frame, bbox, field_quad)
    if mask is None or not np.any(mask):
        return out

    centers = ball_centers_from_mask(mask)
    color = tuple(int(c) for c in overlay_color)
    dot_r = OVERLAY_FUEL_DOT_RADIUS_PX
    x0, y0 = bbox[0], bbox[1]
    for cx, cy in centers:
        cv2.circle(out, (int(x0 + cx), int(y0 + cy)), dot_r, color, -1, lineType=cv2.LINE_AA)
    return out


def component_radius(area):
    return int(np.clip(4.5 + np.sqrt(max(area, 1)) / 2.0, 5, 16))


def project_fuel_points(frame, bbox, field_quad, projection_matrix, field_width, field_height):
    x, y, width, height = bbox
    mask = roi_yellow_mask_binary(frame, bbox, field_quad)
    if mask is None or not np.any(mask):
        return []

    peaks = ball_centers_from_mask(mask)
    points = []

    for centroid_x, centroid_y in peaks:
        source_point = np.array([[[x + centroid_x, y + centroid_y]]], dtype=np.float32)
        projected_point = cv2.perspectiveTransform(source_point, projection_matrix)[0, 0]

        normalized_x = int(round(np.clip(projected_point[0] / field_width, 0.0, 1.0) * 10000))
        normalized_y = int(round(np.clip(projected_point[1] / field_height, 0.0, 1.0) * 10000))
        points.append([normalized_x, normalized_y, FIELD_MAP_FUEL_RADIUS])

    return points


def write_dynamic_assets(
    video_path,
    overlay_frames_dir,
    field_map_path,
    bbox,
    field_quad,
    average_display_color,
    field_image_path,
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        live_overlay = create_live_overlay_frame(frame, bbox, field_quad, average_display_color)
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

        field_frames.append(project_fuel_points(frame, bbox, field_quad, projection_matrix, field_width, field_height))
        frame_count += 1

    cap.release()

    with open(field_map_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "imageWidth": field_width,
                "imageHeight": field_height,
                "fps": float(fps),
                "frameCount": frame_count,
                "frames": field_frames,
            },
            handle,
            separators=(",", ":"),
        )

    return {
        "fps": float(fps),
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
    parser.add_argument("--field-image", default=DEFAULT_FIELD_IMAGE_PATH, help="Top-down field asset for field-map projection")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    field_quad = parse_quad(args.quad) if args.quad else None
    average_display_color = tuple(int(part.strip()) for part in args.average_display_color.split(","))

    print("Analyzing video...")
    raw_data = analysis.get_total_yellow_pixels_from_video(args.video, bbox=bbox)

    os.makedirs(args.session_dir, exist_ok=True)

    raw_data_path = os.path.join(args.session_dir, "raw_data.txt")
    overlay_path = os.path.join(args.session_dir, "overlay.png")
    transparent_overlay_path = os.path.join(args.session_dir, "overlay-transparent.png")
    overlay_frames_dir = os.path.join(args.session_dir, "overlay-frames")
    field_map_path = os.path.join(args.session_dir, "field-map.json")
    stats_path = os.path.join(args.session_dir, "stats.json")

    print("Saving raw data...")
    np.savetxt(raw_data_path, raw_data, fmt="%d")

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

    print("Creating overlay frames...")
    overlay_timing = write_dynamic_assets(
        args.video,
        overlay_frames_dir,
        field_map_path,
        bbox,
        field_quad,
        average_display_color,
        args.field_image,
    )

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
