import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

import analysis


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


def create_live_overlay_frame(frame_mask, overlay_color):
    mask_8bit = (frame_mask > 0).astype(np.uint8) * 255
    if not np.any(mask_8bit):
        return np.zeros((*frame_mask.shape, 3), dtype=np.uint8)

    expanded = cv2.dilate(mask_8bit, np.ones((5, 5), dtype=np.uint8), iterations=1)
    smoothed = cv2.GaussianBlur(expanded, (0, 0), sigmaX=1.6, sigmaY=1.6)
    normalized = np.clip(smoothed.astype(np.float32) / 255.0, 0.0, 1.0)
    color = np.array(overlay_color, dtype=np.float32)
    return np.clip(normalized[:, :, None] * color[None, None, :], 0, 255).astype(np.uint8)


def write_overlay_frames(video_path, overlay_frames_dir, bbox, average_display_color):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bbox = analysis.clamp_bbox(bbox, frame_width, frame_height)

    os.makedirs(overlay_frames_dir, exist_ok=True)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_mask = analysis.analyze_frame(frame, bbox=bbox)
        live_overlay = create_live_overlay_frame(frame_mask, average_display_color)
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

    cap.release()
    return {
        "fps": float(fps),
        "frameCount": frame_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a fuel-density overlay from a video.")
    parser.add_argument("--video", required=True, help="Path to the local video file")
    parser.add_argument("--session-dir", required=True, help="Directory to store generated files")
    parser.add_argument("--bbox", required=True, help="Bounding box in x,y,width,height pixels")
    parser.add_argument("--average-display-color", default="255,0,255", help="RGB color for the average intensity point")
    parser.add_argument("--pct-from-average-to-max", type=float, default=0.5)
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    average_display_color = tuple(int(part.strip()) for part in args.average_display_color.split(","))

    print("Analyzing video...")
    raw_data = analysis.get_total_yellow_pixels_from_video(args.video, bbox=bbox)

    os.makedirs(args.session_dir, exist_ok=True)

    raw_data_path = os.path.join(args.session_dir, "raw_data.txt")
    overlay_path = os.path.join(args.session_dir, "overlay.png")
    transparent_overlay_path = os.path.join(args.session_dir, "overlay-transparent.png")
    overlay_frames_dir = os.path.join(args.session_dir, "overlay-frames")
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
    overlay_timing = write_overlay_frames(
        args.video,
        overlay_frames_dir,
        bbox,
        average_display_color,
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
