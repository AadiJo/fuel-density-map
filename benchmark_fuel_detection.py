#!/usr/bin/env python3
"""
Still-frame latency: legacy (single HSV) vs unified fuel_mask_bgr + same ROI/morph path as production.
Run from repo root: python benchmark_fuel_detection.py [--image path.png]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

import analysis
import processor_cli as pc

_MORPH_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
_MORPH_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _post_morph(m):
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _MORPH_OPEN)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, _MORPH_CLOSE)


def _parse_bbox(s: str):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise SystemExit("bbox must be x,y,width,height")
    return tuple(parts)


def _parse_quad(s: str | None):
    if not s:
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 8:
        raise SystemExit("quad must be 8 integers")
    return np.array(
        [[parts[0], parts[1]], [parts[2], parts[3]], [parts[4], parts[5]], [parts[6], parts[7]]],
        dtype=np.float32,
    )


def _legacy_mask_slice(frame_slice, bbox, field_quad, quad_mask_roi):
    m = analysis.yellow_pixel_mask_hsv_legacy(frame_slice)
    if field_quad is not None:
        qm = quad_mask_roi if quad_mask_roi is not None else pc.quad_mask(bbox, field_quad)
        m = cv2.bitwise_and(m, qm)
    return _post_morph(m)


def _new_mask_slice(frame_slice, bbox, field_quad, quad_mask_roi):
    m = analysis.fuel_mask_bgr(frame_slice)
    if field_quad is not None:
        qm = quad_mask_roi if quad_mask_roi is not None else pc.quad_mask(bbox, field_quad)
        m = cv2.bitwise_and(m, qm)
    return _post_morph(m)


def _fast_mask_slice(frame_slice, bbox, field_quad, quad_mask_roi):
    m = analysis.fuel_mask_bgr_fast(frame_slice)
    if field_quad is not None:
        qm = quad_mask_roi if quad_mask_roi is not None else pc.quad_mask(bbox, field_quad)
        m = cv2.bitwise_and(m, qm)
    return _post_morph(m)


def _time_calls(fn, repeats: int, warmup: int):
    for _ in range(warmup):
        fn()
    samples_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark fuel detection on a still frame.")
    parser.add_argument(
        "--image",
        help="Path to a still frame (e.g. 640x360). Default: synthetic yellow-on-green.",
    )
    parser.add_argument("--bbox", default="0,0,640,360", help="x,y,width,height")
    parser.add_argument("--quad", default=None, help="Optional field quad x1,y1,...")
    parser.add_argument("--repeats", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=40)
    args = parser.parse_args()

    bbox = _parse_bbox(args.bbox)
    field_quad = _parse_quad(args.quad)

    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Could not read {args.image}", file=sys.stderr)
            sys.exit(1)
    else:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :] = (40, 90, 40)
        cv2.circle(frame, (320, 180), 8, (0, 220, 255), -1, lineType=cv2.LINE_8)

    h, w = frame.shape[:2]
    bbox = analysis.clamp_bbox(bbox, w, h)
    if bbox is None:
        print("Invalid bbox", file=sys.stderr)
        sys.exit(1)

    x, y, bw, bh = bbox
    frame_slice = frame[y : y + bh, x : x + bw]
    roi_quad_mask = pc.precompute_roi_quad_mask(bbox, field_quad)

    def legacy_full():
        m = _legacy_mask_slice(frame_slice, bbox, field_quad, roi_quad_mask)
        pc.ball_centers_from_mask(m, shape_filter=True)

    def new_full():
        m = _new_mask_slice(frame_slice, bbox, field_quad, roi_quad_mask)
        pc.ball_centers_from_mask(m, shape_filter=True)

    def legacy_mask_only():
        _legacy_mask_slice(frame_slice, bbox, field_quad, roi_quad_mask)

    def new_mask_only():
        _new_mask_slice(frame_slice, bbox, field_quad, roi_quad_mask)

    def fast_full():
        m = _fast_mask_slice(frame_slice, bbox, field_quad, roi_quad_mask)
        pc.ball_centers_from_mask(m, shape_filter=True)

    def fast_mask_only():
        _fast_mask_slice(frame_slice, bbox, field_quad, roi_quad_mask)

    for name, fn in [
        ("legacy mask + centers", legacy_full),
        ("new mask + centers", new_full),
        ("fast HSV-only mask + centers", fast_full),
        ("legacy mask only", legacy_mask_only),
        ("new mask only", new_mask_only),
        ("fast mask only", fast_mask_only),
    ]:
        samples = _time_calls(fn, args.repeats, args.warmup)
        mean = statistics.mean(samples)
        med = statistics.median(samples)
        stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        fps = 1000.0 / mean if mean > 0 else 0.0
        print(f"{name}: mean={mean:.4f} ms  median={med:.4f} ms  stdev={stdev:.4f} ms  ~{fps:.1f} calls/s")

    # ROI path timing (includes cvt + optional motion off)
    def roi_new():
        m, _ = pc.roi_yellow_mask_binary(
            frame, bbox, field_quad, quad_mask_roi=roi_quad_mask, use_motion_gate=False
        )
        if m is not None:
            pc.ball_centers_from_mask(m, shape_filter=True)

    def roi_legacy():
        m = _legacy_mask_slice(frame_slice, bbox, field_quad, roi_quad_mask)
        pc.ball_centers_from_mask(m, shape_filter=True)

    samples = _time_calls(roi_new, args.repeats, args.warmup)
    print(
        f"roi_yellow_mask_binary(new)+centers (no motion): mean={statistics.mean(samples):.4f} ms  "
        f"median={statistics.median(samples):.4f} ms"
    )
    samples = _time_calls(roi_legacy, args.repeats, args.warmup)
    print(
        f"legacy slice+centers (same as old overlay mask): mean={statistics.mean(samples):.4f} ms  "
        f"median={statistics.median(samples):.4f} ms"
    )

    def roi_fast():
        m, _ = pc.roi_yellow_mask_binary(
            frame,
            bbox,
            field_quad,
            quad_mask_roi=roi_quad_mask,
            use_motion_gate=False,
            use_fast_mask=True,
        )
        if m is not None:
            pc.ball_centers_from_mask(m, shape_filter=True)

    samples = _time_calls(roi_fast, args.repeats, args.warmup)
    print(
        f"roi_yellow_mask_binary(fast)+centers (no motion): mean={statistics.mean(samples):.4f} ms  "
        f"median={statistics.median(samples):.4f} ms"
    )

    try:
        import ml_followup  # noqa: PLC0415

        print("ML follow-up:", ml_followup.evaluation_hook())
    except ImportError:
        pass


if __name__ == "__main__":
    main()
