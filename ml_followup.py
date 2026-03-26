"""
Optional second stage: ONNX/YOLO or CNN on yellow-blob crops to reject non-fuel.

The classical pipeline (`analysis.fuel_mask_bgr`, `processor_cli.ball_centers_from_mask`,
motion gating) is the supported path. When that is not enough, integrate a small model
here and measure FPS using `benchmark_fuel_detection.py` on the same still frames.
"""


def evaluation_hook():
    return {
        "status": "not_implemented",
        "hint": "Export a tiny classifier to ONNX; run inference on cv2.boundingRect crops of mask components.",
        "benchmark": "python benchmark_fuel_detection.py",
    }
