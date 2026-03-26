import cv2
import numpy as np

# Pixels with absdiff below this vs previous frame count as "static" (fixed camera).
MOTION_DIFF_THRESHOLD = 15


def yellow_pixel_mask(frame):
    """Legacy RGB thresholds (used for benchmarks vs old heat-map behavior)."""
    return (
        (frame[:, :, 2] > 200)
        & (frame[:, :, 1] > 200)
        & (frame[:, :, 0] < 100)
    )


def yellow_pixel_mask_hsv_legacy(frame_bgr):
    """Original single-range HSV mask (overlay path before unification). For benchmarks."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([15, 70, 110]), np.array([45, 255, 255]))


def fuel_mask_bgr(frame_bgr):
    """
    Unified fuel-yellow mask: dual HSV (bright + shadow) plus LAB chrominance to
    reduce green-turf false positives and recover shaded fuel.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m_bright = cv2.inRange(hsv, np.array([15, 70, 100]), np.array([45, 255, 255]))
    m_shadow = cv2.inRange(hsv, np.array([16, 75, 28]), np.array([45, 255, 165]))
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    # Slightly wider a/b on bright path so shaded interiors of large clumps still pass.
    m_lab = cv2.inRange(lab, np.array([28, 112, 115]), np.array([255, 150, 255]))
    m_lab_shadow = cv2.inRange(lab, np.array([20, 102, 110]), np.array([255, 158, 255]))
    return cv2.bitwise_or(
        cv2.bitwise_and(m_bright, m_lab),
        cv2.bitwise_and(m_shadow, m_lab_shadow),
    )


def fuel_mask_bgr_fast(frame_bgr):
    """
    HSV-only dual range (no LAB). Faster; use --fast when quality tradeoff is OK.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m_bright = cv2.inRange(hsv, np.array([15, 70, 100]), np.array([45, 255, 255]))
    m_shadow = cv2.inRange(hsv, np.array([16, 75, 28]), np.array([45, 255, 165]))
    return cv2.bitwise_or(m_bright, m_shadow)


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


def get_total_yellow_pixels_from_video(
    vid_path,
    start_time=0,
    end_time=None,
    bbox=None,
    progress_callback=None,
    use_motion_gate=True,
    motion_threshold=MOTION_DIFF_THRESHOLD,
    use_fast_mask=False,
):
    cap = cv2.VideoCapture(vid_path)

    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second

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

    phase_total = max(1, end_frame - start_frame)
    processed = 0
    progress_stride = max(1, phase_total // 150)

    prev_gray_roi = None

    while frame_count < end_frame:
        if frame_count == start_frame or frame_count % max(int(fps), 1) == 0:
            print(f"Processing frame {frame_count}/{end_frame}", end="\r")
        ret, frame = cap.read()
        if not ret:
            break

        frame_mask = analyze_frame(
            frame,
            bbox=bbox,
            prev_gray_roi=prev_gray_roi,
            use_motion_gate=use_motion_gate,
            motion_threshold=motion_threshold,
            use_fast_mask=use_fast_mask,
        )
        if bbox is not None:
            x, y, w, h = bbox
            prev_gray_roi = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        else:
            prev_gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        video_frame_totals += frame_mask

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


def analyze_frame(
    frame,
    bbox=None,
    prev_gray_roi=None,
    use_motion_gate=True,
    motion_threshold=MOTION_DIFF_THRESHOLD,
    use_fast_mask=False,
):
    """Accumulate per-pixel fuel hits using the same color model as the overlay path."""
    mask_fn = fuel_mask_bgr_fast if use_fast_mask else fuel_mask_bgr
    if bbox is None:
        mask_bool = mask_fn(frame) > 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if use_motion_gate and prev_gray_roi is not None and prev_gray_roi.shape == gray.shape:
            mask_bool &= np.abs(gray.astype(np.int16) - prev_gray_roi.astype(np.int16)) < motion_threshold
        return mask_bool.astype(np.uint32)

    x, y, width, height = bbox
    mask = np.zeros(frame.shape[:2], dtype=np.uint32)
    frame_slice = frame[y : y + height, x : x + width]
    mask_bool = mask_fn(frame_slice) > 0
    gray_roi = cv2.cvtColor(frame_slice, cv2.COLOR_BGR2GRAY)
    if use_motion_gate and prev_gray_roi is not None and prev_gray_roi.shape == gray_roi.shape:
        mask_bool &= np.abs(gray_roi.astype(np.int16) - prev_gray_roi.astype(np.int16)) < motion_threshold
    mask[y : y + height, x : x + width] = mask_bool.astype(np.uint32)
    return mask
