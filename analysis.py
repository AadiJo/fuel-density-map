import cv2
import numpy as np

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


def get_total_yellow_pixels_from_video(vid_path, start_time=0, end_time=None, bbox=None):
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

    # go frame by frame
    while frame_count < end_frame:
        if frame_count == start_frame or frame_count % max(int(fps), 1) == 0:
            print(f"Processing frame {frame_count}/{end_frame}", end="\r")
        ret, frame = cap.read()
        if not ret:
            break

        frame_mask = analyze_frame(frame, bbox=bbox)
        video_frame_totals += frame_mask

        frame_count += 1
    cap.release()
    print()
    return video_frame_totals


def analyze_frame(frame, bbox=None):
    # OpenCV frames are BGR. Match the original RGB thresholds without converting.
    if bbox is None:
        return (
            (frame[:, :, 2] > 200) &
            (frame[:, :, 1] > 200) &
            (frame[:, :, 0] < 100)
        ).astype(np.uint32)

    x, y, width, height = bbox
    mask = np.zeros(frame.shape[:2], dtype=np.uint32)
    frame_slice = frame[y : y + height, x : x + width]
    mask[y : y + height, x : x + width] = (
        (frame_slice[:, :, 2] > 200) &
        (frame_slice[:, :, 1] > 200) &
        (frame_slice[:, :, 0] < 100)
    ).astype(np.uint32)
    return mask
