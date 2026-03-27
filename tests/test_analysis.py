import unittest
from pathlib import Path

import cv2
import numpy as np

import analysis
import processor_cli


def reference_analyze_frame(frame, bbox=None):
    if bbox is None:
        return analysis.yellow_pixel_mask(frame).astype(np.uint32)

    x, y, width, height = bbox
    mask = np.zeros(frame.shape[:2], dtype=np.uint32)
    frame_slice = frame[y : y + height, x : x + width]
    mask[y : y + height, x : x + width] = analysis.yellow_pixel_mask(frame_slice).astype(np.uint32)
    return mask


class AnalysisTests(unittest.TestCase):
    def _write_test_video(self, path, frames, fps=5.0):
        for codec, suffix in (("MJPG", ".avi"), ("mp4v", ".mp4"), ("XVID", ".avi")):
            candidate = path.with_suffix(suffix)
            writer = cv2.VideoWriter(
                str(candidate),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (frames[0].shape[1], frames[0].shape[0]),
            )
            if not writer.isOpened():
                writer.release()
                continue
            for frame in frames:
                writer.write(frame)
            writer.release()
            return candidate
        self.fail("Unable to create a test video with OpenCV VideoWriter.")

    def test_analyze_frame_bbox_matches_reference(self):
        frame = np.zeros((6, 7, 3), dtype=np.uint8)
        frame[1:5, 2:6] = (10, 220, 230)
        frame[0, 0] = (10, 220, 230)
        bbox = (2, 1, 3, 3)

        actual = analysis.analyze_frame(frame, bbox=bbox)
        expected = reference_analyze_frame(frame, bbox=bbox)

        np.testing.assert_array_equal(actual, expected)

    def test_video_aggregation_with_bbox_matches_reference(self):
        source_frames = []

        frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_a[1:4, 1:4] = (50, 210, 210)
        frame_a[6, 6] = (50, 210, 210)
        source_frames.append(frame_a)

        frame_b = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_b[2:6, 2:6] = (20, 240, 240)
        source_frames.append(frame_b)

        frame_c = np.zeros((8, 8, 3), dtype=np.uint8)
        frame_c[0:2, 0:2] = (20, 240, 240)
        frame_c[3:7, 3:7] = (120, 240, 240)
        source_frames.append(frame_c)

        bbox = (1, 1, 5, 5)
        temp_dir = Path(__file__).resolve().parent / "_tmp"
        temp_dir.mkdir(exist_ok=True)
        video_path = self._write_test_video(temp_dir / "analysis_fixture", source_frames)
        try:
            cap = cv2.VideoCapture(str(video_path))
            decoded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                decoded_frames.append(frame)
            cap.release()

            expected = np.zeros(decoded_frames[0].shape[:2], dtype=np.uint32)
            for frame in decoded_frames:
                expected += reference_analyze_frame(frame, bbox=bbox)

            actual = analysis.get_total_yellow_pixels_from_video(str(video_path), bbox=bbox)
        finally:
            if video_path.exists():
                video_path.unlink()

        np.testing.assert_array_equal(actual, expected)

    def test_projected_points_inside_side_hexes_are_excluded(self):
        field_width = 3901
        field_height = 1583
        bbox = (0, 0, 1, 1)
        projection_matrix = np.eye(3, dtype=np.float32)

        inside_left_hex = (int(round(field_width * 0.3139)), int(round(field_height * 0.4991)))
        inside_right_hex = (int(round(field_width * 0.6506)), int(round(field_height * 0.4991)))
        open_field = (int(round(field_width * 0.5)), int(round(field_height * 0.5)))

        points = processor_cli.project_fuel_points_from_centers(
            [inside_left_hex, inside_right_hex, open_field],
            bbox=bbox,
            projection_matrix=projection_matrix,
            field_width=field_width,
            field_height=field_height,
        )

        expected_open_field = [
            int(round((open_field[0] / field_width) * 10000)),
            int(round((open_field[1] / field_height) * 10000)),
            processor_cli.FIELD_MAP_FUEL_RADIUS,
        ]
        self.assertEqual(points, [expected_open_field])

    def test_temporal_stabilizer_holds_stationary_fuel_through_short_dropout(self):
        stabilizer = processor_cli.FuelTemporalStabilizer(max_misses=2, history_size=4)

        first, bad_first = stabilizer.stabilize([(10, 10), (30, 30)])
        second, bad_second = stabilizer.stabilize([(10, 10)])
        third, bad_third = stabilizer.stabilize([(10, 10), (30, 30)])

        self.assertFalse(bad_first)
        self.assertFalse(bad_second)
        self.assertFalse(bad_third)
        self.assertEqual(sorted(first), [(10, 10), (30, 30)])
        self.assertEqual(sorted(second), [(10, 10), (30, 30)])
        self.assertEqual(sorted(third), [(10, 10), (30, 30)])

    def test_temporal_stabilizer_reuses_previous_centers_on_bad_count_frame(self):
        stabilizer = processor_cli.FuelTemporalStabilizer(
            history_size=4,
            min_baseline_count=3,
            min_count_delta=4,
            count_delta_ratio=0.4,
        )

        baseline = [(i * 20, 20) for i in range(10)]
        for _ in range(4):
            stable, bad = stabilizer.stabilize(baseline)
            self.assertFalse(bad)
            self.assertEqual(stable, baseline)

        noisy_frame = baseline + [(200 + i * 10, 60) for i in range(7)]
        stable, bad = stabilizer.stabilize(noisy_frame)

        self.assertTrue(bad)
        self.assertEqual(stable, baseline)

    def test_compute_frame_stride_caps_processing_rate(self):
        self.assertEqual(processor_cli.compute_frame_stride(30.0, 15.0), 2)
        self.assertEqual(processor_cli.compute_frame_stride(24.714, 15.0), 2)
        self.assertEqual(processor_cli.compute_frame_stride(14.0, 15.0), 1)


if __name__ == "__main__":
    unittest.main()
