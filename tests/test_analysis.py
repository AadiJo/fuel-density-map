import unittest
from pathlib import Path
import shutil
import subprocess
import tempfile

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
        if not hasattr(cv2, "VideoWriter"):
            ffmpeg = shutil.which('ffmpeg') or processor_cli.ffmpeg_binary()
            if not ffmpeg:
                self.skipTest("ffmpeg is required when OpenCV VideoWriter is unavailable.")
            candidate = path.with_suffix(".mp4")
            command = [
                ffmpeg,
                '-hide_banner',
                '-loglevel',
                'error',
                '-y',
                '-f',
                'rawvideo',
                '-pix_fmt',
                'bgr24',
                '-s',
                f'{frames[0].shape[1]}x{frames[0].shape[0]}',
                '-r',
                str(fps),
                '-i',
                'pipe:0',
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                str(candidate),
            ]
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert proc.stdin is not None
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            proc.stdin.close()
            stderr_text = ""
            if proc.stderr is not None:
                stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()
            return_code = proc.wait()
            if return_code != 0:
                self.fail(stderr_text or "Unable to create a test video with FFmpeg.")
            return candidate

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
            decoded_frames = []
            if hasattr(cv2, "VideoCapture"):
                cap = cv2.VideoCapture(str(video_path))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    decoded_frames.append(frame)
                cap.release()
            else:
                metadata = processor_cli.probe_video_metadata(str(video_path))
                reader = processor_cli.FFmpegFrameReader(
                    str(video_path),
                    metadata["width"],
                    metadata["height"],
                    stride=1,
                )
                try:
                    decoded_frames = list(reader)
                finally:
                    reader.close()

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

    def test_temporal_stabilizer_smooths_motion_and_predicts_short_dropout(self):
        stabilizer = processor_cli.FuelTemporalStabilizer(
            max_misses=2,
            history_size=4,
            min_baseline_count=999,
            smoothing_alpha=0.5,
            velocity_alpha=0.5,
            velocity_decay=0.5,
        )

        first, bad_first = stabilizer.stabilize([(10, 10)])
        second, bad_second = stabilizer.stabilize([(20, 10)])
        third, bad_third = stabilizer.stabilize([])

        self.assertFalse(bad_first)
        self.assertFalse(bad_second)
        self.assertFalse(bad_third)
        self.assertEqual(first, [(10, 10)])
        self.assertEqual(second, [(15, 10)])
        self.assertEqual(third, [(20, 10)])

    def test_temporal_stabilizer_updates_multiple_tracks_without_type_errors(self):
        stabilizer = processor_cli.FuelTemporalStabilizer(
            max_misses=2,
            history_size=4,
            min_baseline_count=999,
            smoothing_alpha=0.5,
            velocity_alpha=0.5,
            velocity_decay=0.5,
        )

        first, bad_first = stabilizer.stabilize([(10, 10), (30, 30), (50, 50)])
        second, bad_second = stabilizer.stabilize([(12, 10), (28, 30), (52, 50)])
        third, bad_third = stabilizer.stabilize([(14, 10), (26, 30), (54, 50)])

        self.assertFalse(bad_first)
        self.assertFalse(bad_second)
        self.assertFalse(bad_third)
        self.assertEqual(first, [(10, 10), (30, 30), (50, 50)])
        self.assertEqual(second, [(11, 10), (29, 30), (51, 50)])
        self.assertEqual(len(third), 3)
        self.assertTrue(all(isinstance(x, int) and isinstance(y, int) for x, y in third))
        self.assertTrue(any(x >= 12 and y == 10 for x, y in third))
        self.assertTrue(any(x <= 27 and y == 30 for x, y in third))
        self.assertTrue(any(x >= 52 and y == 50 for x, y in third))

    def test_temporal_stabilizer_ignores_malformed_centers(self):
        stabilizer = processor_cli.FuelTemporalStabilizer(
            max_misses=2,
            history_size=4,
            min_baseline_count=999,
        )

        first, bad_first = stabilizer.stabilize([(10, 10), (30, 30)])
        second, bad_second = stabilizer.stabilize([(12, 10), (stabilizer, 30), (32, 30), ("bad", 5)])

        self.assertFalse(bad_first)
        self.assertFalse(bad_second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertTrue(any(x >= 10 and y == 10 for x, y in second))
        self.assertTrue(any(x >= 30 and y == 30 for x, y in second))

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

    def test_peak_detector_respects_budget(self):
        mask = np.zeros((120, 120), dtype=np.uint8)
        for row in range(15, 110, 20):
            for col in range(15, 110, 20):
                cv2.circle(mask, (col, row), 4, 255, -1)

        detector = processor_cli.PeakBallDetector(working_scale=1.0, detector_budget=5)
        centers, saturated = detector.detect(mask)

        self.assertTrue(saturated)
        self.assertLessEqual(len(centers), 5)
        self.assertGreater(len(centers), 0)

    def test_hybrid_detector_falls_back_to_legacy_when_peak_detector_misses(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[40, 40] = 255

        peak = processor_cli.ball_centers_from_mask(mask, detector_mode="peak")
        hybrid = processor_cli.ball_centers_from_mask(mask, detector_mode="hybrid")
        legacy = processor_cli.ball_centers_from_mask(mask, detector_mode="legacy")

        self.assertEqual(peak, [])
        self.assertEqual(hybrid, legacy)
        self.assertEqual(hybrid, [(40, 40)])

    def test_legacy_detector_splits_dense_merged_blob_into_multiple_centers(self):
        mask = np.zeros((140, 140), dtype=np.uint8)
        points = [
            (50, 70),
            (60, 70),
            (70, 70),
            (80, 70),
            (90, 70),
            (55, 80),
            (65, 80),
            (75, 80),
            (85, 80),
        ]
        for point in points:
            cv2.circle(mask, point, 7, 255, -1)

        legacy = processor_cli.ball_centers_from_mask(mask, detector_mode="legacy")

        self.assertGreaterEqual(len(legacy), 8)

    def test_hybrid_detector_prefers_richer_split_for_dense_merged_blob(self):
        mask = np.zeros((140, 140), dtype=np.uint8)
        points = [
            (50, 70),
            (60, 70),
            (70, 70),
            (80, 70),
            (90, 70),
            (55, 80),
            (65, 80),
            (75, 80),
            (85, 80),
        ]
        for point in points:
            cv2.circle(mask, point, 7, 255, -1)

        peak = processor_cli.ball_centers_from_mask(mask, detector_mode="peak")
        hybrid = processor_cli.ball_centers_from_mask(mask, detector_mode="hybrid")
        legacy = processor_cli.ball_centers_from_mask(mask, detector_mode="legacy")

        self.assertEqual(len(peak), 1)
        self.assertEqual(hybrid, legacy)
        self.assertGreaterEqual(len(hybrid), 8)

    def test_temporal_stabilizer_caps_active_tracks(self):
        stabilizer = processor_cli.FuelTemporalStabilizer(
            max_active_tracks=10,
            max_track_age=4,
            max_misses=1,
            history_size=4,
            min_baseline_count=999,
        )
        centers = [(x * 12, 20) for x in range(30)]

        first, bad_first = stabilizer.stabilize(centers, saturated=True)
        second, bad_second = stabilizer.stabilize(centers, saturated=True)

        self.assertFalse(bad_first)
        self.assertFalse(bad_second)
        self.assertLessEqual(len(first), 10)
        self.assertLessEqual(len(second), 10)

    def test_write_dynamic_assets_video_output_creates_overlay_video(self):
        if shutil.which('ffmpeg') is None and not processor_cli.ffmpeg_binary():
            self.skipTest('ffmpeg is not available')

        source_frames = []
        for idx in range(4):
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            frame[20 + idx:24 + idx, 30 + idx:34 + idx] = (30, 240, 240)
            source_frames.append(frame)

        temp_dir = Path(tempfile.mkdtemp(prefix='fdm-video-out-'))
        video_path = self._write_test_video(temp_dir / "overlay_video_fixture", source_frames, fps=4.0)
        try:
            session_dir = temp_dir / 'session'
            result = processor_cli.write_dynamic_assets(
                str(video_path),
                str(session_dir),
                str(session_dir / 'field-map.json'),
                str(session_dir / 'raw-data.txt'),
                bbox=(0, 0, 100, 80),
                field_quad=None,
                average_display_color=(255, 0, 255),
                field_image_path=processor_cli.DEFAULT_FIELD_IMAGE_PATH,
                target_process_fps=4.0,
                overlay_output='video',
                backend_name='cpu',
                working_scale=1.0,
                detector_budget=32,
                max_active_tracks=64,
            )

            self.assertEqual(result["overlayOutput"], "video")
            self.assertEqual(result["overlayOutputMeta"]["overlayVideoFileName"], "overlay-video.mp4")
            self.assertTrue((session_dir / 'overlay-video.mp4').exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
