import unittest

import cv2
import numpy as np

import analysis
import processor_cli as pc


class TestFuelMask(unittest.TestCase):
    def test_fuel_mask_detects_bright_yellow(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(img, (32, 32), 6, (0, 220, 255), -1, lineType=cv2.LINE_8)
        m = analysis.fuel_mask_bgr(img)
        self.assertGreater(int(m.sum()), 0)

    def test_analyze_frame_matches_fuel_mask_bbox(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        frame[30:50, 30:50] = (0, 220, 255)
        bbox = (10, 10, 60, 60)
        out = analysis.analyze_frame(frame, bbox=bbox, use_motion_gate=False)
        sl = analysis.fuel_mask_bgr(frame[10:70, 10:70]) > 0
        self.assertTrue(np.array_equal(out[10:70, 10:70].astype(bool), sl))


class TestBallCentersShapeFilter(unittest.TestCase):
    def test_centers_on_single_blob(self):
        m = np.zeros((40, 40), dtype=np.uint8)
        cv2.circle(m, (20, 20), 4, 255, -1)
        c = pc.ball_centers_from_mask(m, shape_filter=True)
        self.assertEqual(len(c), 1)

    def test_large_compact_clump_not_dropped(self):
        """Regression: area cap used to reject entire piles; compact blobs must yield centers."""
        m = np.zeros((220, 220), dtype=np.uint8)
        cv2.circle(m, (110, 110), 85, 255, -1)
        c = pc.ball_centers_from_mask(m, shape_filter=True)
        self.assertGreater(len(c), 3)


if __name__ == "__main__":
    unittest.main()
