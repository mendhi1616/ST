import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import os
import tempfile
import shutil
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from eyes_detection import analyze_tadpole_microscope
from utils import read_image_with_unicode

class TestVision(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_img_path = os.path.join(self.test_dir, "test_tadpole.jpg")

        # Create a dummy image
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        # Draw a "tadpole"
        cv2.circle(img, (250, 250), 50, (100, 100, 100), -1)
        cv2.imwrite(self.test_img_path, img)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('eyes_detection.segment_tadpole_sam2')
    @patch('eyes_detection.detect_eyes_ilastik')
    @patch('eyes_detection.classify_orientation')
    def test_analyze_tadpole_success(self, mock_orient, mock_eyes, mock_seg):
        # Mock segmentation: Return a circle mask in the center
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(mask, (250, 250), 50, 255, -1)
        mock_seg.return_value = mask

        # Mock eyes: Return valid eye coordinates
        mock_eyes.return_value = ((240, 240), (260, 240), 20.0, "Success")

        # Mock orientation
        mock_orient.return_value = ("profile_ok", 0.95)

        img, body_len, eye_dist, status, orientation = analyze_tadpole_microscope(self.test_img_path, debug=False)

        self.assertIsNotNone(img)
        self.assertTrue(body_len > 0)
        self.assertEqual(eye_dist, 20.0)
        self.assertEqual(status, "Success")
        self.assertEqual(orientation, "profile_ok")

    @patch('eyes_detection.segment_tadpole_sam2')
    def test_analyze_tadpole_seg_fail(self, mock_seg):
        # Mock segmentation: Return empty mask
        mock_seg.return_value = np.zeros((500, 500), dtype=np.uint8)

        # analyze_tadpole_microscope returns 5 values now
        img, body_len, eye_dist, status, orientation = analyze_tadpole_microscope(self.test_img_path, debug=False)

        self.assertEqual(body_len, 0.0)
        self.assertIn("Body not detected", status)
        self.assertEqual(orientation, "unknown")

    @patch('eyes_detection.segment_tadpole_sam2')
    @patch('eyes_detection.detect_eyes_ilastik')
    @patch('eyes_detection.classify_orientation')
    def test_analyze_tadpole_eyes_fail(self, mock_orient, mock_eyes, mock_seg):
        # Mock valid segmentation
        mask = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(mask, (250, 250), 50, 255, -1)
        mock_seg.return_value = mask

        # Mock eyes failure
        mock_eyes.return_value = (None, None, 0.0, "Eyes not detected")

        # Mock orientation
        mock_orient.return_value = ("profile_ok", 0.95)

        img, body_len, eye_dist, status, orientation = analyze_tadpole_microscope(self.test_img_path, debug=False)

        self.assertTrue(body_len > 0)
        self.assertEqual(eye_dist, 0.0)
        self.assertIn("Eyes not detected", status)
        self.assertEqual(orientation, "profile_ok")

if __name__ == '__main__':
    unittest.main()
