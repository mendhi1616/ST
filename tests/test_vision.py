import unittest
import numpy as np
import cv2
import os
import tempfile
import shutil
from src.eyes_detection import analyze_tadpole_microscope, read_image

class TestVision(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.test_img_path = os.path.join(self.test_dir, "test_tadpole.jpg")

        # Background: Gray ~200
        img = np.ones((500, 500, 3), dtype=np.uint8) * 200

        # Body: Darker gray ~100
        center = (250, 250)
        axes = (100, 60)
        cv2.ellipse(img, center, axes, 0, 0, 360, (100, 100, 100), -1)

        # Eyes: Very dark ~20 inside body.
        eye1_center = (250 - 20, 250 - 15)
        cv2.circle(img, eye1_center, 6, (20, 20, 20), -1)

        eye2_center = (250 + 20, 250 - 15)
        cv2.circle(img, eye2_center, 6, (20, 20, 20), -1)

        cv2.imwrite(self.test_img_path, img)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_read_image(self):
        img = read_image(self.test_img_path)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (500, 500, 3))

    def test_analyze_tadpole(self):
        img, body_len, eye_dist, status = analyze_tadpole_microscope(self.test_img_path, debug=False)

        self.assertIsNotNone(img)
        self.assertGreater(body_len, 0)
        self.assertGreater(eye_dist, 0)
        self.assertIn("Succès", status)

    def test_analyze_tadpole_fail(self):
        # Test with a blank black image
        blank_path = os.path.join(self.test_dir, "blank.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(blank_path, img)

        _, body_len, eye_dist, status = analyze_tadpole_microscope(blank_path, debug=False)
        self.assertNotEqual(status, "Succès")

if __name__ == '__main__':
    unittest.main()
