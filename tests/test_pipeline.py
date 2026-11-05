import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from acne_demo.inpainting import blend_inside_mask, fast_inpaint
from acne_demo.pipeline import AcnePipeline, Detection, make_mask, normalize_image


class ImagePipelineTests(unittest.TestCase):
    def test_mask_clips_boxes_and_selection(self):
        boxes = [[-10, -5, 6, 8, 0.9, 0], [15, 12, 30, 30, 0.8, 0]]
        mask = make_mask((20, 20, 3), boxes, selected=[0], padding=1)
        self.assertTrue(np.all(mask[:9, :7] == 255))
        self.assertEqual(np.count_nonzero(mask), 63)
        self.assertFalse(np.any(make_mask((20, 20), boxes, [], 0)))

    def test_blending_preserves_unselected_pixels_exactly(self):
        rgb = np.random.default_rng(42).integers(0, 256, (40, 50, 3), dtype=np.uint8)
        mask = np.zeros((40, 50), np.uint8)
        mask[10:25, 15:30] = 255
        result = blend_inside_mask(rgb, fast_inpaint(rgb, mask), mask)
        np.testing.assert_array_equal(result[mask == 0], rgb[mask == 0])
        np.testing.assert_array_equal(blend_inside_mask(rgb, np.zeros_like(rgb), mask, 0), rgb)

    def test_empty_selection_does_not_load_restoration(self):
        pipeline = AcnePipeline.__new__(AcnePipeline)
        detection = Detection(np.full((32, 32, 3), 100, np.uint8), [[2, 2, 8, 8, 0.9, 0]], (32, 32), 0)
        with patch("acne_demo.pipeline.DeepFill", side_effect=AssertionError("Should not load")):
            result, mask, _ = pipeline.restore(detection, selected=[])
        np.testing.assert_array_equal(result, detection.rgb)
        self.assertFalse(np.any(mask))

    def test_resize_preserves_aspect_ratio(self):
        result, original_size = normalize_image(Image.new("RGB", (2400, 1200)))
        self.assertEqual(result.shape, (800, 1600, 3))
        self.assertEqual(original_size, (2400, 1200))

    def test_bad_image_rejected(self):
        for image in [None, np.zeros((10, 10)), np.ones((10, 10, 3), dtype=float)]:
            with self.assertRaises(ValueError):
                normalize_image(image)


if __name__ == "__main__":
    unittest.main()
