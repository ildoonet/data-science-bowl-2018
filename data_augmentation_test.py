import unittest
import numpy as np

from data_augmentation import resize_shortedge, random_crop, center_crop, resize_shortedge_if_small, \
    flip_lr
from data_feeder import CellImageData, master_dir_train


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        # 256, 320, 3
        self.d = CellImageData('a6515d73077866808ad4cb837ecdac33612527b8a1041e82135e40fce2bb9380', path=master_dir_train)

    def testResize(self):
        d = resize_shortedge(self.d, 224)
        self.assertEqual(
            d.image(is_gray=False).shape[0],
            224
        )

    def test_aug_flip_lr(self):
        points = np.where(self.d.masks[0] > 0.8)
        flipped = flip_lr(self.d)

        for y, x in zip(*points):
            for val1, val2 in zip(self.d.img[y, x], flipped.img[y, -x-1]):
                self.assertAlmostEqual(float(val1), float(val2), delta=2)
            val1, val2 = self.d.masks[0][y, x], flipped.masks[0][y, -x - 1]
            self.assertAlmostEqual(float(val1), float(val2), delta=2)

    def test_random_crop(self):
        d = random_crop(self.d, 224, 224)
        self.assertListEqual(
            list(d.image(is_gray=False).shape),
            [224, 224, 3]
        )

    def test_center_crop(self):
        d = center_crop(self.d, 224, 224)
        self.assertListEqual(
            list(d.image(is_gray=False).shape),
            [224, 224, 3]
        )
        # TODO : centered?

    def test_resize_shortedge_if_small(self):
        # not changed, since its size is larger than target_size
        d = resize_shortedge_if_small(self.d, 224)
        self.assertGreater(d.image(is_gray=False).shape[0], 224)
        self.assertGreater(d.image(is_gray=False).shape[1], 224)

        # should be changed
        d = resize_shortedge(self.d, 120)   # generate a small image
        d = resize_shortedge_if_small(d, 224)
        self.assertEqual(d.image(is_gray=False).shape[0], 224)
