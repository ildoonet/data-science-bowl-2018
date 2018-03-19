import unittest

import cv2
import numpy as np

from data_augmentation import resize_shortedge, random_crop, center_crop, resize_shortedge_if_small, \
    flip, data_to_elastic_transform, random_color2
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
        flipped = flip(self.d)

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

    def test_elastic_transformation(self):
        image, masks = data_to_elastic_transform(
            self.d, self.d.img.shape[1] * 2,
            self.d.img.shape[1] * 0.08, self.d.img.shape[1] * 0.08
        )

        self.assertEqual(self.d.img.shape[0], image.shape[0])
        self.assertEqual(self.d.masks[0].shape[0], masks[0].shape[0])

        # for visual inspection
        # cv2.imshow('before-elastic-tran-img', self.d.img)
        # cv2.imshow('before-elastic-tran-mask', self.d.masks[0] * 255)
        #
        # cv2.imshow('after-elastic-tran-img', image)
        # cv2.imshow('after-elastic-tran-mask', masks[0] * 255)
        #
        # cv2.waitKeyEx(0)

    def test_random_color2(self):
        for _ in range(5):
            d = random_color2(self.d)
            cv2.imshow('random_color2', d.img)
            cv2.waitKeyEx(0)
