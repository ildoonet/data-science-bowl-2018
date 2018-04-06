import unittest

import cv2
import numpy as np

from data_augmentation import resize_shortedge, random_crop, center_crop, resize_shortedge_if_small, \
    flip, data_to_elastic_transform, random_color2, mask_size_normalize, get_max_size_of_masks, crop, crop_mirror, \
    random_add_thick_area, random_transparent
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
        original = self.d.img.copy()
        orig_mask = self.d.masks
        flipped = flip(self.d, orientation=1)

        for y, x in zip(*points):
            for val1, val2 in zip(original[y, x], flipped.img[y, -x-1]):
                self.assertAlmostEqual(float(val1), float(val2), delta=2)
            val1, val2 = orig_mask[0][y, x], flipped.masks[0][y, -x - 1]
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

    def test_crop_mirror(self):
        img = crop_mirror(self.d.img, 0, 0, 224, 224)
        self.assertListEqual(
            list(img.shape),
            [224, 224, 3]
        )
        cropped = self.d.img[0:224, 0:224, :]
        self.assertTrue(np.array_equal(img, cropped))

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
            # cv2.imshow('random_color2', d.img)
            # cv2.waitKeyEx(0)

    def test_size_normalization(self):
        for target_cell_size in [30, 50, 100, 150]:
            d = mask_size_normalize(self.d, target_cell_size)

            self.assertAlmostEqual(get_max_size_of_masks(d.masks), target_cell_size, delta=2.0)

            # cv2.imshow('size_normalization', d.img)
            # cv2.waitKeyEx(0)

    def test_add_thick_area(self):
        cv2.imshow('original', self.d.img)
        cv2.waitKey(0)
        data = random_add_thick_area(self.d)
        cv2.imshow('image', data.img)
        cv2.waitKey(0)

    def test_transparent(self):
        cv2.imshow('original', self.d.img)
        cv2.waitKey(0)
        data = random_transparent(self.d)
        cv2.imshow('image', data.img)
        cv2.waitKey(0)


if __name__ == "__main__":
    unittest.main()
