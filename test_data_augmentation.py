import unittest

from data_augmentation import resize_shortedge, random_crop
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

    def test_rcrop(self):
        d = random_crop(self.d, 224, 224)
        self.assertListEqual(
            list(d.image(is_gray=False).shape),
            [224, 224, 3]
        )