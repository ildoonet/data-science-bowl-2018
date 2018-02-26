import unittest
import numpy as np

from network import Network


class TestNetwork(unittest.TestCase):
    def test_visualize(self):
        pass

    def test_sliding_window(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cascades, _ = Network.sliding_window(img, 100, 0.0)
        self.assertEqual(len(cascades), 9)
        self.assertListEqual(list(cascades[0].shape), [100, 100, 3])

        cascades, _ = Network.sliding_window(img, 100, 0.5)
        self.assertEqual(len(cascades), 25)
        self.assertListEqual(list(cascades[0].shape), [100, 100, 3])

        img = np.zeros((224, 224, 1), dtype=np.float32)
        cascades, _ = Network.sliding_window(img, 224, 0.2)
        self.assertEqual(len(cascades), 1)
        self.assertListEqual(list(cascades[0].shape), [224, 224, 1])
        self.assertTrue(np.array_equal(cascades[0], img))

    def test_parse_merged_output(self):
        merged_output = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        instances = Network.parse_merged_output(merged_output)
        self.assertEqual(len(instances), 5)

        merged_output = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        instances = Network.parse_merged_output(merged_output)
        self.assertEqual(len(instances), 3)

    def test_resize_instances(self):
        merged_output = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8)
        merged_output = merged_output[..., np.newaxis]
        instances = Network.parse_merged_output(merged_output)
        resized = Network.resize_instances(instances, (20, 30))
        self.assertEqual(len(resized), 5)
        self.assertEqual(resized[0].shape[0], 20)
        self.assertEqual(resized[0].shape[1], 30)
        self.assertEqual(resized[0].shape[2], 1)
