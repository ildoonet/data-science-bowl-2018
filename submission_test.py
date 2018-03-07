import unittest
import numpy as np
import time

from submission import rle_encoding, get_iou, get_metric, get_iou1, get_iou2


class TestSubmission(unittest.TestCase):
    def setUp(self):
        self.a = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        self.b = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])
        self.a = self.a[..., np.newaxis]
        self.b = self.b[..., np.newaxis]

        self.instances = np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ],
        ])

    def test_rle(self):
        a = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ])
        a = a[..., np.newaxis]

        rles, cnt = rle_encoding(a)
        self.assertEqual(cnt, 5)
        self.assertListEqual(rles, [3, 2, 19, 2, 25, 1])

    def test_iou(self):
        iou = get_iou(self.a, self.b)
        self.assertAlmostEqual(iou, 0.5, 0.001)

    def test_iou_speed(self):
        t = time.time()
        for _ in range(10000):
            iou1 = get_iou1(self.a, self.b)
        t1 = time.time() - t

        t = time.time()
        for _ in range(10000):
            iou2 = get_iou2(self.a, self.b)
        t2 = time.time() - t

        self.assertEqual(iou1, iou2)
        print('t_iou1=', t1, 't_iou2=', t2)

    def test_metric(self):
        labels = np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ])
        tp, fp, fn = get_metric(self.instances, labels, thr_list=[0.5])
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

        tp, fp, fn = get_metric(self.instances, labels, thr_list=[0.95])
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 2)
        self.assertEqual(fn, 1)

        labels = np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ])
        tp, fp, fn = get_metric(self.instances, labels, thr_list=[0.5])
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

        labels = np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ])
        tp, fp, fn = get_metric(self.instances, labels, thr_list=[0.5])
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

        labels = np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ]
        ])
        tp, fp, fn = get_metric(self.instances, labels, thr_list=[0.5])
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)
