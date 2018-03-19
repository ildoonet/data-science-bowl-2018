import os
import cv2
import logging
import json
import numpy as np
import pandas as pd
import time

import sys

from hyperparams import HyperParams
from kaggle.api.kaggle_api_extended import KaggleApi


logger = logging.getLogger('submission')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


def rle_encoding(x):
    """
    reference : https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
    :param x: (h, w, 1) numpy
    :return: run length encoded list
    """
    dots = np.where(x.T.flatten() == 1)[0]
    rle = []
    prev = -2
    cnt = 0
    for b in dots:
        if b > prev + 1:
            rle.extend((b + 1, 0))
        rle[-1] += 1
        cnt += 1
        prev = b
    return rle, cnt


class KaggleSubmission:
    BASEPATH = os.path.dirname(os.path.realpath(__file__)) + "/submissions"
    CNAME = 'data-science-bowl-2018'

    def __init__(self, name):
        self.name = name
        self.test_ids = []
        self.rles = []

        try:
            os.mkdir(os.path.join(KaggleSubmission.BASEPATH, self.name))
            os.mkdir(os.path.join(KaggleSubmission.BASEPATH, self.name, 'valid'))
        except:
            pass

    def save_valid_image(self, idx, image):
        cv2.imwrite(os.path.join(KaggleSubmission.BASEPATH, self.name, 'valid', idx + '.jpg'), image)

    def save_image(self, idx, image):
        cv2.imwrite(os.path.join(KaggleSubmission.BASEPATH, self.name, idx + '.jpg'), image)

    def add_result(self, idx, instances):
        """

        :param idx: test sample id
        :param instances: list of (h, w, 1) numpy containing
        """
        for instance in instances:
            rles, cnt = rle_encoding(instance)

            if cnt < 3:
                continue
            assert len(rles) % 2 == 0

            self.test_ids.append(idx)
            self.rles.append(rles)

    def get_filepath(self):
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'submission.csv')
        return filepath

    def get_confpath(self):
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'config.json')
        return filepath

    def save(self):
        sub = pd.DataFrame()
        sub['ImageId'] = self.test_ids
        sub['EncodedPixels'] = pd.Series(self.rles).apply(lambda x: ' '.join(str(y) for y in x))

        filepath = self.get_filepath()
        f = open(filepath, 'w')
        f.close()
        sub.to_csv(filepath, index=False)
        logger.info('%s saved at %s.' % (self.name, filepath))

        filepath = self.get_confpath()
        f = open(filepath, 'w')
        a = json.dumps(HyperParams.get().__dict__, indent=4)
        f.write(a)
        f.close()

    def submit_result(self, submit_msg='KakaoAutoML'):
        logger.info('kaggle.submit_result: initialization')
        api_client = KaggleApi()
        api_client.authenticate()
        submissions = api_client.competitionSubmissions(KaggleSubmission.CNAME)
        last_idx = submissions[0].ref if len(submissions) > 0 else -1

        # submit
        logger.info('kaggle.submit_result: trying to submit @ %s' % self.get_filepath())
        submit_result = api_client.competitionSubmit(self.get_filepath(), submit_msg, KaggleSubmission.CNAME)
        logger.info('kaggle.submit_result: submitted!')

        # wait for the updated LB
        wait_interval = 10   # in seconds
        for _ in range(60 // wait_interval * 5):
            submissions = api_client.competitionSubmissions(KaggleSubmission.CNAME)
            if len(submissions) == 0:
                continue
            if submissions[0].status == 'complete' and submissions[0].ref != last_idx:
                # updated
                logger.info('kaggle.submit_result: LB Score Updated!')
                return submit_result, submissions[0]
            time.sleep(wait_interval)
        logger.info('kaggle.submit_result: LB Score NOT Updated!')
        return submit_result, None


def get_iou1(a, b):
    if len(a.shape) == 2:
        a = a[..., np.newaxis]
    if len(b.shape) == 2:
        b = b[..., np.newaxis]
    a = a.astype(np.bool_)
    b = b.astype(np.bool_)

    intersection = np.sum(np.logical_and(a, b), dtype=np.float32)
    if intersection == 0:
        return 0.0
    union = np.sum(np.logical_or(a, b), dtype=np.float32)
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou


def get_iou2(a, b):
    if len(a.shape) == 2:
        a = a[..., np.newaxis]
    if len(b.shape) == 2:
        b = b[..., np.newaxis]
    a[a > 0] = 1.
    b[b > 0] = 1.
    intersection = a * b
    union = a + b
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    if intersection == 0:
        return 0.0
    union = np.sum(union)
    if union == 0:
        return 0.0
    return intersection / union


get_iou = get_iou2


def get_metric(instances, label_trues, thr_list):
    """
    :param instances:  list of (h, w) numpy array
    :param label_trues:  list of (h, w) numpy array
    :param thr_list:
    :return:
    """
    if len(label_trues) == 0:
        return 0.0

    cnt_tps = np.zeros((len(thr_list)), dtype=np.int32)
    cnt_fps = np.zeros((len(thr_list)), dtype=np.int32)
    cnt_ass = np.zeros((len(thr_list), len(label_trues)), dtype=np.int32)
    for label_pred in instances:
        max_label_idx = -1
        max_label_iou = 0.0
        max_label = None
        for idx_label, label_true in enumerate(label_trues):
            # measure ious between label_preds & label_true
            iou = get_iou(label_true, label_pred)

            if iou > max_label_iou:
                max_label_idx = idx_label
                max_label_iou = iou
                max_label = label_true

        if max_label is None:
            # false positive
            cnt_fps = cnt_fps + 1
        else:
            for th_idx, thr in enumerate(thr_list):
                if max_label_iou > thr:
                    cnt_tps[th_idx] += 1
                    cnt_ass[th_idx][max_label_idx] = 1
                else:
                    cnt_fps[th_idx] += 1

    cnt_fns = len(label_trues) - np.sum(cnt_ass, axis=1)

    # return the metric
    return cnt_tps, cnt_fps, cnt_fns


def get_multiple_metric(thr_list, instances, label_trues):
    """
    :param thr_list:
    :param instances:  list of (h, w) numpy array
    :param label_trues:  list of (h, w) numpy array
    :return:
    """
    t = time.time()
    cnt_tp, cnt_fp, cnt_fn = get_metric(instances, label_trues, thr_list)
    # print('thr_miou', time.time() - t)
    return cnt_tp, cnt_fp, cnt_fn
