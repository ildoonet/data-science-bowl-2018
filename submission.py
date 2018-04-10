import os
from collections import OrderedDict

import cv2
import logging
import json
import numpy as np
import pandas as pd
import time
import pickle

import sys

from data_augmentation import get_rect_of_mask
from data_feeder import CellImageDataManagerTest, CellImageDataManagerValid
from hyperparams import HyperParams
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except:
    logging.warning('~/.kaggle/kaggle.json not set. Can not submit to kaggle automatically.')


logger = logging.getLogger('submission')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


thr_list = np.arange(0.5, 1.0, 0.05)


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
    try:
        rmin1, rmax1, cmin1, cmax1 = get_rect_of_mask(a)
        rmin2, rmax2, cmin2, cmax2 = get_rect_of_mask(b)
        if not ((rmin1 <= rmin2 <= rmax1 or rmin1 <= rmax2 <= rmax1) or
                    (cmin1 <= cmin2 <= cmax1 or cmin1 <= cmax2 <= cmax1) or
                    (rmin2 <= rmin1 <= rmax2 or rmin2 <= rmax1 <= rmax2) or
                    (cmin2 <= cmin1 <= cmax2 or cmin2 <= cmax1 <= cmax2)):
            return 0.0
    except:
        pass

    if len(a.shape) == 2:
        a = a[..., np.newaxis]
    if len(b.shape) == 2:
        b = b[..., np.newaxis]
    intersection = a & b
    intersection = np.sum(intersection)
    if intersection == 0:
        return 0.0
    union = a | b
    union = np.sum(union)
    if union == 0:
        return 0.0
    return intersection / union


# get_iou2 is faster version of get_iou1
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
    max_found = set()
    for label_pred in instances:
        max_label_idx = -1
        max_label_iou = 0.0
        max_label = None
        for idx_label, label_true in enumerate(label_trues):
            if idx_label in max_found:
                continue
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
            max_found.add(idx_label)
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


class KaggleSubmission:
    BASEPATH = os.path.dirname(os.path.realpath(__file__)) + "/submissions"
    CNAME = 'data-science-bowl-2018'

    def __init__(self, name):
        self.name = name
        self.test_ids = []
        self.rles = []
        self.train_scores = OrderedDict()
        self.valid_scores = OrderedDict()
        self.test_scores = OrderedDict()
        self.valid_instances = {}   # key : id -> (instances, scores)
        self.test_instances = {}

        logger.info('creating: %s' % os.path.join(KaggleSubmission.BASEPATH, self.name))
        os.makedirs(os.path.join(KaggleSubmission.BASEPATH, self.name), exist_ok=True)
        logger.info('creating: %s' % os.path.join(KaggleSubmission.BASEPATH, self.name, 'valid'))
        os.makedirs(os.path.join(KaggleSubmission.BASEPATH, self.name, 'valid'), exist_ok=True)
        logger.info('creating: %s' % os.path.join(KaggleSubmission.BASEPATH, self.name, 'train'))
        os.makedirs(os.path.join(KaggleSubmission.BASEPATH, self.name, 'train'), exist_ok=True)

    def save_train_image(self, idx, image, loss=0.0, score=0.0, score_desc=[]):
        cv2.imwrite(os.path.join(KaggleSubmission.BASEPATH, self.name, 'train', idx + '.jpg'), image)

        if isinstance(idx, bytes):
            idx = idx.decode("utf-8")
        self.train_scores[idx] = (loss, score, score_desc)

    def save_valid_image(self, idx, image, loss=0.0, score=0.0, score_desc=[]):
        cv2.imwrite(os.path.join(KaggleSubmission.BASEPATH, self.name, 'valid', idx + '.jpg'), image)
        if isinstance(idx, bytes):
            idx = idx.decode("utf-8")
        self.valid_scores[idx] = (loss, score, score_desc)

    def save_image(self, idx, image, loss=0.0):
        cv2.imwrite(os.path.join(KaggleSubmission.BASEPATH, self.name, idx + '.jpg'), image)
        self.test_scores[idx] = (loss, 0.0)

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
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'submission_%s.csv' % self.name)
        return filepath

    def get_confpath(self):
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'config.json')
        return filepath

    def get_train_htmlpath(self):
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'train', 'train.html')
        return filepath

    def get_valid_htmlpath(self):
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'valid', 'valid.html')
        return filepath

    def get_test_htmlpath(self):
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'test.html')
        return filepath

    def get_pklpath(self):
        filepath = os.path.join(KaggleSubmission.BASEPATH, self.name, 'submission.pkl')
        return filepath

    def save(self):
        sub = pd.DataFrame()
        sub['ImageId'] = self.test_ids
        sub['EncodedPixels'] = pd.Series(self.rles).apply(lambda x: ' '.join(str(y) for y in x))

        # save a submission file
        filepath = self.get_filepath()
        f = open(filepath, 'w')
        f.close()
        sub.to_csv(filepath, index=False)
        logger.info('%s saved at %s.' % (self.name, filepath))

        # save hyperparameters
        filepath = self.get_confpath()
        f = open(filepath, 'w')
        a = json.dumps(HyperParams.get().__dict__, indent=4)
        f.write(a)
        f.close()

        total_html = "<html><body>Average Score=$avg_score$<br/><br/><table>" \
                     "  <tr>" \
                     "      <th>ID</th><th>Image</th>" \
                     "  </tr>" \
                     "  $rows$" \
                     "</table></body></html>"
        row_html = "<tr>" \
                   "    <td><b>{idx}</b><br/>{iou}<br/>{iou2}</td><td><img src=\"./{idx}.jpg\"</td>" \
                   "</tr>"
        # save training results
        rows = []
        metrics = []
        for idx, (loss, metric, metric_desc) in self.train_scores.items():
            row = row_html.format(idx=idx, iou=format(metric, '.3f'), iou2='<br/>'.join(metric_desc))
            rows.append(row)
            metrics.append(metric)
        html = total_html.replace('$rows$', ''.join(rows)).replace('$avg_score$', str(np.mean(metrics)))

        filepath = self.get_train_htmlpath()
        f = open(filepath, 'w')
        f.write(html)
        f.close()

        # save validation results
        rows = []
        metrics = []
        for idx, (loss, metric, metric_desc) in self.valid_scores.items():
            row = row_html.format(idx=idx, iou=format(metric, '.3f'), iou2='<br/>'.join(metric_desc))
            rows.append(row)
            metrics.append(metric)
        html = total_html.replace('$rows$', ''.join(rows)).replace('$avg_score$', str(np.mean(metrics)))

        filepath = self.get_valid_htmlpath()
        f = open(filepath, 'w')
        f.write(html)
        f.close()

        # save test results
        total_html = "<html><body><table>" \
                     "  <tr>" \
                     "      <th>IDX</th><th>ID</th><th>Image</th>" \
                     "  </tr>" \
                     "  $rows$" \
                     "</table></body></html>"
        row_html = "<tr>" \
                   "    <td>{idx}</td><td><img src=\"./{idx}.jpg\"</td>" \
                   "</tr>"
        rows = []
        for idx, (loss, metric) in self.test_scores.items():
            row = row_html.format(idx=idx)
            rows.append(row)
        html = total_html.replace('$rows$', ''.join(rows))

        filepath = self.get_test_htmlpath()
        f = open(filepath, 'w')
        f.write(html)
        f.close()

        # save pkl
        f = open(self.get_pklpath(), 'wb')
        pickle.dump({
            'valid_instances': self.valid_instances,
            'test_instances': self.test_instances
        }, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def submit_result(self, submit_msg='KakaoAutoML'):
        """
        Submit result to kaggle and wait for getting the result.
        """
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
