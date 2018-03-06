import random

import numpy as np
import os
import cv2
from scipy import ndimage
from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow import PrefetchData

from data_augmentation import random_crop_224, data_to_segment_input, data_to_normalize01

master_dir_train = '/data/public/rw/datasets/dsb2018/train'
master_dir_test = '/data/public/rw/datasets/dsb2018/test'


class CellImageData:
    def __init__(self, target_id, path, erosion_mask=False):
        self.target_id = target_id

        # read
        target_dir = os.path.join(path, target_id)

        img_path = os.path.join(target_dir, 'images', target_id + '.png')

        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.img_h, self.img_w = self.img.shape[:2]
        assert self.img_h > 0 and self.img_w > 0
        self.masks = []
        mask_dir = os.path.join(target_dir, 'masks')

        if not os.path.exists(mask_dir):
            return

        for mask_file in next(os.walk(mask_dir))[2]:
            mask_path = os.path.join(target_dir, 'masks', mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask >> 7    # faster than mask // 129
            if erosion_mask:
                mask = ndimage.morphology.binary_erosion((mask > 0), border_value=1).astype(np.uint8)
            self.masks.append(mask)

    def single_mask(self, ch1=True):
        """
        :return: (h, w, 1) numpy
        """
        multi_masks = self.multi_masks()
        multi_masks = np.sum(multi_masks, axis=2)
        if ch1:
            multi_masks = multi_masks[..., np.newaxis]
        return multi_masks

    def multi_masks(self):
        """
        :return: (h, w, m) numpy
        """
        return np.stack(self.masks, axis=0).transpose([1, 2, 0])

    def multi_masks_batch(self):
        img_h, img_w = self.img.shape[:2]
        m = np.zeros(shape=(img_h, img_w, 1), dtype=np.uint8)
        for idx, mask in enumerate(self.masks):
            m = m + mask[..., np.newaxis] * (idx + 1)
        return m

    @staticmethod
    def batch_to_multi_masks(multi_masks_batch, transpose=True):
        a = np.array([multi_masks_batch == (idx + 1) for idx in range(np.max(multi_masks_batch))], dtype=np.uint8)

        if transpose:
            return a[..., 0].transpose([1, 2, 0])
        else:
            return a[..., 0]

    def image(self, is_gray=True):
        """
        :return: (h, w, 3) or (h, w, 1) numpy
        """
        img = self.img
        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[..., np.newaxis]
        return img

    def unet_weights(self):
        # ref : https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34/notebook
        w0 = 10
        sigma = 5

        merged_mask = self.single_mask(ch1=False)

        distances = np.array([ndimage.distance_transform_edt(m == 0) for m in self.masks])
        shortest_dist = np.sort(distances, axis=0)

        # distance to the border of the nearest cell
        d1 = shortest_dist[0]
        # distance to the border of the second nearest cell
        d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

        weight = w0 * np.exp(-(d1 + d2) ** 2 / (2 * sigma ** 2)).astype(np.float32)
        weight = 1 + (merged_mask == 0) * weight
        return weight[..., np.newaxis]


class CellImageDataManager(RNGDataFlow):
    def __init__(self, path, idx_list, is_shuffle=False, erosion_mask=False):
        self.path = path
        self.idx_list = idx_list
        self.is_shuffle = is_shuffle
        self.erosion_mask = erosion_mask

    def size(self):
        return len(self.idx_list)

    def get_data(self):
        if self.is_shuffle:
            random.shuffle(self.idx_list)

        for idx in self.idx_list:
            yield [CellImageData(idx, self.path, erosion_mask=self.erosion_mask)]


class CellImageDataManagerTrain(CellImageDataManager):
    def __init__(self, erosion_mask=False):
        super().__init__(
            master_dir_train,
            list(next(os.walk(master_dir_train))[1])[:576],
            True,
            erosion_mask
        )
        # TODO : train/valid set k folds implementation


class CellImageDataManagerValid(CellImageDataManager):
    def __init__(self, erosion_mask=False):
        super().__init__(
            master_dir_train,
            list(next(os.walk(master_dir_train))[1])[576:],
            False,
            erosion_mask
        )


class CellImageDataManagerTest(CellImageDataManager):
    def __init__(self):
        super().__init__(
            master_dir_test,
            list(next(os.walk(master_dir_test))[1]),
            False
        )


def get_default_dataflow():
    ds = CellImageDataManagerTrain()
    ds = MapDataComponent(ds, random_crop_224)
    ds = PrefetchData(ds, 1000, 12)

    return ds


def get_default_dataflow_batch(batchsize=32):
    ds = get_default_dataflow()
    ds = MapData(ds, data_to_segment_input)
    ds = BatchData(ds, batchsize)
    ds = MapDataComponent(ds, data_to_normalize01)
    ds = PrefetchData(ds, 10, 2)

    return ds
