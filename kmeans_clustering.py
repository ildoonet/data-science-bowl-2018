import os
import shutil

import numpy as np

from sklearn.cluster import KMeans
from tensorpack.dataflow.common import MapDataComponent
from tensorpack.dataflow import PrefetchData
from tqdm import tqdm

from data_augmentation import random_crop_224
from data_feeder import CellImageDataManagerTrainAll

master_dir_train = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/train_100_gray'

master_dir_origin_ext_train_kmeans = '/data/public/rw/datasets/dsb2018/train_kmeans_100_gray'
master_dir_origin_ext_valid_kmeans = '/data/public/rw/datasets/dsb2018/valid_kmeans_100_gray'

ratio = 0.8
n_clusters = 4


def cluster_features(features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1111)
    kmeans.fit(features)
    labels = kmeans.labels_
    return labels


def get_test_valid_split_labels():
    ds_train = CellImageDataManagerTrainAll()
    ds_train = MapDataComponent(ds_train, random_crop_224)
    ds_train = PrefetchData(ds_train, 1000, 12)
    ds_train_img = ds_train.get_data()

    features = []
    train_lists = []
    valid_lists = []
    for idx, dp in tqdm(enumerate(ds_train_img)):
        img = np.asarray(dp[0].image(is_gray=False))
        img = np.ndarray.flatten(img)
        features.append(img)

    features = np.stack(features, axis=0)

    labels = cluster_features(features, n_clusters=n_clusters)

    idx_labels = []
    for i in range(n_clusters):
        idx_labels.append(np.transpose(np.argwhere((labels == i))))

    # Split train and valid data set
    for n in range(n_clusters):
        train_lists.extend(idx_labels[n][:, :int(idx_labels[n].shape[1] * ratio)])
        valid_lists.extend(idx_labels[n][:, int(idx_labels[n].shape[1] * ratio):])

    return train_lists, valid_lists


def copy_clustered_image(train_lists, valid_lists):
    try:
        shutil.rmtree(os.path.join(master_dir_origin_ext_train_kmeans))
        shutil.rmtree(os.path.join(master_dir_origin_ext_valid_kmeans))
    except Exception as err:
        print('copy_clustered_image error:', err)

    # if file directory does not exist, create new one
    if not os.path.exists(master_dir_origin_ext_train_kmeans):
        os.mkdir(master_dir_origin_ext_train_kmeans)
    if not os.path.exists(master_dir_origin_ext_valid_kmeans):
        os.mkdir(master_dir_origin_ext_valid_kmeans)

    train_files_list = list(next(os.walk(master_dir_train))[1])

    for n in range(n_clusters):
        for col_train in range(train_lists[n].shape[0]):
            shutil.copytree(os.path.join(master_dir_train, train_files_list[train_lists[n][col_train]])
                            , os.path.join(master_dir_origin_ext_train_kmeans, train_files_list[train_lists[n][col_train]]))

        for col_valid in range(valid_lists[n].shape[0]):
            shutil.copytree(os.path.join(master_dir_train, train_files_list[valid_lists[n][col_valid]])
                            , os.path.join(master_dir_origin_ext_valid_kmeans, train_files_list[valid_lists[n][col_valid]]))

    print('=================== DONE ======================')


if __name__ == '__main__':
    train_lists, valid_lists = get_test_valid_split_labels()

    print("train_list: {}".format(train_lists))
    print("valid_list: {}".format(valid_lists))

    copy_clustered_image(train_lists, valid_lists)