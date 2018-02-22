import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from data_augmentation import random_crop_224, data_to_segment_input, data_to_normalize01
from data_feeder import CellImageDataManagerTrain, CellImageDataManagerValid, CellImageDataManagerTest
from network import Network

from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow import PrefetchData

slim = tf.contrib.slim


class NetworkBasic(Network):
    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.input_batch = tf.placeholder(tf.float32, shape=(batchsize, 224, 224, 1), name='image')
        self.mask_batch = tf.placeholder(tf.float32, shape=(batchsize, 224, 224, 1), name='mask')
        self.unused = None

    def get_placeholders(self):
        return self.input_batch, self.mask_batch, self.unused

    def build(self):
        net = self.input_batch
        for i in range(3):
            net = slim.convolution(net, 64*(2**i), [3, 3], 1, padding='SAME',
                                   scope='conv%d' % (i + 1),
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
                                   normalizer_fn=slim.batch_norm,
                                   activation_fn=tf.nn.relu6)
            net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool%d' % (i + 1))

        net = slim.convolution(net, 512, [3, 3], 1, padding='SAME',
                               scope='conv4',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
                               normalizer_fn=slim.batch_norm,
                               activation_fn=tf.nn.relu6)
        net = slim.convolution(net, 1, [3, 3], 1, padding='SAME',
                               scope='conv5',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
                               normalizer_fn=None,
                               activation_fn=None)

        # upsample
        net = tf.image.resize_bicubic(net, (224, 224))
        return net

    def get_input_flow(self):
        ds_train = CellImageDataManagerTrain()
        # TODO : Augmentation?
        ds_train = MapDataComponent(ds_train, random_crop_224)
        ds_train = PrefetchData(ds_train, 1000, 12)
        ds_train = MapData(ds_train, data_to_segment_input)
        ds_train = BatchData(ds_train, self.batchsize)
        ds_train = MapDataComponent(ds_train, data_to_normalize01)
        ds_train = PrefetchData(ds_train, 10, 2)

        ds_valid = CellImageDataManagerValid()
        ds_valid = MapDataComponent(ds_valid, random_crop_224)  # TODO : Center Crop?
        ds_valid = PrefetchData(ds_valid, 1000, 12)
        ds_valid = MapData(ds_valid, data_to_segment_input)
        ds_valid = BatchData(ds_valid, self.batchsize)
        ds_valid = MapDataComponent(ds_valid, data_to_normalize01)
        ds_valid = PrefetchData(ds_valid, 10, 2)

        ds_test = CellImageDataManagerTest()
        ds_test = MapDataComponent(ds_test, random_crop_224)
        ds_test = PrefetchData(ds_test, 1000, 12)
        ds_test = MapData(ds_test, data_to_segment_input)
        ds_test = BatchData(ds_test, self.batchsize)
        ds_test = MapDataComponent(ds_test, data_to_normalize01)
        ds_test = PrefetchData(ds_test, 10, 2)

        return ds_train, ds_valid, ds_test

    def postprocess(self, output):
        pass
