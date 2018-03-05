import tensorflow as tf
import numpy as np
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from data_augmentation import random_crop_224, data_to_segment_input, data_to_normalize01, center_crop_224, \
    data_to_image, resize_shortedge_if_small_224, random_flip_lr, random_flip_ud, random_scaling, \
    resize_shortedge_if_small, random_crop, center_crop
from data_feeder import CellImageDataManagerTrain, CellImageDataManagerValid, CellImageDataManagerTest
from network import Network

from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow.parallel import PrefetchData
from tensorflow.contrib import slim

SIZE = 128


class NetworkBasicNoPool(Network):
    def __init__(self, batchsize):
        super().__init__()

        self.batchsize = batchsize
        self.input_batch = tf.placeholder(tf.float32, shape=(None, SIZE, SIZE, 1), name='image')
        self.mask_batch = tf.placeholder(tf.float32, shape=(None, SIZE, SIZE, 1), name='mask')
        self.unused = None
        self.logit = None
        self.output = None
        self.loss = None

    def get_placeholders(self):
        return self.input_batch, self.mask_batch, self.unused

    def build(self):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L429
        batch_norm_params = {
            'is_training': self.is_training,
            'center': True,
            'scale': False,
            'decay': 0.9,
            'epsilon': 0.001,
            'fused': True,
            'zero_debias_moving_mean': True
        }
        activation = tf.nn.relu6

        net = self.input_batch
        features = []
        for i in range(3):
            net = slim.convolution(net, int(32 * (2 ** i)), [3, 3], 1, padding='SAME',
                                   scope='conv%d' % (i + 1),
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params,
                                   activation_fn=activation)
            features.append(net)

        net = slim.convolution(net, int(256), [3, 3], 1, padding='SAME',
                               scope='conv4',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_params,
                               activation_fn=activation)
        features.append(net)

        # upsample
        net = tf.concat(axis=3, values=features, name='concat_features')

        net = slim.convolution(net, int(256), [1, 1], 1, padding='SAME',
                               scope='bottleneck',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_params,
                               activation_fn=activation)

        net = slim.convolution(net, 1, [3, 3], 1, padding='SAME',  # TODO : Tuning 3x3?
                               scope='conv_last',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               normalizer_fn=None,
                               activation_fn=None)

        self.logit = net
        self.output = tf.nn.sigmoid(net, 'visualization')
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask_batch, logits=self.logit))
        return net

    def get_input_flow(self):
        ds_train = CellImageDataManagerTrain()
        # TODO : Augmentation?
        ds_train = MapDataComponent(ds_train, random_scaling)
        ds_train = MapDataComponent(ds_train, lambda x: resize_shortedge_if_small(x, SIZE))
        ds_train = MapDataComponent(ds_train, lambda x: random_crop(x, SIZE, SIZE))
        ds_train = MapDataComponent(ds_train, random_flip_lr)
        ds_train = MapDataComponent(ds_train, random_flip_ud)
        ds_train = PrefetchData(ds_train, 1000, 12)
        ds_train = MapData(ds_train, data_to_segment_input)
        ds_train = BatchData(ds_train, self.batchsize)
        ds_train = MapDataComponent(ds_train, data_to_normalize01)
        ds_train = PrefetchData(ds_train, 10, 2)

        ds_valid = CellImageDataManagerValid()
        ds_valid = MapDataComponent(ds_valid, lambda x: random_crop(x, SIZE, SIZE))
        ds_valid = MapData(ds_valid, data_to_segment_input)
        ds_valid = BatchData(ds_valid, self.batchsize, remainder=True)
        ds_valid = MapDataComponent(ds_valid, data_to_normalize01)
        ds_valid = PrefetchData(ds_valid, 20, 8)

        ds_valid2 = CellImageDataManagerValid()
        ds_valid2 = MapDataComponent(ds_valid2, lambda x: resize_shortedge_if_small(x, SIZE))
        ds_valid2 = MapData(ds_valid2, data_to_segment_input)
        ds_valid2 = MapDataComponent(ds_valid2, data_to_normalize01)

        ds_test = CellImageDataManagerTest()
        ds_test = MapDataComponent(ds_test, lambda x: resize_shortedge_if_small(x, SIZE))
        ds_test = MapData(ds_test, data_to_image)
        ds_test = MapDataComponent(ds_test, data_to_normalize01)

        return ds_train, ds_valid, ds_valid2, ds_test

    def get_logit(self):
        if self.logit is None:
            raise Exception('NetworkBasic.get_logit() should be called after build() is called.')
        return self.logit

    def get_output(self):
        if self.logit is None:
            raise Exception('NetworkBasic.get_output() should be called after build() is called.')
        return self.output

    def get_loss(self):
        return self.loss

    def inference(self, tf_sess, image):
        cascades, windows = Network.sliding_window(image, SIZE, 0.2)

        outputs = tf_sess.run(self.get_output(), feed_dict={
            self.input_batch: cascades,
            self.is_training: False
        })

        # merge multiple results
        merged_output = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
        for window, output in zip(windows, outputs):
            merged_output[window.indices()] = np.maximum(output, merged_output[window.indices()])
        merged_output = merged_output.reshape((image.shape[0], image.shape[1]))

        # sementation to instance-aware segmentations.
        instances = Network.parse_merged_output(merged_output, cutoff=0.5, use_separator=True)

        return instances