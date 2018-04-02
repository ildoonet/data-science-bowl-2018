import tensorflow as tf
import numpy as np
from tensorflow.python.ops.losses.losses_impl import Reduction

from data_augmentation import data_to_segment_input, \
    data_to_image, random_flip_lr, random_flip_ud, random_scaling, random_affine, \
    random_color, data_to_normalize1, data_to_elastic_transform_wrapper, random_color2, erosion_mask, random_crop, \
    resize_shortedge_if_small, center_crop
from data_feeder import CellImageDataManagerTrain, CellImageDataManagerValid, CellImageDataManagerTest
from network import Network

from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow.parallel import PrefetchData
from tensorflow.contrib import slim


class NetworkBasic(Network):
    def __init__(self, batchsize, unet_weight):
        super().__init__()

        self.batchsize = batchsize
        self.is_color = True
        if self.is_color:
            self.input_batch = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image')
        else:
            self.input_batch = tf.placeholder(tf.float32, shape=(None, 224, 224, 1), name='image')
        self.mask_batch = tf.placeholder(tf.float32, shape=(None, 224, 224, 1), name='mask')
        self.weight_batch = tf.placeholder(tf.float32, shape=(None, 224, 224, 1), name='weight')
        self.unused = None
        self.logit = None
        self.output = None
        self.loss = None
        self.unet_weight = unet_weight

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

        for i in range(1):
            net = slim.convolution(net, int(128), [3, 3], 1, padding='SAME',
                                   scope='preconv%d' % i,
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params,
                                   activation_fn=activation)
            features.append(net)
            net = slim.convolution(net, int(256), [3, 3], 1, padding='SAME',
                                   scope='preconv%d-2' % i,
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params,
                                   activation_fn=activation)
            features.append(net)
            net = slim.convolution(net, int(32), [1, 1], 1, padding='SAME',
                                   scope='preconv%d-b' % i,
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params,
                                   activation_fn=activation)

        conv_pool_size = 4
        for i in range(conv_pool_size):
            net = slim.convolution(net, int(64 * (2 ** i)), [3, 3], 1, padding='SAME',
                                   scope='conv%d' % (i + 1),
                                   weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                   normalizer_fn=slim.batch_norm,
                                   normalizer_params=batch_norm_params,
                                   activation_fn=activation)
            net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool%d' % (i + 1))
            features.append(net)

        net = slim.convolution(net, int(256), [3, 3], 1, padding='SAME',
                               scope='conv%d' % (conv_pool_size + 1),
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_params,
                               activation_fn=activation)
        features.append(net)

        # upsample
        features_up = [tf.image.resize_bilinear(f, (112, 112)) for f in features]
        net = tf.concat(axis=3, values=features_up, name='concat_features')

        net = slim.convolution(net, int(256), [1, 1], 1, padding='SAME',
                               scope='bottleneck',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               normalizer_fn=slim.batch_norm,
                               normalizer_params=batch_norm_params,
                               activation_fn=activation)

        net = slim.convolution(net, 1, [5, 5], 1, padding='SAME',
                               scope='conv_last',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                               normalizer_fn=None,
                               activation_fn=None)
        net = tf.image.resize_bilinear(net, (224, 224))

        self.logit = net
        self.output = tf.nn.sigmoid(net, 'visualization')
        if self.unet_weight:
            w = self.weight_batch
        else:
            w = 1.0

        self.loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=self.mask_batch,
            logits=self.logit,
            weights=w,
            reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
        )
        return net

    def get_input_flow(self):
        ds_train = CellImageDataManagerTrain()
        # Augmentation :
        ds_train = MapDataComponent(ds_train, random_affine)
        ds_train = MapDataComponent(ds_train, random_color)
        # ds_train = MapDataComponent(ds_train, random_color2)  # not good
        ds_train = MapDataComponent(ds_train, random_scaling)
        ds_train = MapDataComponent(ds_train, lambda x: resize_shortedge_if_small(x, 224))
        ds_train = MapDataComponent(ds_train, lambda x: random_crop(x, 224, 224))
        ds_train = MapDataComponent(ds_train, random_flip_lr)
        # ds_train = MapDataComponent(ds_train, data_to_elastic_transform_wrapper)
        ds_train = MapDataComponent(ds_train, random_flip_ud)
        if self.unet_weight:
            ds_train = MapDataComponent(ds_train, erosion_mask)
        ds_train = PrefetchData(ds_train, 1000, 24)
        ds_train = MapData(ds_train, lambda x: data_to_segment_input(x, not self.is_color, self.unet_weight))
        ds_train = BatchData(ds_train, self.batchsize)
        ds_train = MapDataComponent(ds_train, data_to_normalize1)
        ds_train = PrefetchData(ds_train, 10, 2)

        ds_valid = CellImageDataManagerValid()
        ds_valid = MapDataComponent(ds_valid, lambda x: center_crop(x, 224, 224))
        if self.unet_weight:
            ds_valid = MapDataComponent(ds_valid, erosion_mask)
        ds_valid = MapData(ds_valid, lambda x: data_to_segment_input(x, not self.is_color, self.unet_weight))
        ds_valid = BatchData(ds_valid, self.batchsize, remainder=True)
        ds_valid = MapDataComponent(ds_valid, data_to_normalize1)
        ds_valid = PrefetchData(ds_valid, 20, 24)

        ds_valid2 = CellImageDataManagerValid()
        ds_valid2 = MapDataComponent(ds_valid2, lambda x: resize_shortedge_if_small(x, 224))
        ds_valid2 = MapData(ds_valid2, lambda x: data_to_segment_input(x, not self.is_color))
        ds_valid2 = MapDataComponent(ds_valid2, data_to_normalize1)

        ds_test = CellImageDataManagerTest()
        ds_test = MapDataComponent(ds_test, lambda x: resize_shortedge_if_small(x, 224))
        ds_test = MapData(ds_test, lambda x: data_to_image(x, not self.is_color))
        ds_test = MapDataComponent(ds_test, data_to_normalize1)

        return ds_train, ds_valid, ds_valid2, ds_test

    def get_feeddict(self, dp, is_training):
        feed_dict = {
            self.input_batch: dp[0],
            self.mask_batch: dp[1],
            self.is_training: is_training
        }
        if self.unet_weight:
            feed_dict[self.weight_batch] = dp[3]
        return feed_dict

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

    def preprocess(self, x):
        x = resize_shortedge_if_small_224(x)
        x = data_to_normalize1(x)
        return x

    def inference(self, tf_sess, image):
        cascades, windows = Network.sliding_window(image, 224, 0.5)

        outputs = tf_sess.run(self.get_output(), feed_dict={
            self.input_batch: cascades,
            self.is_training: False
        })

        # merge multiple results
        merged_output = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
        # merged_counts = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.int32)
        for window, output in zip(windows, outputs):
            # suppress with maximum value
            merged_output[window.indices()] = np.maximum(output, merged_output[window.indices()])
            # suppress with average
            # merged_output[window.indices()] = output + merged_output[window.indices()]
            # merged_counts[window.indices()] = merged_counts[window.indices()] + 1

        # notzeroidx = merged_counts > 0
        # merged_output[notzeroidx] = merged_output[notzeroidx] / merged_counts[notzeroidx]

        merged_output = merged_output.reshape((image.shape[0], image.shape[1]))

        # sementation to instance-aware segmentations.
        instances = Network.parse_merged_output(
            merged_output, cutoff=0.5, use_separator=False
        )

        return instances
