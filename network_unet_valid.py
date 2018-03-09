import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from data_augmentation import data_to_segment_input, \
    data_to_image, random_flip_lr, random_flip_ud, random_scaling, random_affine, \
    random_color, data_to_normalize1, data_to_elastic_transform_wrapper, resize_shortedge_if_small, random_crop, \
    center_crop, random_color2, erosion_mask
from data_feeder import CellImageDataManagerTrain, CellImageDataManagerValid, CellImageDataManagerTest
from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow.parallel import PrefetchData

from network import Network
from network_basic import NetworkBasic


def get_net_input_size(image_size, num_block):
    network_input_size = image_size
    for _ in range(num_block):
        assert network_input_size % 2 == 0, network_input_size
        network_input_size = (network_input_size + 4) // 2
    network_input_size += 4
    for _ in range(num_block):
        network_input_size = network_input_size * 2 + 4
    return network_input_size


INPUT_IMAGE_SIZE = 228
NUM_BLOCK = 4
NETWORK_INPUT_SIZE = get_net_input_size(INPUT_IMAGE_SIZE, NUM_BLOCK)

assert (NETWORK_INPUT_SIZE - INPUT_IMAGE_SIZE) % 2 == 0
INPUT_MIRROR_PAD = (NETWORK_INPUT_SIZE - INPUT_IMAGE_SIZE) // 2


class NetworkUnetValid(NetworkBasic):
    def __init__(self, batchsize, unet_weight):
        super().__init__(batchsize, unet_weight)

        self.batchsize = batchsize
        self.is_color = True
        if self.is_color:
            self.input_batch = tf.placeholder(tf.float32, shape=(None, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3), name='image')
        else:
            self.input_batch = tf.placeholder(tf.float32, shape=(None, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1), name='image')
        self.mask_batch = tf.placeholder(tf.float32, shape=(None, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1), name='mask')
        self.weight_batch = tf.placeholder(tf.float32, shape=(None, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1), name='weight')
        self.unused = None
        self.logit = None
        self.output = None
        self.loss = None
        self.unet_weight = unet_weight

    @staticmethod
    def double_conv(net, nb_filter, scope):
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_1' % scope)
        net = slim.dropout(net)
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_2' % scope)
        return net

    def build(self):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L429
        weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        batch_norm_params = {
            'is_training': self.is_training,
            'center': True,
            'scale': True,
            'decay': 0.9,
            'epsilon': 0.001,
            'fused': True,
            'zero_debias_moving_mean': True
        }

        dropout_params = {
            'keep_prob': 0.9,
            'is_training': self.is_training,
        }

        conv_args = {
            'padding': 'VALID',
            'weights_initializer': weight_init,
            'normalizer_fn': slim.batch_norm,
            'normalizer_params': batch_norm_params,
            'activation_fn': tf.nn.elu
        }

        net = self.input_batch

        # mirror padding
        paddings = tf.constant([
            [0, 0],
            [INPUT_MIRROR_PAD, INPUT_MIRROR_PAD],
            [INPUT_MIRROR_PAD, INPUT_MIRROR_PAD],
            [0, 0],
        ])
        net = tf.pad(net, paddings, "REFLECT", name='img2inp')
        assert net.shape[1] == NETWORK_INPUT_SIZE, net.shape[2] == NETWORK_INPUT_SIZE

        features = []
        with slim.arg_scope([slim.convolution, slim.conv2d_transpose], **conv_args):
            with slim.arg_scope([slim.dropout], **dropout_params):
                base_feature_size = 32
                max_feature_size = base_feature_size * (2 ** NUM_BLOCK)

                # down sampling steps
                for i in range(NUM_BLOCK):
                    net = NetworkUnetValid.double_conv(net, int(base_feature_size*(2**i)), scope='down_conv_%d' % (i + 1))
                    features.append(net)
                    net = slim.max_pool2d(net, [2, 2], 2, padding='VALID', scope='pool%d' % (i + 1))

                # middle
                net = NetworkUnetValid.double_conv(net, max_feature_size, scope='middle_conv_1')

                # upsampling steps
                for i in range(NUM_BLOCK):
                    # up-conv
                    net = slim.conv2d_transpose(net, int(max_feature_size/(2**(i+1))), [2, 2], 2, scope='up_trans_conv_%d' % (i + 1))

                    # get lower layer's feature
                    down_feat = features.pop()
                    assert net.shape[3] == down_feat.shape[3], '%d, %d, %d' % (i, net.shape[3], down_feat.shape[3])

                    y, x = [int(down_feat.shape[idx] - net.shape[idx]) // 2 for idx in [1, 2]]
                    h, w = map(int, net.shape[1:3])
                    down_feat = tf.slice(down_feat, [0, y, x, 0], [-1, h, w, -1])

                    net = tf.concat([down_feat, net], axis=-1)
                    net = NetworkUnetValid.double_conv(net, int(max_feature_size/(2**(i+1))), scope='up_conv_%d' % (i + 1))

        # original paper : one 1x1 conv
        net = slim.convolution(net, 1, [1, 1], 1, scope='final_conv',
                               activation_fn=None,
                               padding='SAME',
                               weights_initializer=weight_init)

        self.logit = net
        self.output = tf.nn.sigmoid(net, 'visualization')
        if self.unet_weight:
            w = self.weight_batch
        else:
            w = 1.0

        self.loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=self.mask_batch,
            logits=self.logit,
            weights=w
        )
        return net

    def get_input_flow(self):
        ds_train = CellImageDataManagerTrain()
        # Augmentation :
        ds_train = MapDataComponent(ds_train, random_affine)
        ds_train = MapDataComponent(ds_train, random_color)
        ds_train = MapDataComponent(ds_train, random_scaling)
        ds_train = MapDataComponent(ds_train, lambda x: resize_shortedge_if_small(x, INPUT_IMAGE_SIZE))
        ds_train = MapDataComponent(ds_train, lambda x: random_crop(x, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        ds_train = MapDataComponent(ds_train, random_flip_lr)
        ds_train = MapDataComponent(ds_train, random_flip_ud)
        # ds_train = MapDataComponent(ds_train, data_to_elastic_transform_wrapper)
        if self.unet_weight:
            ds_train = MapDataComponent(ds_train, erosion_mask)
        ds_train = PrefetchData(ds_train, 1000, 24)
        ds_train = MapData(ds_train, lambda x: data_to_segment_input(x, not self.is_color, self.unet_weight))
        ds_train = BatchData(ds_train, self.batchsize)
        ds_train = MapDataComponent(ds_train, data_to_normalize1)
        ds_train = PrefetchData(ds_train, 10, 2)

        ds_valid = CellImageDataManagerValid()
        ds_valid = MapDataComponent(ds_valid, lambda x: random_crop(x, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        if self.unet_weight:
            ds_valid = MapDataComponent(ds_valid, erosion_mask)
        ds_valid = MapData(ds_valid, lambda x: data_to_segment_input(x, not self.is_color, self.unet_weight))
        ds_valid = BatchData(ds_valid, self.batchsize, remainder=True)
        ds_valid = MapDataComponent(ds_valid, data_to_normalize1)
        # ds_valid = PrefetchData(ds_valid, 20, 24)

        ds_valid2 = CellImageDataManagerValid()
        ds_valid2 = MapDataComponent(ds_valid2, lambda x: resize_shortedge_if_small(x, INPUT_IMAGE_SIZE))
        ds_valid2 = MapData(ds_valid2, lambda x: data_to_segment_input(x, not self.is_color))
        ds_valid2 = MapDataComponent(ds_valid2, data_to_normalize1)

        ds_test = CellImageDataManagerTest()
        ds_test = MapDataComponent(ds_test, lambda x: resize_shortedge_if_small(x, INPUT_IMAGE_SIZE))
        ds_test = MapData(ds_test, lambda x: data_to_image(x, not self.is_color))
        ds_test = MapDataComponent(ds_test, data_to_normalize1)

        return ds_train, ds_valid, ds_valid2, ds_test

    def inference(self, tf_sess, image):
        cascades, windows = Network.sliding_window(image, INPUT_IMAGE_SIZE, 0.5)

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
        instances = Network.parse_merged_output(
            merged_output, cutoff=0.5, use_separator=False,
            use_dilation=self.unet_weight,
            fill_holes=True
        )

        return instances
