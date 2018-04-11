import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from commons import chunker
from data_augmentation import data_to_segment_input, \
    data_to_image, random_flip_lr, random_flip_ud, random_scaling, random_affine, \
    random_color, data_to_normalize1, data_to_elastic_transform_wrapper, resize_shortedge_if_small, random_crop, \
    center_crop, random_color2, erosion_mask, resize_shortedge, mask_size_normalize, crop_mirror, pad_if_small, \
    mirror_pad, center_crop_if_tcga, random_add_thick_area
from data_feeder import CellImageDataManagerTrain, CellImageDataManagerValid, CellImageDataManagerTest
from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow.parallel import PrefetchData, MultiThreadPrefetchData

from hyperparams import HyperParams
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


class NetworkUnetValid(NetworkBasic):
    def __init__(self, batchsize):
        super().__init__(batchsize, unet_weight=True)

        self.img_size = 228
        self.num_block = HyperParams.get().unet_step_size
        self.inp_size = get_net_input_size(self.img_size, self.num_block)
        assert (self.inp_size - self.img_size) % 2 == 0
        self.pad_size = (self.inp_size - self.img_size) // 2

        self.batchsize = batchsize
        self.input_batch = tf.placeholder(tf.float32, shape=(None, self.img_size + self.pad_size * 2, self.img_size + self.pad_size * 2, 3), name='image')
        self.mask_batch = tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, 1), name='mask')
        self.weight_batch = tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, 1), name='weight')
        self.unused = None
        self.logit = None
        self.output = None

    @staticmethod
    def double_conv(net, nb_filter, scope):
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_1' % scope)
        net = slim.dropout(net)
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_2' % scope)
        return net

    def build(self):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L429
        weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=HyperParams.get().net_init_stddev)
        batch_norm_params = {
            'is_training': self.is_training,
            'center': True,
            'scale': True,
            'decay': HyperParams.get().net_bn_decay,
            'epsilon': HyperParams.get().net_bn_epsilon,
            'fused': True,
            'zero_debias_moving_mean': True
        }

        dropout_params = {
            'keep_prob': HyperParams.get().net_dropout_keep,
            'is_training': self.is_training,
        }

        conv_args = {
            'padding': 'VALID',
            'weights_initializer': weight_init,
            'normalizer_fn': slim.batch_norm,
            'normalizer_params': batch_norm_params,
            'activation_fn': tf.nn.elu,
            'weights_regularizer': slim.l2_regularizer(0.0001)
        }

        net = self.input_batch

        features = []
        with slim.arg_scope([slim.convolution, slim.conv2d_transpose], **conv_args):
            with slim.arg_scope([slim.dropout], **dropout_params):
                base_feature_size = HyperParams.get().unet_base_feature
                max_feature_size = base_feature_size * (2 ** self.num_block)

                # down sampling steps
                for i in range(self.num_block):
                    net = NetworkUnetValid.double_conv(net, int(base_feature_size*(2**i)), scope='down_conv_%d' % (i + 1))
                    features.append(net)
                    net = slim.max_pool2d(net, [2, 2], 2, padding='VALID', scope='pool%d' % (i + 1))

                # middle
                net = NetworkUnetValid.double_conv(net, max_feature_size, scope='middle_conv_1')

                # upsampling steps
                for i in range(self.num_block):
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
        self.loss_opt = self.loss
        return net

    def get_input_flow(self):
        ds_train = CellImageDataManagerTrain()
        # ds_train = MapDataComponent(ds_train, random_affine)  # TODO : no improvement?
        ds_train = MapDataComponent(ds_train, random_color)
        # ds_train = MapDataComponent(ds_train, random_scaling)
        ds_train = MapDataComponent(ds_train, mask_size_normalize)  # Resize by instance size - normalization
        ds_train = MapDataComponent(ds_train, lambda x: resize_shortedge_if_small(x, self.img_size))
        # ds_train = MapDataComponent(ds_train, lambda x: pad_if_small(x, self.img_size)) # preseve cell's size
        ds_train = MapDataComponent(ds_train, lambda x: random_crop(x, self.img_size, self.img_size, padding=self.pad_size))
        ds_train = MapDataComponent(ds_train, random_add_thick_area)
        ds_train = MapDataComponent(ds_train, random_flip_lr)
        ds_train = MapDataComponent(ds_train, random_flip_ud)
        # ds_train = MapDataComponent(ds_train, data_to_elastic_transform_wrapper)
        if self.unet_weight:
            ds_train = MapDataComponent(ds_train, erosion_mask)
        ds_train = MapData(ds_train, lambda x: data_to_segment_input(x, not self.is_color, self.unet_weight))
        ds_train = PrefetchData(ds_train, 256, 24)
        ds_train = BatchData(ds_train, self.batchsize)
        ds_train = MapDataComponent(ds_train, data_to_normalize1)

        ds_valid = CellImageDataManagerValid()
        ds_valid = MapDataComponent(ds_valid, lambda x: resize_shortedge_if_small(x, self.img_size))
        ds_valid = MapDataComponent(ds_valid, lambda x: random_crop(x, self.img_size, self.img_size, padding=self.pad_size))
        if self.unet_weight:
            ds_valid = MapDataComponent(ds_valid, erosion_mask)
        ds_valid = MapData(ds_valid, lambda x: data_to_segment_input(x, not self.is_color, self.unet_weight))
        ds_valid = PrefetchData(ds_valid, 32, 8)
        ds_valid = BatchData(ds_valid, self.batchsize, remainder=True)
        ds_valid = MapDataComponent(ds_valid, data_to_normalize1)

        ds_valid2 = CellImageDataManagerValid()
        ds_valid2 = MapDataComponent(ds_valid2, lambda x: resize_shortedge_if_small(x, self.img_size))
        ds_valid2 = MapDataComponent(ds_valid2, lambda x: center_crop_if_tcga(x, self.img_size, self.img_size))
        # ds_valid2 = MapDataComponent(ds_valid2, lambda x: resize_shortedge(x, self.img_size))
        ds_valid2 = MapData(ds_valid2, lambda x: data_to_segment_input(x, not self.is_color))
        ds_valid2 = MapDataComponent(ds_valid2, data_to_normalize1)

        ds_test = CellImageDataManagerTest()
        ds_test = MapDataComponent(ds_test, lambda x: resize_shortedge_if_small(x, self.img_size))
        # ds_test = MapDataComponent(ds_test, lambda x: resize_shortedge(x, self.img_size))
        ds_test = MapData(ds_test, lambda x: data_to_image(x, not self.is_color))
        ds_test = MapDataComponent(ds_test, data_to_normalize1)

        return ds_train, ds_valid, ds_valid2, ds_test

    def inference(self, tf_sess, image, cutoff_instance_max=0.0, cutoff_instance_avg=0.0):
        # TODO : Mirror Padding?
        cascades, windows = Network.sliding_window(image, self.img_size, 0.5)

        # by batch
        outputs = []
        padding = self.pad_size
        mirror_padded = mirror_pad(image, padding)
        for ws in chunker(windows, 64):
            b = []
            for w in ws:
                b.append(mirror_padded[w.y:w.y+w.h+padding*2, w.x:w.x+w.w+padding*2])
            output = tf_sess.run(self.get_output(), feed_dict={
                self.input_batch: b,
                self.is_training: False
            })
            outputs.append(output)
        outputs = np.concatenate(outputs, axis=0)

        # merge multiple results
        merged_output = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
        for window, output in zip(windows, outputs):
            merged_output[window.indices()] = np.maximum(output, merged_output[window.indices()])
        merged_output = merged_output.reshape((image.shape[0], image.shape[1]))

        # sementation to instance-aware segmentations.
        instances, scores = Network.parse_merged_output(
            merged_output,
            cutoff=0.5,
            cutoff_instance_max=cutoff_instance_max,
            cutoff_instance_avg=cutoff_instance_avg
        )

        # instances = Network.watershed_merged_output(instances)

        return {
            'instances': instances,
            'scores': scores
        }

    def preprocess(self, x):
        x = resize_shortedge_if_small(x, self.img_size)   # self.img_size
        x = data_to_normalize1(x)
        return x
