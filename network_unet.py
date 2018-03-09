import tensorflow as tf
from tensorflow.contrib import slim
from network_basic import NetworkBasic


class NetworkUnet(NetworkBasic):
    def __init__(self,
                 batchsize,
                 unet_weight,
                 batch_norm_decay=0.9,
                 batch_norm_epsilon=0.001,
                 keep_prob=0.9,
                 stddev=0.01,
                 ):
        super(NetworkUnet, self).__init__(batchsize=batchsize, unet_weight=unet_weight)
        self.batch_norm_decay = batch_norm_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.keep_prob = keep_prob
        self.stddev = stddev

    @staticmethod
    def double_conv(net, nb_filter, scope, keep_prob):
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_1' % scope)
        net = slim.dropout(net, keep_prob=keep_prob)
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_2' % scope)
        return net

    def build(self):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L429
        batch_norm_params = {
            'is_training': self.is_training,
            'center': True,
            'scale': True,
            'decay': self.batch_norm_decay,
            'epsilon': self.batch_norm_epsilon,
            'fused': True,
            'zero_debias_moving_mean': True
        }

        dropout_params = {
            'keep_prob': self.keep_prob,
            'is_training': self.is_training,
        }

        conv_args = {
            'padding': 'SAME',
            'weights_initializer': tf.truncated_normal_initializer(mean=0.0, stddev=self.stddev),
            'normalizer_fn': slim.batch_norm,
            'normalizer_params': batch_norm_params,
            'activation_fn': tf.nn.elu
        }

        net = self.input_batch
        features = []

        with slim.arg_scope([slim.convolution, slim.conv2d_transpose], **conv_args):
            with slim.arg_scope([slim.dropout], **dropout_params):
                step_size = 4
                base_feature_size = 32
                max_feature_size = base_feature_size * (2 ** step_size)

                # down sampling steps
                for i in range(step_size):
                    net = NetworkUnet.double_conv(net, int(base_feature_size*(2**i)), scope='down_conv_%d' % (i + 1))
                    features.append(net)
                    net = slim.max_pool2d(net, [2, 2], 2, padding='SAME', scope='pool%d' % (i + 1))

                # middle
                net = NetworkUnet.double_conv(net, max_feature_size, scope='middle_conv_1')

                # upsampling steps
                for i in range(step_size):
                    net = slim.conv2d_transpose(net, int(max_feature_size/(2**(i+1))), [2, 2], 2, scope='up_trans_conv_%d' % (i + 1))
                    down_feat = features.pop()  # upsample with origin version

                    assert net.shape[3] == down_feat.shape[3], '%d, %d, %d' % (i, net.shape[3], down_feat.shape[3])
                    net = tf.concat([down_feat, net], axis=-1)
                    net = NetworkUnet.double_conv(net, int(max_feature_size/(2**(i+1))), scope='up_conv_%d' % (i + 1))

                # not in the original paper
                net = NetworkUnet.double_conv(net, 32, scope='output_conv_1')

        # original paper : one 1x1 conv
        net = slim.convolution(net, 1, [3, 3], 1, scope='final_conv',
                               activation_fn=None,
                               padding='SAME',
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

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
