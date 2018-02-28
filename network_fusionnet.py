import tensorflow as tf
from tensorflow.contrib import slim
from network_basic import NetworkBasic


class NetworkFusionNet(NetworkBasic):
    def __init__(self, batchsize):
        super(NetworkFusionNet, self).__init__(batchsize=batchsize)

    # ResBlock
    @staticmethod
    def res_block(net, nb_filter, scope):
        residual = net
        net = slim.convolution(net, nb_filter, [1, 1], 1, scope='%s_res_1' % scope)
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_res_2' % scope)
        net = slim.convolution(net, nb_filter, [1, 1], 1, scope='%s_res_3' % scope)
        return net + residual

    # ConvResConv
    @staticmethod
    def conv_res_conv(net, nb_filter, scope):
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_1' % scope)
        net = __class__.res_block(net, nb_filter, scope)
        net = slim.convolution(net, nb_filter, [3, 3], 1, scope='%s_2' % scope)
        return net

    def build(self):
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L429
        batch_norm_params = {
            'is_training': self.is_training,
            'center': True,
            'scale': False,
            'decay': 0.97,
            'epsilon': 0.001,
            'fused': True,
            # 'zero_debias_moving_mean': True
        }

        conv_args = {
            'padding': 'SAME',
            # 'weights_initializer': tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
            'weights_initializer': tf.contrib.layers.xavier_initializer(),
            'normalizer_fn': slim.batch_norm,
            'normalizer_params': batch_norm_params,
            'activation_fn': tf.nn.elu
        }

        net = self.input_batch
        features = []
        with slim.arg_scope([slim.convolution, slim.conv2d_transpose], **conv_args):
            layers_depth = 3
            for i in range(layers_depth):
                net = __class__.conv_res_conv(net, int(64 * (2 ** i)), scope='down_conv_res_conv_%d' % (i + 1))
                features.append(net)
                net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool%d' % (i + 1))

            mid_conv_size = 64 * (2 ** layers_depth)
            net = __class__.conv_res_conv(net, mid_conv_size, scope='middle_conv_1')

            down_conv_size = mid_conv_size // 2
            for i in range(layers_depth):
                current_down_conv_size = down_conv_size // (2 ** i)
                net = slim.conv2d_transpose(net, current_down_conv_size, [3, 3], 2, scope='up_trans_conv_%d' % (i + 1))
                down_feat = features.pop()  # upsample with origin version
                # net = tf.concat([down_feat, net], axis=-1)
                net = down_feat + net
                net = __class__.conv_res_conv(net, current_down_conv_size, scope='up_conv_%d' % (i + 1))

        net = slim.convolution(net, 1, [3, 3], 1, scope='final_conv',
                               padding='SAME',
                               activation_fn=None,
                               weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

        self.logit = net
        self.output = tf.nn.sigmoid(net, 'visualization')
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.mask_batch, logits=self.logit))
        return net
