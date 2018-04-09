import numpy as np
import tensorflow as tf

from deeplab import common as deeplab_common
from deeplab import model as deeplab_model
from data_augmentation import data_to_segment_input, \
    data_to_image, random_flip_lr, random_flip_ud, random_scaling, random_affine, \
    random_color, data_to_normalize1, data_to_elastic_transform_wrapper, resize_shortedge_if_small, random_crop, \
    center_crop, random_color2, erosion_mask, resize_shortedge, mask_size_normalize, crop_mirror, center_crop_if_tcga
from data_feeder import CellImageDataManagerTrain, CellImageDataManagerValid, CellImageDataManagerTest
from tensorpack.dataflow.common import BatchData, MapData, MapDataComponent
from tensorpack.dataflow.parallel import PrefetchData

from hyperparams import HyperParams
from network import Network


class NetworkDeepLabV3p(Network):
    def __init__(self, batchsize):
        super().__init__()

        self.img_size = 224     # TODO : 513?
        self.atrous_rates = [6, 12, 18]
        self.output_stride = 16
        self.batchsize = batchsize

        self.input_batch = tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, 3), name='image')
        self.mask_batch = tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, 1), name='mask')
        self.weight_batch = tf.placeholder(tf.float32, shape=(None, self.img_size, self.img_size, 1), name='weight')
        self.unused = None
        self.logit = None
        self.output = None
        self.loss = None
        self.loss_opt = None

    def build(self):
        model_options = deeplab_common.ModelOptions(
            outputs_to_num_classes={'semantic': 1},
            crop_size=(self.img_size, self.img_size),
            atrous_rates=self.atrous_rates,
            output_stride=self.output_stride
        )

        outputs_to_scales_to_logits = deeplab_model.multi_scale_logits(
            self.input_batch,
            model_options=model_options,
            image_pyramid=None,     # [0.5, 0.75, 1.0, 1.25, 1.5]
            weight_decay=0.00004,
            is_training=self.is_training,
            fine_tune_batch_norm=True
        )

        self.logit = outputs_to_scales_to_logits['semantic']['merged_logits']
        self.output = tf.nn.sigmoid(self.logit, 'output')

        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=self.mask_batch,
            logits=self.logit,
            weights=self.weight_batch,
            scope='weighted_celoss'
        )
        self.loss = tf.check_numerics(loss, 'celoss_chk')
        self.loss_opt = self.loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='regterm')

        return self.output

    def get_input_flow(self):
        ds_train = CellImageDataManagerTrain()
        # ds_train = MapDataComponent(ds_train, random_affine)  # TODO : no improvement?
        ds_train = MapDataComponent(ds_train, random_color)
        # ds_train = MapDataComponent(ds_train, random_scaling)
        ds_train = MapDataComponent(ds_train, mask_size_normalize)  # Resize by instance size - normalization
        ds_train = MapDataComponent(ds_train, lambda x: resize_shortedge_if_small(x, self.img_size))
        ds_train = MapDataComponent(ds_train, lambda x: random_crop(x, self.img_size, self.img_size))
        ds_train = MapDataComponent(ds_train, random_flip_lr)
        ds_train = MapDataComponent(ds_train, random_flip_ud)
        # ds_train = MapDataComponent(ds_train, data_to_elastic_transform_wrapper)
        ds_train = MapDataComponent(ds_train, erosion_mask)
        ds_train = MapData(ds_train, lambda x: data_to_segment_input(x, is_gray=False, unet_weight=True))
        ds_train = PrefetchData(ds_train, 256, 24)
        ds_train = BatchData(ds_train, self.batchsize)
        ds_train = MapDataComponent(ds_train, data_to_normalize1)

        ds_valid = CellImageDataManagerValid()
        ds_valid = MapDataComponent(ds_valid, lambda x: resize_shortedge_if_small(x, self.img_size))
        ds_valid = MapDataComponent(ds_valid, lambda x: random_crop(x, self.img_size, self.img_size))
        ds_valid = MapDataComponent(ds_valid, erosion_mask)
        ds_valid = MapData(ds_valid, lambda x: data_to_segment_input(x, is_gray=False, unet_weight=True))
        ds_valid = PrefetchData(ds_valid, 20, 12)
        ds_valid = BatchData(ds_valid, self.batchsize, remainder=True)
        ds_valid = MapDataComponent(ds_valid, data_to_normalize1)

        ds_valid2 = CellImageDataManagerValid()
        ds_valid2 = MapDataComponent(ds_valid2, lambda x: resize_shortedge_if_small(x, self.img_size))
        ds_valid2 = MapDataComponent(ds_valid2, lambda x: center_crop_if_tcga(x, self.img_size, self.img_size))
        # ds_valid2 = MapDataComponent(ds_valid2, lambda x: resize_shortedge(x, self.img_size))
        ds_valid2 = MapData(ds_valid2, lambda x: data_to_segment_input(x, is_gray=False))
        ds_valid2 = MapDataComponent(ds_valid2, data_to_normalize1)

        ds_test = CellImageDataManagerTest()
        ds_test = MapDataComponent(ds_test, lambda x: resize_shortedge_if_small(x, self.img_size))
        # ds_test = MapDataComponent(ds_test, lambda x: resize_shortedge(x, self.img_size))
        ds_test = MapData(ds_test, lambda x: data_to_image(x, is_gray=False))
        ds_test = MapDataComponent(ds_test, data_to_normalize1)

        return ds_train, ds_valid, ds_valid2, ds_test

    def inference(self, tf_sess, image):
        # TODO : Mirror Padding?
        cascades, windows = Network.sliding_window(image, self.img_size, 0.5)

        # by batch
        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))

        outputs = []
        for b in chunker(cascades, 64):
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
            use_separator=False,
            cutoff_instance_max=0.9,
            cutoff_instance_avg=0.0
        )

        # instances = Network.watershed_merged_output(instances)

        return {
            'instances': instances,
            'scores': scores
        }

    def get_logit(self):
        return self.logit

    def get_placeholders(self):
        return self.input_batch, self.mask_batch, self.unused

    def get_output(self):
        return self.output

    def get_loss(self):
        return self.loss

    def get_loss_opt(self):
        return self.loss_opt

    def preprocess(self, x):
        x = resize_shortedge_if_small(x, self.img_size)   # self.img_size
        x = data_to_normalize1(x)
        return x

    def get_feeddict(self, dp, is_training):
        feed_dict = {
            self.input_batch: dp[0],
            self.mask_batch: dp[1],
            self.weight_batch: dp[3],
            self.is_training: is_training,
        }
        return feed_dict

    def get_pretrain_path(self):
        return './deeplab/pretrained/coco/model.ckpt'

    def get_optimize_op(self, global_step, learning_rate):
        """
        Need to override if you want to use different optimization policy.
        :param learning_rate:
        :param global_step:
        :return: (learning_rate, optimizer) tuple
        """
        learning_rate = tf.train.polynomial_decay(
            learning_rate, global_step,
            decay_steps=HyperParams.get().opt_decay_steps_deeplab,
            power=HyperParams.get().opt_decay_power_deeplab,
            end_learning_rate=0.0000001
        )
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate, global_step,
        #     decay_steps=HyperParams.get().opt_decay_steps,
        #     decay_rate=HyperParams.get().opt_decay_rate,
        #     staircase=True
        # )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=HyperParams.get().opt_momentum)
            # optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
            optimize_op = optimizer.minimize(self.get_loss_opt(), global_step, colocate_gradients_with_ops=True)
        return learning_rate, optimize_op
