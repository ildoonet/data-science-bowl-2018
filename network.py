import abc

import cv2
import numpy as np
import tensorflow as tf
import slidingwindow as sw
from scipy import ndimage
from skimage.morphology import label

from colors import get_colors
from data_feeder import CellImageData
from separator import separation


class Network:
    @staticmethod
    def visualize(image, label, segments, weights, norm='norm01'):
        """
        For Visualization
        TODO
        """
        if image.dtype != np.uint8:
            if norm == 'norm01':
                image = (image * 255).astype(np.uint8)
            elif norm == 'norm1':
                image = ((image + 1.0) * 128).astype(np.uint8)
            else:
                raise

        columns = 1 + sum([1 for x in [label, segments, weights] if x is not None])
        colcnt = 0

        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        img_h, img_w = image.shape[:2]
        canvas = np.zeros((img_h, img_w * columns, 3), dtype=np.uint8)
        canvas[:, 0:img_w, :] = image
        colcnt += 1

        if segments is not None:
            canvas[:, img_w * colcnt:img_w * (colcnt + 1), :] = Network.visualize_segments(segments, image)
            colcnt += 1

        if label is not None:
            if not isinstance(label, list):
                label = CellImageData.batch_to_multi_masks(label, transpose=False)
                label = list(label)
            canvas[:, img_w * colcnt:img_w * (colcnt + 1), :] = Network.visualize_segments(label, image)
            colcnt += 1

        if weights is not None:
            canvas[:, img_w * colcnt:img_w * (colcnt + 1), :] = weights / 10. * 255
            colcnt += 1

        for n in range(colcnt):
            cv2.line(canvas, (img_w * (n + 1), 0), (img_w * (n + 1), img_h), (128, 128, 128), 1)

        return canvas

    @staticmethod
    def visualize_segments(segments, original_image):
        """
        Visualize Segments
        :param segments: (# of instances, h, w) or list of (h, w)
        :return: (h, w, 3) numpy image with colored instances
        """
        if not isinstance(segments, list):
            segments = Network.parse_merged_output(segments)

        img_h, img_w = original_image.shape[:2]
        canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for idx, seg in enumerate(segments):
            r, g, b = get_colors(idx)
            canvas[:, :, 0] = canvas[:, :, 0] + b * seg[:, :]
            canvas[:, :, 1] = canvas[:, :, 1] + g * seg[:, :]
            canvas[:, :, 2] = canvas[:, :, 2] + r * seg[:, :]
        return canvas

    @staticmethod
    def sliding_window(a, window, step_size):
        windows = sw.generate(a, sw.DimOrder.HeightWidthChannel, window, step_size)
        cascades = []
        for window in windows:
            subset = a[window.indices()]
            cascades.append(subset)
        return cascades, windows

    @staticmethod
    def parse_merged_output(output, cutoff=0.5, use_separator=True, use_dilation=False):
        """
        Split 1-channel merged output for instance segmentation
        :param cutoff:
        :param output: (h, w, 1) segmentation image
        :return: list of (h, w, 1). instance-aware segmentations.
        """
        # TODO : Sharpening?
        if use_separator:
            # Ref: https://www.kaggle.com/bostjanm/overlapping-objects-separation-method/notebook
            labels = label(output > cutoff, connectivity=1)
            reconstructed_mask = np.zeros(output.shape, dtype=np.bool)
            for i in range(1, labels.max() + 1):
                # separate objects
                img_ = separation(labels == i)
                # copy to reconstructed mask
                reconstructed_mask = reconstructed_mask + img_
            output = reconstructed_mask
        lab_img = label(output > cutoff, connectivity=1)
        if use_dilation:
            for i in range(1, lab_img.max() + 1):
                lab_img = np.maximum(lab_img, ndimage.morphology.binary_dilation(lab_img == i, iterations=2) * i)
        instances = []
        for i in range(1, lab_img.max() + 1):
            instances.append(lab_img == i)
        return instances

    @staticmethod
    def resize_instances(instances, target_size):
        h, w = target_size

        def resize_instance(instance):
            shp = instance.shape
            instance = (instance * 255).astype(np.uint8)
            instance = cv2.resize(instance, (w, h))
            instance = instance >> 7
            instance = instance[..., np.newaxis]
            assert len(shp[:2]) == len(instance.shape[:2])
            return instance

        instances = [resize_instance(instance) for instance in instances]

        # make sure that there are no overlappings
        lab_img = np.zeros((h, w, 1), dtype=np.int32)
        for i, instance in enumerate(instances):
            lab_img = np.maximum(lab_img, instance * (i+1))
        new_instances = []
        for i in range(1, lab_img.max() + 1):
            new_instances.append(lab_img == i)

        return new_instances

    def __init__(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    @abc.abstractmethod
    def get_input_flow(self):
        pass

    @abc.abstractmethod
    def get_placeholders(self):
        pass

    @abc.abstractmethod
    def get_feeddict(self, dp, is_training):
        pass

    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def inference(self, tf_sess, image):
        """
        Function to Process an image and Generate instance-aware segmentation result.
        :return:    list of (h, w), which have 0 or 1 values
        """
        pass

    @abc.abstractmethod
    def get_output(self):
        pass

    @abc.abstractmethod
    def get_logit(self):
        pass

    @abc.abstractmethod
    def get_loss(self):
        pass

    def get_optimize_op(self, learning_rate, global_step):
        """
        Need to override if you want to use different optimization policy.
        :param learning_rate:
        :param global_step:
        :return: (learning_rate, optimizer) tuple
        """
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   decay_steps=300, decay_rate=0.33, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0)
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
            optimize_op = optimizer.minimize(self.get_loss(), global_step, colocate_gradients_with_ops=True)
        return learning_rate, optimize_op

    def get_is_training(self):
        return self.is_training
