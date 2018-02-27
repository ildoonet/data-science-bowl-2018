import abc

import cv2
import numpy as np
import tensorflow as tf
import slidingwindow as sw
from skimage.morphology import label

from colors import get_colors
from separator import separation


class Network:
    @staticmethod
    def visualize(image, label, segments):
        """
        For Visualization
        TODO
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        img_h, img_w = image.shape[:2]
        if label is not None:
            canvas = np.zeros((img_h, img_w * 3, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((img_h, img_w * 2, 3), dtype=np.uint8)
        canvas[:, 0:img_w, :] = image
        canvas[:, img_w*1:img_w*2, :] = Network.visualize_segments(segments, image)

        if label is not None:
            canvas[:, img_w*2:img_w*3, :] = Network.visualize_segments(label, image)

        cv2.line(canvas, (img_w, 0), (img_w, img_h), (128, 128, 128), 1)
        cv2.line(canvas, (img_w*2, 0), (img_w*2, img_h), (128, 128, 128), 1)

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
    def parse_merged_output(output, cutoff=0.5, use_separator=True):
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
        return instances

    def __init__(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    @abc.abstractmethod
    def get_input_flow(self):
        pass

    @abc.abstractmethod
    def get_placeholders(self):
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
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
            optimize_op = optimizer.minimize(self.get_loss(), global_step, colocate_gradients_with_ops=True)
        return learning_rate, optimize_op

    def get_is_training(self):
        return self.is_training
