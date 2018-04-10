import abc

import cv2
import numpy as np
import tensorflow as tf
import slidingwindow as sw
from scipy import ndimage
from skimage.filters import threshold_local
from skimage.morphology import label

from colors import get_colors
from data_augmentation import get_size_of_mask
from hyperparams import HyperParams
from data_feeder import batch_to_multi_masks
from separator import separation
from submission import get_iou


class Network:
    @staticmethod
    def visualize(image, label, segments, weights, norm='norm1'):
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
                raise Exception('unspecified norm')

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
                label = batch_to_multi_masks(label, transpose=False)
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
    def parse_merged_output(output, cutoff=0.5, use_separator=False, cutoff_instance_max=0.8, cutoff_instance_avg=0.2):
        """
        Split 1-channel merged output for instance segmentation
        :param cutoff:
        :param output: (h, w, 1) segmentation image
        :return: list of (h, w, 1). instance-aware segmentations.
        """
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

        if cutoff > 0.0:
            cutoffed = output > cutoff
        else:
            # local threshold : https://www.kaggle.com/nirofa95/basic-thresholding-and-morphological-approach/code
            block_size = 25
            thresh = threshold_local(output, block_size, method='median', offset=-2)
            cutoffed = output > thresh

        lab_img = label(cutoffed, connectivity=1)
        instances = []
        for i in range(1, lab_img.max() + 1):
            instances.append((lab_img == i).astype(np.bool))

        filtered_instances = []
        scores = []
        for instance in instances:
            # TODO : max or avg?
            instance_score_max = np.max(instance * output)    # score max
            if instance_score_max < cutoff_instance_max:
                continue
            instance_score_avg = np.sum(instance * output) / np.sum(instance)   # score avg
            if instance_score_avg < cutoff_instance_avg:
                continue
            filtered_instances.append(instance)
            scores.append(instance_score_avg)
        instances = filtered_instances

        # dilation
        instances_tmp = []
        if HyperParams.get().post_dilation_iter > 0:
            for instance in filtered_instances:
                instance = ndimage.morphology.binary_dilation(instance, iterations=HyperParams.get().post_dilation_iter)
                instances_tmp.append(instance)
            instances = instances_tmp

        # sorted by size
        sorted_idx = [i[0] for i in sorted(enumerate(instances), key=lambda x: get_size_of_mask(x[1]))]
        instances = [instances[x] for x in sorted_idx]
        scores = [scores[x] for x in sorted_idx]

        # make sure there are no overlaps
        instances, scores = Network.remove_overlaps(instances, scores)

        # fill holes
        if HyperParams.get().post_fill_holes:
            instances = [ndimage.morphology.binary_fill_holes(i) for i in instances]

        return instances, scores

    @staticmethod
    def remove_overlaps(instances, scores):
        if len(instances) == 0:
            return []
        lab_img = np.zeros(instances[0].shape, dtype=np.int32)
        for i, instance in enumerate(instances):
            lab_img = np.maximum(lab_img, instance * (i + 1))
        instances = []
        new_scores = []
        for i in range(1, lab_img.max() + 1):
            instance = (lab_img == i).astype(np.bool)
            if np.max(instance) == 0:
                continue
            instances.append(instance)
            new_scores.append(scores[i - 1])
        return instances, new_scores

    @staticmethod
    def watershed_merged_output(instances):
        # ref : https://docs.opencv.org/3.3.1/d3/db4/tutorial_py_watershed.html
        new_instances = []
        for idx, instance in enumerate(instances):
            if idx == 0:
                continue

            # noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(instance.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=8)
            # sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            # Now, mark the region of unknown with zero
            markers[unknown == 1] = 0

            markers = cv2.watershed(cv2.cvtColor(instance.astype(np.uint8), cv2.COLOR_GRAY2BGR), markers)

            for idx in range(1, markers.max() + 1):
                instance = (markers == idx)
                # instance = ndimage.morphology.binary_fill_holes(markers == idx)
                instance = instance.astype(np.uint8)
                # instance = cv2.dilate(instance, kernel, iterations=2)
                new_instances.append(instance)
        return new_instances

    @staticmethod
    def resize_instances(instances, target_size):
        h, w = target_size

        def resize_instance(instance):
            shp = instance.shape
            instance = (instance * 255).astype(np.uint8)
            instance = cv2.resize(instance, (w, h), interpolation=cv2.INTER_AREA)
            instance = instance >> 7
            if len(shp) > len(instance.shape):
                instance = instance[..., np.newaxis]
            assert len(shp) == len(instance.shape)
            return instance

        instances = [resize_instance(instance) for instance in instances]

        # make sure that there are no overlappings
        if len(instances) > 0 and len(instances[0].shape) == 2:
            lab_img = np.zeros((h, w), dtype=np.int32)
        else:
            lab_img = np.zeros((h, w, 1), dtype=np.int32)
        for i, instance in enumerate(instances):
            lab_img = np.maximum(lab_img, instance * (i+1))
        new_instances = []
        for i in range(1, lab_img.max() + 1):
            new_instances.append(lab_img == i)

        return new_instances

    @staticmethod
    def nms(instances, scores, from_set=None, thresh=0.3):
        scores = np.array(scores)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx1 = order[0]
            keep.append(idx1)

            ovr = []
            for idx2 in order[1:]:
                if from_set is not None and from_set[idx1] == from_set[idx2]:
                    ovr.append(0.0)
                    continue
                ovr.append(get_iou(instances[idx1], instances[idx2]))
            ovr = np.array(ovr, dtype=np.float32)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return [instances[x] for x in keep], [scores[x] for x in keep]

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

    @abc.abstractmethod
    def get_loss_opt(self):
        pass

    @abc.abstractmethod
    def preprocess(self, x):
        pass

    def get_pretrain_path(self):
        return ''

    def get_optimize_op(self, global_step, learning_rate):
        """
        Need to override if you want to use different optimization policy.
        :param learning_rate:
        :param global_step:
        :return: (learning_rate, optimizer) tuple
        """
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   decay_steps=HyperParams.get().opt_decay_steps,
                                                   decay_rate=HyperParams.get().opt_decay_rate,
                                                   staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if HyperParams.get().optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
            elif HyperParams.get().optimizer == 'rmsprop':
                # not good
                optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=HyperParams.get().opt_momentum)
            elif HyperParams.get().optimizer == 'sgd':
                # not optimized
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif HyperParams.get().optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=HyperParams.get().opt_momentum)
            else:
                raise Exception('invalid optimizer: %s' % HyperParams.get().optimizer)
            optimize_op = optimizer.minimize(self.get_loss_opt(), global_step, colocate_gradients_with_ops=True)
        return learning_rate, optimize_op

    def get_is_training(self):
        return self.is_training
