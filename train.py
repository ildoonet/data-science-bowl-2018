import logging

import cv2
import fire
import numpy
import tensorflow as tf

from data_queue import DataFlowToQueue
from network_basic import NetworkBasic


logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Trainer:
    def run(self, model, epoch=20, batchsize=32, learning_rate=0.01):
        if model == 'basic':
            network = NetworkBasic(batchsize)
        else:
            raise

        ds_train, ds_valid, ds_test = network.get_input_flow()
        ph_image, ph_mask, ph_masks = network.get_placeholders()

        net_output = network.build()
        net_sigmoid = tf.nn.sigmoid(net_output, 'visualization')
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_mask, logits=net_output))

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                   decay_steps=100, decay_rate=0.1, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step, colocate_gradients_with_ops=True)

        logger.info('constructed-')

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            logger.info('training started+')
            sess.run(tf.global_variables_initializer())

            tmp_img = None
            tmp_label = None
            tmp_mask = None
            for e in range(epoch):
                for dp_train in ds_train.get_data():
                    _, step, lr, loss_val, output = sess.run([train_op, global_step, learning_rate, loss, net_sigmoid], feed_dict={
                        ph_image: dp_train[0],
                        ph_mask: dp_train[1]
                    })
                    tmp_img = dp_train[0]
                    tmp_label = dp_train[1]
                    tmp_mask = output
                logger.info('training %d epoch %d step, lr=%.6f loss=%.4f' % (e, step, lr, loss_val))

            # show sample
            print(numpy.max(tmp_mask[0]))
            cv2.imshow('image', (tmp_img[0] * 255).astype(numpy.uint8))
            cv2.imshow('label', (tmp_label[0] * 255).astype(numpy.uint8))
            cv2.imshow('pred', (tmp_mask[0] * 255).astype(numpy.uint8))
            cv2.waitKey(0)


if __name__ == '__main__':
    fire.Fire(Trainer)