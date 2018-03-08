import logging

import os
import cv2
import datetime
import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from checkmate.checkmate import BestCheckpointSaver, get_best_checkpoint
from data_feeder import CellImageData
from network import Network
from network_basic import NetworkBasic
from network_unet import NetworkUnet
from network_fusionnet import NetworkFusionNet
from submission import KaggleSubmission, get_multiple_metric

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Trainer:
    def run(self, model, epoch=30, batchsize=32, learning_rate=0.01, valid_interval=2,
            tag='', show_train=0, show_valid=0, show_test=0, checkpoint=''):
        if model == 'basic':
            network = NetworkBasic(batchsize, unet_weight=True)
        elif model == 'simple_unet':
            network = NetworkUnet(batchsize, unet_weight=True)
        elif model == 'simple_fusion':
            network = NetworkFusionNet(batchsize)
        else:
            raise Exception('model name(%s) is not valid' % model)

        logger.info('constructing network model: %s' % model)

        ds_train, ds_valid, ds_valid_full, ds_test = network.get_input_flow()
        network.build()

        net_output = network.get_output()
        net_loss = network.get_loss()

        global_step = tf.Variable(0, trainable=False)
        learning_rate_v, train_op = network.get_optimize_op(learning_rate, global_step)

        logger.info('constructed-')

        best_loss_val = 999999
        best_miou_val = 0.0
        name = '%s_%s_lr=%.4f_epoch=%d_bs=%d' % (
            tag if tag else datetime.datetime.now().strftime("%y%m%dT%H%M"),
            model,
            learning_rate,
            epoch,
            batchsize,
        )
        model_path = os.path.join(KaggleSubmission.BASEPATH, name, 'model')
        best_ckpt_saver = BestCheckpointSaver(
            save_dir=model_path,
            num_to_keep=100,
            maximize=True
        )
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            logger.info('training started+')
            if not checkpoint:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, checkpoint)
                logger.info('restore from checkpoint, %s' % checkpoint)

            try:
                for e in range(epoch):
                    for dp_train in ds_train.get_data():
                        _, step, lr, loss_val = sess.run(
                            [train_op, global_step, learning_rate_v, net_loss],
                            feed_dict=network.get_feeddict(dp_train, True)
                        )
                        # for debug
                        # cv2.imshow('train', Network.visualize(dp_train[0][0], dp_train[2][0], None, dp_train[3][0], 'norm1'))
                        # cv2.waitKey(0)

                    logger.info('training %d epoch %d step, lr=%.8f loss=%.4f' % (e+1, step, lr, loss_val))

                    if (e + 1) % valid_interval == 0:
                        avg = []
                        metrics = []
                        for _ in range(5):
                            for dp_valid in ds_valid.get_data():
                                loss_val = sess.run(
                                    net_loss,
                                    feed_dict=network.get_feeddict(dp_valid, False)
                                )
                                avg.append(loss_val)

                        avg = sum(avg) / len(avg)
                        logger.info('validation loss=%.4f' % (avg))
                        if best_loss_val > avg:
                            best_loss_val = avg

                    if e > 10 and (e + 1) % 5 == 0:
                        thr_list = np.arange(0.5, 1.0, 0.05)
                        cnt_tps = np.array((len(thr_list)), dtype=np.int32),
                        cnt_fps = np.array((len(thr_list)), dtype=np.int32)
                        cnt_fns = np.array((len(thr_list)), dtype=np.int32)
                        for idx, dp_valid in tqdm(enumerate(ds_valid_full.get_data()), 'validate using the iou metric'):
                            image = dp_valid[0]
                            label = CellImageData.batch_to_multi_masks(dp_valid[2], transpose=False)
                            instances = network.inference(sess, image)

                            cnt_tp, cnt_fp, cnt_fn = get_multiple_metric(thr_list, instances, label)
                            cnt_tps = cnt_tps + cnt_tp
                            cnt_fps = cnt_fps + cnt_fp
                            cnt_fns = cnt_fns + cnt_fn

                        ious = np.divide(cnt_tps, cnt_tps + cnt_fps + cnt_fns)
                        mIou = np.mean(ious)
                        logger.info('validation metric: %.5f' % mIou)
                        if best_miou_val < mIou:
                            best_miou_val = mIou
                        best_ckpt_saver.handle(mIou, sess, global_step)  # save & keep best model
            except KeyboardInterrupt:
                logger.info('interrupted. stop training, start to validate.')

            try:
                chk_path = get_best_checkpoint(model_path, select_maximum_value=True)
                logger.info('training is done. Start to evaluate the best model. %s' % chk_path)
                saver.restore(sess, chk_path)
            except Exception as e:
                logger.warning(str(e))

            # show sample in train set : show_train > 0
            for idx, dp_train in enumerate(ds_train.get_data()):
                if idx >= show_train:
                    break
                image = dp_train[0][0]
                instances = network.inference(sess, image)

                cv2.imshow('train', Network.visualize(image, dp_train[2][0], instances, None))
                cv2.waitKey(0)

            # show sample in valid set : show_valid > 0
            logging.info('Start to test on validation set.... (may take a while)')
            for idx, dp_valid in enumerate(ds_valid_full.get_data()):
                if idx >= show_valid:
                    break
                image = dp_valid[0]
                instances = network.inference(sess, image)

                cv2.imshow('valid', Network.visualize(image, dp_valid[2], instances, None))
                cv2.waitKey(0)

            # show sample in test set
            kaggle_submit = KaggleSubmission(name)
            for idx, dp_test in enumerate(ds_test.get_data()):
                image = dp_test[0]
                test_id = dp_test[1][0]
                img_h, img_w = dp_test[2][0], dp_test[2][1]
                assert img_h > 0 and img_w > 0, '%d %s' % (idx, test_id)
                instances = network.inference(sess, image)

                img_vis = Network.visualize(image, None, instances, None)
                if idx < show_test:
                    cv2.imshow('test', img_vis)
                    cv2.waitKey(0)

                # save to submit
                instances = Network.resize_instances(instances, (img_h, img_w))
                kaggle_submit.save_image(test_id, img_vis)
                kaggle_submit.add_result(test_id, instances)
            kaggle_submit.save()
        logger.info('done. best_loss_val=%.4f best_mIOU=%.4f name=%s' % (best_loss_val, best_miou_val, name))
        return best_miou_val


if __name__ == '__main__':
    fire.Fire(Trainer)
