import logging

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import Pool
from itertools import compress

import sys
from scipy import ndimage

import pickle
import cv2
import datetime
import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from checkmate.checkmate import BestCheckpointSaver, get_best_checkpoint
from commons import chunker
from data_augmentation import get_max_size_of_masks, mask_size_normalize, center_crop, get_size_of_mask
from data_feeder import batch_to_multi_masks, CellImageData, master_dir_test, master_dir_train, \
    CellImageDataManagerValid, CellImageDataManagerTrain, CellImageDataManagerTest, extra1_dir, extra2_dir
from hyperparams import HyperParams
from network import Network
from network_basic import NetworkBasic
from network_deeplabv3p import NetworkDeepLabV3p
from network_unet import NetworkUnet
from network_fusionnet import NetworkFusionNet
from network_unet_valid import NetworkUnetValid
from stopwatch import StopWatch
from submission import KaggleSubmission, get_multiple_metric, thr_list, get_iou

logger = logging.getLogger('train')
logger.setLevel(logging.INFO if os.environ.get('DEBUG', 0) == 0 else logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


class Trainer:
    def __init__(self):
        self.batchsize = 16
        self.network = None
        self.sess = None

        self.ensembles = None

    def set_network(self, model, batchsize=16):
        if model == 'basic':
            self.network = NetworkBasic(batchsize, unet_weight=True)
        elif model == 'simple_unet':
            self.network = NetworkUnet(batchsize, unet_weight=True)
        elif model == 'unet':
            self.network = NetworkUnetValid(batchsize)
        elif model == 'deeplabv3p':
            self.network = NetworkDeepLabV3p(batchsize)
        elif model == 'simple_fusion':
            self.network = NetworkFusionNet(batchsize)
        else:
            raise Exception('model name(%s) is not valid' % model)
        logger.info('constructing network model: %s' % model)

    def init_session(self):
        if self.sess is not None:
            return
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=config)

    def run(self, model, epoch=600,
            batchsize=16, learning_rate=0.0001, early_rejection=False,
            valid_interval=10, tag='', save_result=True, checkpoint='',
            pretrain=False, skip_train=False, validate_train=True, validate_valid=True,
            logdir='/data/public/rw/kaggle-data-science-bowl/logs/',
            **kwargs):
        self.set_network(model, batchsize)
        ds_train, ds_valid, ds_valid_full, ds_test = self.network.get_input_flow()
        self.network.build()
        print(HyperParams.get().__dict__)

        net_output = self.network.get_output()
        net_loss = self.network.get_loss()

        global_step = tf.Variable(0, trainable=False)
        learning_rate_v, train_op = self.network.get_optimize_op(global_step=global_step,
                                                                 learning_rate=learning_rate)

        best_loss_val = 999999
        best_miou_val = 0.0
        name = '%s_%s_lr=%.8f_epoch=%d_bs=%d' % (
            tag if tag else datetime.datetime.now().strftime("%y%m%dT%H%M%f"),
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
        m_epoch = 0

        # initialize session
        self.init_session()

        # tensorboard
        tf.summary.scalar('loss', net_loss, collections=['train', 'valid'])
        s_train = tf.summary.merge_all('train')
        s_valid = tf.summary.merge_all('valid')
        train_writer = tf.summary.FileWriter(logdir + name + '/train', self.sess.graph)
        valid_writer = tf.summary.FileWriter(logdir + name + '/valid', self.sess.graph)

        logger.info('initialization+')
        if not checkpoint:
            self.sess.run(tf.global_variables_initializer())

            if pretrain:
                global_vars = tf.global_variables()

                from tensorflow.python import pywrap_tensorflow
                reader = pywrap_tensorflow.NewCheckpointReader(self.network.get_pretrain_path())
                var_to_shape_map = reader.get_variable_to_shape_map()
                saved_vars = list(var_to_shape_map.keys())

                var_list = [x for x in global_vars if x.name.replace(':0', '') in saved_vars]
                var_list = [x for x in var_list if 'logit' not in x.name]
                logger.info('pretrained weights(%d) loaded : %s' % (len(var_list), self.network.get_pretrain_path()))

                pretrain_loader = tf.train.Saver(var_list)
                pretrain_loader.restore(self.sess, self.network.get_pretrain_path())
        elif checkpoint == 'best':
            path = get_best_checkpoint(model_path)
            saver.restore(self.sess, path)
            logger.info('restored from best checkpoint, %s' % path)
        elif checkpoint == 'latest':
            path = tf.train.latest_checkpoint(model_path)
            saver.restore(self.sess, path)
            logger.info('restored from latest checkpoint, %s' % path)
        else:
            saver.restore(self.sess, checkpoint)
            logger.info('restored from checkpoint, %s' % checkpoint)

        step = self.sess.run(global_step)
        start_e = (batchsize * step) // len(CellImageDataManagerTrain.LIST)

        logger.info('training started+')
        if epoch > 0 and not skip_train:
            try:
                losses = []
                for e in range(start_e, epoch):
                    loss_val_avg = []
                    train_cnt = 0
                    for dp_train in ds_train.get_data():
                        _, loss_val, summary_train = self.sess.run([train_op, net_loss, s_train], feed_dict=self.network.get_feeddict(dp_train, True))
                        loss_val_avg.append(loss_val)
                        train_cnt += 1

                    step, lr = self.sess.run([global_step, learning_rate_v])
                    loss_val_avg = sum(loss_val_avg) / len(loss_val_avg)
                    logger.info('training %d epoch %d step, lr=%.8f loss=%.4f train_iter=%d' % (
                        e + 1, step, lr, loss_val_avg, train_cnt))
                    losses.append(loss_val)
                    train_writer.add_summary(summary_train, global_step=step)

                    if early_rejection and len(losses) > 100 and losses[len(losses) - 100] * 1.05 < loss_val_avg:
                        logger.info('not improved, stop at %d' % e)
                        break

                    # early rejection
                    if early_rejection and ((e == 50 and loss_val > 0.5) or (e == 200 and loss_val > 0.2)):
                        logger.info('not improved training loss, stop at %d' % e)
                        break

                    m_epoch = e
                    avg = 10.0
                    if loss_val < 0.20 and (e + 1) % valid_interval == 0:
                        avg = []
                        for _ in range(5):
                            ds_valid.reset_state()
                            ds_valid_d = ds_valid.get_data()
                            for dp_valid in ds_valid_d:
                                loss_val, summary_valid = self.sess.run(
                                    [net_loss, s_valid],
                                    feed_dict=self.network.get_feeddict(dp_valid, False)
                                )

                                avg.append(loss_val)
                            ds_valid_d.close()

                        avg = sum(avg) / len(avg)
                        logger.info('validation loss=%.4f' % (avg))
                        if best_loss_val > avg:
                            best_loss_val = avg
                        valid_writer.add_summary(summary_valid, global_step=step)

                    if avg < 0.16 and e >= 100 and (e + 1) % valid_interval == 0:
                        cnt_tps = np.array((len(thr_list)), dtype=np.int32),
                        cnt_fps = np.array((len(thr_list)), dtype=np.int32)
                        cnt_fns = np.array((len(thr_list)), dtype=np.int32)
                        pool_args = []
                        ds_valid_full.reset_state()
                        ds_valid_full_d = ds_valid_full.get_data()
                        for idx, dp_valid in tqdm(enumerate(ds_valid_full_d), desc='validate using the iou metric', total=len(CellImageDataManagerValid.LIST)):
                            image = dp_valid[0]
                            inference_result = self.network.inference(self.sess, image, cutoff_instance_max=0.9)
                            instances, scores = inference_result['instances'], inference_result['scores']
                            pool_args.append((thr_list, instances, dp_valid[2]))
                        ds_valid_full_d.close()

                        pool = Pool(processes=8)
                        cnt_results = pool.map(do_get_multiple_metric, pool_args)
                        pool.close()
                        pool.join()
                        pool.terminate()
                        for cnt_result in cnt_results:
                            cnt_tps = cnt_tps + cnt_result[0]
                            cnt_fps = cnt_fps + cnt_result[1]
                            cnt_fns = cnt_fns + cnt_result[2]

                        ious = np.divide(cnt_tps, cnt_tps + cnt_fps + cnt_fns)
                        mIou = np.mean(ious)
                        logger.info('validation metric: %.5f' % mIou)
                        if best_miou_val < mIou:
                            best_miou_val = mIou
                        best_ckpt_saver.handle(mIou, self.sess, global_step)  # save & keep best model

                        # early rejection by mIou
                        if early_rejection and e > 50 and best_miou_val < 0.15:
                            break
                        if early_rejection and e > 100 and best_miou_val < 0.25:
                            break
            except KeyboardInterrupt:
                logger.info('interrupted. stop training, start to validate.')

        try:
            chk_path = get_best_checkpoint(model_path, select_maximum_value=True)
            if chk_path:
                logger.info('training is done. Start to evaluate the best model. %s' % chk_path)
                saver.restore(self.sess, chk_path)
        except Exception as e:
            logger.warning('error while loading the best model:' + str(e))

        # show sample in train set : show_train > 0
        kaggle_submit = KaggleSubmission(name)
        if validate_train in [True, 'True', 'true']:
            logger.info('Start to test on training set.... (may take a while)')
            train_metrics = []
            for single_id in tqdm(CellImageDataManagerTrain.LIST[:20], desc='training set test'):
                result = self.single_id(None, None, single_id, set_type='train', show=False, verbose=False)
                image = result['image']
                labels = result['labels']
                instances = result['instances']
                score = result['score']
                score_desc = result['score_desc']

                img_vis = Network.visualize(image, labels, instances, None)
                kaggle_submit.save_train_image(single_id, img_vis, score=score, score_desc=score_desc)
                train_metrics.append(score)
            logger.info('trainset validation ends. score=%.4f' % np.mean(train_metrics))

        # show sample in valid set : show_valid > 0
        if validate_valid in [True, 'True', 'true']:
            logger.info('Start to test on validation set.... (may take a while)')
            valid_metrics = []
            for single_id in tqdm(CellImageDataManagerValid.LIST, desc='validation set test'):
                result = self.single_id(None, None, single_id, set_type='train', show=False, verbose=False)
                image = result['image']
                labels = result['labels']
                instances = result['instances']
                score = result['score']
                score_desc = result['score_desc']

                img_vis = Network.visualize(image, labels, instances, None)
                kaggle_submit.save_valid_image(single_id, img_vis, score=score, score_desc=score_desc)
                kaggle_submit.valid_instances[single_id] = (instances, result['instance_scores'])
                valid_metrics.append(score)
            logger.info('validation ends. score=%.4f' % np.mean(valid_metrics))

        # show sample in test set
        logger.info('saving...')
        if save_result:
            for single_id in tqdm(CellImageDataManagerTest.LIST, desc='test set evaluation'):
                result = self.single_id(None, None, single_id, set_type='test', show=False, verbose=False)
                image = result['image']
                instances = result['instances']
                img_h, img_w = image.shape[:2]

                img_vis = Network.visualize(image, None, instances, None)

                # save to submit
                instances = Network.resize_instances(instances, (img_h, img_w))
                kaggle_submit.save_image(single_id, img_vis)
                kaggle_submit.test_instances[single_id] = (instances, result['instance_scores'])
                kaggle_submit.add_result(single_id, instances)
            kaggle_submit.save()
        logger.info('done. epoch=%d best_loss_val=%.4f best_mIOU=%.4f name= %s' % (m_epoch, best_loss_val, best_miou_val, name))
        return best_miou_val, name

    def validate(self, network=None, checkpoint=None, **kwargs):
        if network is not None:
            self.set_network(network)
            self.network.build()

        self.init_session()

        mIOU = []
        self.init_session()
        if checkpoint:
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint)
            logger.info('restored from checkpoint, %s' % checkpoint)

        for single_id in CellImageDataManagerValid.LIST:
            result = self.single_id(None, None, single_id, set_type='train', show=False, verbose=True)
            score = result['score']
            mIOU.append(score)
        mIOU = np.mean(mIOU)
        logger.info('mScore = %.5f' % mIOU)
        return mIOU

    def _get_cell_data(self, single_id, set_type):
        if 'TCGA' in single_id:
            d = CellImageData(single_id, extra1_dir, ext='tif')
            # generally, TCGAs have lots of instances -> slow matching performance
            d = center_crop(d, 224, 224, padding=0)
        elif 'TNBC' in single_id:
            d = CellImageData(single_id, extra2_dir, ext='png')
            # generally, TCGAs have lots of instances -> slow matching performance
            d = center_crop(d, 224, 224, padding=0)
        else:
            d = CellImageData(single_id, (master_dir_train if set_type == 'train' else master_dir_test))
        return d

    def single_id(self, model, checkpoint, single_id, set_type='train', show=True, verbose=True):
        if model:
            self.set_network(model)
            self.network.build()

        self.init_session()
        if checkpoint:
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint)
            if verbose:
                logger.info('restored from checkpoint, %s' % checkpoint)

        d = self._get_cell_data(single_id, set_type)
        h, w = d.img.shape[:2]
        shortedge = min(h, w)
        if verbose:
            logger.info('%s image size=(%d x %d)' % (single_id, w, h))

        watch = StopWatch()
        logger.debug('preprocess+')
        d = self.network.preprocess(d)

        image = d.image(is_gray=False)

        total_instances = []
        total_scores = []
        total_from_set = []
        cutoff_instance_max = HyperParams.get().post_cutoff_max_th
        cutoff_instance_avg = HyperParams.get().post_cutoff_avg_th

        watch.start()
        logger.debug('inference at default scale+')
        inference_result = self.network.inference(self.sess, image, cutoff_instance_max=cutoff_instance_max, cutoff_instance_avg=cutoff_instance_avg)
        instances_pre, scores_pre = inference_result['instances'], inference_result['scores']
        instances_pre = Network.resize_instances(instances_pre, target_size=(h, w))
        total_instances = total_instances + instances_pre
        total_scores = total_scores + scores_pre
        total_from_set = [1] * len(instances_pre)
        watch.stop()
        logger.debug('inference- elapsed=%.5f' % watch.get_elapsed())
        watch.reset()

        logger.debug('inference with flips+')
        # re-inference using flip
        for flip_orientation in range(2):
            flipped = cv2.flip(image.copy(), flip_orientation)
            inference_result = self.network.inference(self.sess, flipped, cutoff_instance_max=cutoff_instance_max, cutoff_instance_avg=cutoff_instance_avg)
            instances_flip, scores_flip = inference_result['instances'], inference_result['scores']
            instances_flip = [cv2.flip(instance.astype(np.uint8), flip_orientation) for instance in instances_flip]
            instances_flip = Network.resize_instances(instances_flip, target_size=(h, w))

            total_instances = total_instances + instances_flip
            total_scores = total_scores + scores_flip
            total_from_set = total_from_set + [2 + flip_orientation] * len(instances_flip)

        watch.stop()
        logger.debug('inference- elapsed=%.5f' % watch.get_elapsed())
        watch.reset()
        logger.debug('inference with scaling+flips+')

        # re-inference after rescale image
        def inference_with_scale(image, resize_target):
            image = cv2.resize(image.copy(), None, None, resize_target, resize_target, interpolation=cv2.INTER_AREA)
            inference_result = self.network.inference(self.sess, image, cutoff_instance_max=cutoff_instance_max, cutoff_instance_avg=cutoff_instance_avg)
            instances_rescale, scores_rescale = inference_result['instances'], inference_result['scores']

            instances_rescale = Network.resize_instances(instances_rescale, target_size=(h, w))
            return instances_rescale, scores_rescale

        max_mask = get_max_size_of_masks(instances_pre)
        logger.debug('max_mask=%d' % max_mask)
        resize_target = HyperParams.get().test_aug_scale_t / max_mask
        resize_target = min(HyperParams.get().test_aug_scale_max, resize_target)
        resize_target = max(HyperParams.get().test_aug_scale_min, resize_target)
        import math
        # resize_target = 2.0 / (1.0 + math.exp(-1.5*(resize_target - 1.0)))
        # resize_target = max(0.5, resize_target)
        resize_target = max(228.0 / shortedge, resize_target)
        logger.debug('resize_target=%.4f' % resize_target)

        instances_rescale, scores_rescale = inference_with_scale(image, resize_target)
        total_instances = total_instances + instances_rescale
        total_scores = total_scores + scores_rescale
        total_from_set = total_from_set + [4] * len(instances_rescale)

        # re-inference using flip + rescale
        for flip_orientation in range(2):
            flipped = cv2.flip(image.copy(), flip_orientation)
            instances_flip, scores_flip = inference_with_scale(flipped, resize_target)
            instances_flip = [cv2.flip(instance.astype(np.uint8), flip_orientation) for instance in instances_flip]
            instances_flip = Network.resize_instances(instances_flip, target_size=(h, w))

            total_instances = total_instances + instances_flip
            total_scores = total_scores + scores_flip
            total_from_set = total_from_set + [5 + flip_orientation] * len(instances_flip)

        watch.stop()
        logger.debug('inference- elapsed=%.5f' % watch.get_elapsed())
        watch.reset()

        watch.start()
        logger.debug('voting+ size=%d' % len(total_instances))
        # TODO : Voting?
        voting_th = HyperParams.get().post_voting_th
        with ProcessPoolExecutor(max_workers=None) as executor:
            voted = executor.map(filter_by_voting, [(x, total_instances, voting_th) for x in total_instances], chunksize=64)
            voted = list(voted)
        total_instances = list(compress(total_instances, voted))
        total_scores = list(compress(total_scores, voted))
        total_from_set = list(compress(total_from_set, voted))

        watch.stop()
        logger.debug('voting elapsed=%.5f' % watch.get_elapsed())
        watch.reset()

        # nms
        watch.start()
        logger.debug('nms+ size=%d' % len(total_instances))
        instances, scores = Network.nms(total_instances, total_scores, total_from_set, thresh=HyperParams.get().test_aug_nms_iou)
        watch.stop()
        logger.debug('nms elapsed=%.5f' % watch.get_elapsed())
        watch.reset()

        # remove overlaps
        logger.debug('remove overlaps+')
        sorted_idx = [i[0] for i in sorted(enumerate(instances), key=lambda x: get_size_of_mask(x[1]), reverse=True)]
        instances = [instances[x] for x in sorted_idx]
        scores = [scores[x] for x in sorted_idx]

        instances = [ndimage.morphology.binary_fill_holes(i) for i in instances]
        instances, scores = Network.remove_overlaps(instances, scores)

        # TODO : Filter by score?
        logger.debug('filter by score+')
        score_filter_th = HyperParams.get().post_filter_th
        if score_filter_th > 0.0:
            logger.debug('filter_by_score=%.3f' % score_filter_th)
            instances = [i for i, s in zip(instances, scores) if s > score_filter_th]
            scores = [s for i, s in zip(instances, scores) if s > score_filter_th]

        logger.debug('finishing+')
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        score_desc = []
        labels = []
        if len(d.masks) > 0:    # has label masks
            labels = list(d.multi_masks(transpose=False))
            labels = Network.resize_instances(labels, target_size=(h, w))
            tp, fp, fn = get_multiple_metric(thr_list, instances, labels)

            if verbose:
                logger.info('instances=%d, reinf(%.3f) labels=%d' % (len(instances), resize_target, len(labels)))
            for i, thr in enumerate(thr_list):
                desc = 'score=%.3f, tp=%d, fp=%d, fn=%d --- iou %.2f' % (
                    (tp / (tp + fp + fn))[i],
                    tp[i],
                    fp[i],
                    fn[i],
                    thr
                )
                if verbose:
                    logger.info(desc)
                score_desc.append(desc)
            score = np.mean(tp / (tp + fp + fn))
            if verbose:
                logger.info('score=%.3f, tp=%.1f, fp=%.1f, fn=%.1f --- mean' % (
                    score,
                    np.mean(tp),
                    np.mean(fp),
                    np.mean(fn)
                ))
        else:
            score = 0.0

        if show:
            img_vis = Network.visualize(image, labels, instances, None)
            cv2.imshow('valid', img_vis)
            cv2.waitKey(0)
        if not model:
            return {
                'instance_scores': scores,
                'score': score,
                'image': image,
                'instances': instances,
                'labels': labels,
                'score_desc': score_desc
            }

    def _load_ensembles(self, model):
        ensemble_models = {
            'stage1_test': {
                'rcnn': [
                ],
                'unet': [
                    "/data/public/rw/kaggle-data-science-bowl/submissions/lb525_ensemble_s80px(0.5-2.0)xflip/submission_lb525.pkl"
                ]
            },
            'stage1_unet': {
                'rcnn': [
                    "/data/public/rw/datasets/dsb2018/pickles/pickled_mask_info_full.pkl"
                ],
                'unet': [
                    "/data/public/rw/kaggle-data-science-bowl/submissions/lb525_ensemble_s80px(0.5-2.0)xflip/submission_lb525.pkl",
                    "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold1_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
                    "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold2_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
                    "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold3_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
                    "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold4_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
                    "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold5_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
                    "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold6_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
                    "/data/public/rw/kaggle-data-science-bowl/submissions/stage1_folds7/unetv1_fold7_thick_unet_lr=0.00010000_epoch=600_bs=16/submission.pkl",
                ]
            }
        }

        if self.ensembles is not None:
            return
        self.ensembles = {'rcnn': [], 'unet': []}

        models = ensemble_models[model]
        # TODO : RCNN Load
        for path in models['rcnn']:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.ensembles['rcnn'].append(data)

        for path in models['unet']:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.ensembles['unet'].append(data)

        logger.debug('_load_ensembles-')

    def ensemble_models(self, model='stage1_unet', set_type='test', tag='default', **kwargs):
        kaggle_submit = KaggleSubmission('ensemble_%s_%s' % (tag, model))

        # show sample in test set
        logger.info('testset...')
        for single_id in tqdm(CellImageDataManagerTest.LIST, desc='test set evaluation'):
            result = self.ensemble_models_id(single_id, set_type=set_type, model=model, show=False, verbose=False)
            image = result['image']
            instances = result['instances']
            img_h, img_w = image.shape[:2]

            img_vis = Network.visualize(image, None, instances, None)

            # save to submit
            instances = Network.resize_instances(instances, (img_h, img_w))
            kaggle_submit.save_image(single_id, img_vis)
            kaggle_submit.test_instances[single_id] = (instances, result['instance_scores'])
            kaggle_submit.add_result(single_id, instances)
        kaggle_submit.save()

    def ensemble_models_id(self, single_id, set_type='train', model='stage1_unet', show=True, verbose=True):
        self._load_ensembles(model)
        d = self._get_cell_data(single_id, set_type)
        logger.debug('image size=%dx%d' % (d.img_h, d.img_w))

        total_model_size = len(self.ensembles['rcnn']) + len(self.ensembles['unet'])
        logger.debug('total_model_size=%d rcnn=%d unet=%d' % (total_model_size, len(self.ensembles['rcnn']), len(self.ensembles['unet'])))

        rcnn_instances = []
        rcnn_scores = []

        # TODO : RCNN Ensemble
        for idx, data in enumerate(self.ensembles['rcnn']):
            if set_type == 'train':
                instances, scores = data['valid_instances'].get(single_id, (None, None))
            else:
                # TODO
                ls = data['test_instances'].get(single_id, None)
                instances = [x[0] for x in ls]
                scores = [x[1] for x in ls]
                logger.debug('rcnn # instances = %d' % len(instances))

            if instances is None:
                logger.debug('Not found id=%s in UNet %d Model' % (single_id, idx + 1))
                continue

            rcnn_instances.extend([instance[:d.img_h, :d.img_w] for instance in instances])
            rcnn_scores.extend([s * HyperParams.get().rcnn_score_rescale for s in scores])     # rescale scores

        total_instances = []
        total_scores = []

        # TODO : UNet Ensemble
        for idx, data in enumerate(self.ensembles['unet']):
            if set_type == 'train':
                instances, scores = data['valid_instances'].get(single_id, (None, None))
            else:
                instances, scores = data['test_instances'].get(single_id, (None, None))

            if instances is None:
                logger.debug('Not found id=%s in UNet %d Model' % (single_id, idx + 1))
                continue

            total_instances.extend(instances)
            total_scores.extend(scores)

        watch = StopWatch()
        watch.start()
        logger.debug('voting+ size=%d' % len(total_instances))

        # TODO : Voting?
        voting_th = HyperParams.get().ensemble_voting_th
        voted = []

        with ProcessPoolExecutor(max_workers=12) as executor:
            try:
                args = [(x, total_instances, voting_th) for x in total_instances]
                result = executor.map(filter_by_voting, args, chunksize=64)
                voted.extend(list(result))
            except Exception as e:
                logger.error('%s instances=%d err=%s' % (single_id, len(total_instances), str(e)))
                sys.exit(-1)

        total_instances = list(compress(total_instances, voted))
        total_scores = list(compress(total_scores, voted))

        watch.stop()
        logger.debug('voting elapsed=%.5f' % watch.get_elapsed())
        watch.reset()

        # nms
        watch.start()
        logger.debug('nms+ size=%d' % len(total_instances))
        instances, scores = Network.nms(total_instances, total_scores, None, thresh=HyperParams.get().ensemble_nms_iou)
        watch.stop()
        logger.debug('nms elapsed=%.5f' % watch.get_elapsed())
        watch.reset()

        # high threshold if not exists in RCNN
        voted = []
        with ProcessPoolExecutor(max_workers=12) as executor:
            try:
                result = executor.map(filter_by_voting, [(x, rcnn_instances, 1, 0.3) for x in instances], chunksize=64)
                voted.extend(list(result))
            except Exception as e:
                logger.error('%s instances=%d err_rcnn=%s' % (single_id, len(total_instances), str(e)))
                sys.exit(-1)

        new_instances = []
        new_scores = []
        for instance, score, v in zip(instances, scores, voted):
            if v:
                new_instances.append(instance)
                new_scores.append(score)
            elif score > HyperParams.get().ensemble_th_no_rcnn:
                new_instances.append(instance)
                new_scores.append(score)
        instances, scores = new_instances, new_scores

        # nms with rcnn
        instances = instances + rcnn_instances
        scores = scores + rcnn_scores
        watch.start()
        logger.debug('nms_rcnn+ size=%d' % len(instances))
        instances, scores = Network.nms(instances, scores, None, thresh=HyperParams.get().ensemble_nms_iou)
        watch.stop()
        logger.debug('nms_rcnn elapsed=%.5f' % watch.get_elapsed())
        watch.reset()

        # remove overlaps
        logger.debug('remove overlaps+')
        sorted_idx = [i[0] for i in sorted(enumerate(instances), key=lambda x: get_size_of_mask(x[1]), reverse=False)]
        instances = [instances[x] for x in sorted_idx]
        scores = [scores[x] for x in sorted_idx]

        instances2 = [ndimage.morphology.binary_fill_holes(i) for i in instances]
        instances2, scores2 = Network.remove_overlaps(instances2, scores)

        # remove deleted instances
        voted = []
        with ProcessPoolExecutor(max_workers=12) as executor:
            try:
                result = executor.map(filter_by_voting, [(x, instances, 1, 0.75) for x in instances2], chunksize=64)
                voted.extend(list(result))
            except:
                logger.error('%s instances=%d err2 ' % (single_id, len(total_instances)))
                sys.exit(-1)

        instances = list(compress(instances2, voted))
        scores = list(compress(scores2, voted))

        # TODO : Filter by score?
        logger.debug('filter by score+')
        score_filter_th = HyperParams.get().ensemble_score_th
        if score_filter_th > 0.0:
            logger.debug('filter_by_score=%.3f' % score_filter_th)
            instances = [i for i, s in zip(instances, scores) if s > score_filter_th]
            scores = [s for i, s in zip(instances, scores) if s > score_filter_th]

        logger.debug('finishing+')
        image = d.image(is_gray=False)
        score_desc = []
        labels = []
        if len(d.masks) > 0:  # has label masks
            labels = list(d.multi_masks(transpose=False))
            tp, fp, fn = get_multiple_metric(thr_list, instances, labels)

            if verbose:
                logger.info('instances=%d, labels=%d' % (len(instances), len(labels)))
            for i, thr in enumerate(thr_list):
                desc = 'score=%.3f, tp=%d, fp=%d, fn=%d --- iou %.2f' % (
                    (tp / (tp + fp + fn))[i],
                    tp[i],
                    fp[i],
                    fn[i],
                    thr
                )
                if verbose:
                    logger.info(desc)
                score_desc.append(desc)
            score = np.mean(tp / (tp + fp + fn))
            if verbose:
                logger.info('score=%.3f, tp=%.1f, fp=%.1f, fn=%.1f --- mean' % (
                    score,
                    np.mean(tp),
                    np.mean(fp),
                    np.mean(fn)
                ))
        else:
            score = 0.0

        if show:
            img_vis = Network.visualize(image, labels, instances, None)
            cv2.imshow('valid', img_vis)
            cv2.waitKey(0)
        else:
            return {
                'instance_scores': scores,
                'score': score,
                'image': image,
                'instances': instances,
                'labels': labels,
                'score_desc': score_desc
            }


def do_get_multiple_metric(args):
    thr_list, instances, multi_masks_batch = args
    if np.max(multi_masks_batch) == 0:
        # no label
        label = []
    else:
        label = batch_to_multi_masks(multi_masks_batch, transpose=False)
    return get_multiple_metric(thr_list, instances, label)


def filter_by_voting(args):
    if len(args) == 3:
        x, total_list, voting_th = args
        iou_th = 0.3
    else:
        x, total_list, voting_th, iou_th = args
    if np.sum(np.array([get_iou(x, x2) for x2 in total_list]) > iou_th) >= voting_th:
        return True
    return False


if __name__ == '__main__':
    fire.Fire(Trainer)
    print(HyperParams.get().__dict__)
