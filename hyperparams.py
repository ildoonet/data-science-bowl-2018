import os


class HyperParams:
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def get():
        """ Static access method. """
        if HyperParams.__instance is None:
            HyperParams()
        return HyperParams.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if HyperParams.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            HyperParams.__instance = self

        self.opt_decay_steps = 450
        self.opt_decay_rate = 0.98          # 0.3?
        self.opt_decay_steps_deeplab = 4000
        self.opt_decay_power_deeplab = 0.9
        self.opt_momentum = 0.9
        self.optimizer = 'adam'

        self.pre_scale_f1 = 0.4
        self.pre_scale_f2 = 0.4
        self.pre_affine_rotate = 45
        self.pre_affine_shear = 5
        self.pre_affine_translate = 0.1     # 0.4?
        self.pre_size_norm_min = 10         # in pixel
        self.pre_size_norm_max = 150         # in pixel

        # 1~7, 7-folds
        self.data_fold = int(os.environ.get('fold', 1))
        print('---------- data folds = %d ---------' % self.data_fold)

        self.net_bn_decay = 0.9
        self.net_bn_epsilon = 0.001
        self.net_dropout_keep = 0.9
        self.net_init_stddev = 0.01

        self.unet_base_feature = 32
        self.unet_step_size = 4

        self.pre_erosion_iter = 1
        self.post_dilation_iter = 2
        self.post_fill_holes = False
        self.post_filter_th = 0.0
        self.post_voting_th = 4
        self.post_cutoff_max_th = 0.9
        self.post_cutoff_avg_th = 0.0
        self.test_aug_nms_iou = 0.5
        self.test_aug_scale_max = 2.0
        self.test_aug_scale_min = 0.75
        self.test_aug_scale_t = 80.0

        # ensemble between models
        self.rcnn_score_rescale = 0.95
        self.ensemble_voting_th = 3
        self.ensemble_th_no_rcnn = 0.75
        self.ensemble_nms_iou = 0.3
        self.ensemble_score_th = 0.6
