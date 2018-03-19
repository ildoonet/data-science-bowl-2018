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

        self.pre_scale_f1 = 0.4
        self.pre_scale_f2 = 0.4
        self.pre_affine_rotate = 45
        self.pre_affine_shear = 5
        self.pre_affine_translate = 0.1     # 0.4?

        self.net_dropout_keep = 0.9         # 0.8?
        self.net_bn_decay = 0.9
        self.net_bn_epsilon = 0.001
        self.net_init_stddev = 0.01

        self.unet_base_feature = 32         # 24?
        self.unet_step_size = 4

        self.pre_erosion_iter = 1
        self.post_dilation_iter = 2

        self.post_fill_holes = False
