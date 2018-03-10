import hyperopt
from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials

import hyperopt_optimizer


if __name__ == '__main__':
    space = hp.choice('parameters', [
        {
            # 'batchsize': hp.choice('batchsize', [16, 32]),
            # 'opt_decay_steps': hp.choice('opt_decay_steps', [250, 350, 450, 1000]),
            # 'opt_decay_rate': hp.choice('opt_decay_rate', [0.1, 0.3, 0.5, 0.9]),
            # 'net_bn_decay': hp.choice('net_bn_decay', [0.9]),
            # 'net_bn_epsilon': hp.choice('net_bn_epsilon', [0.001]),
            # 'net_dropout_keep': hp.choice('net_dropout_keep', [0.9, 0.8, 0.7, 0.5]),
            # 'net_init_stddev': hp.choice('net_init_stddev', [0.02, 0.01, 0.005, 0.001]),
            # 'unet_base_feature': hp.choice('unet_base_feature', [16, 32]),
            # 'unet_step_size': hp.choice('unet_step_size', [3, 4]),

            'pre_erosion_iter': hp.choice('pre_erosion_iter', [1, 2, 3]),
            'post_dilation_iter': hp.choice('post_dilation_iter', [1, 2, 3, 4]),

            # 'pre_scale_f1': hp.choice('pre_scale_f1', [0.3, 0.4, 0.5]),
            # 'pre_scale_f2': hp.choice('pre_scale_f2', [0.3, 0.4, 0.5, 0.6]),
            # 'pre_affine_rotate': hp.choice('pre_affine_rotate', [30, 45, 60, 90]),
            # 'pre_affine_shear': hp.choice('pre_affine_shear', [5, 10, 20, 30]),
            # 'pre_affine_translate': hp.choice('pre_affine_translate', [0.1, 0.3]),

            # 'post_fill_holes': hp.choice('post_fill_holes', [False, True]),

            'model': hp.choice('model', ['unet']),
            # 'learning_rate': hp.choice('learning_rate', [0.001, 0.0001]),
        }
    ])
    trials = MongoTrials('mongo://hyper-mongo.devel.kakao.com:10247/curtis_db/jobs', exp_key='exp4')
    best = fmin(hyperopt_optimizer.objective, space, trials=trials, algo=tpe.suggest, max_evals=12)
    print(trials.best_trial['result'])
    # print(best)
    print(hyperopt.space_eval(space, best))
