import hyperopt
from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials

import hyperopt_optimizer
from submission import KaggleSubmission

if __name__ == '__main__':
    exp_key = 'unnetarch'
    print('---------------- experiments : %s ----------------' % exp_key)
    space = hp.choice('parameters', [
        {
            # 'batchsize': hp.choice('batchsize', [16, 32]),
            # 'learning_rate': hp.choice('learning_rate', [0.0005, 0.0001]),
            # 'opt_decay_steps': hp.choice('opt_decay_steps', [450, 700, 1200]),
            # 'opt_decay_rate': hp.choice('opt_decay_rate', [0.5, 0.9, 0.99]),
            # 'net_bn_decay': hp.choice('net_bn_decay', [0.9]),
            # 'net_bn_epsilon': hp.choice('net_bn_epsilon', [0.001]),
            # 'net_dropout_keep': hp.choice('net_dropout_keep', [0.9, 0.8, 0.7, 0.5]),
            # 'net_init_stddev': hp.choice('net_init_stddev', [0.02, 0.01, 0.005]),
            # 'unet_base_feature': hp.choice('unet_base_feature', [32, 48]),
            # 'unet_step_size': hp.choice('unet_step_size', [4, 5]),       # depth5 = out of memory

            # 'pre_erosion_iter': hp.choice('pre_erosion_iter', [1, 2, 3]),
            # 'post_dilation_iter': hp.choice('post_dilation_iter', [1, 2, 3, 4]),

            # 'pre_scale_f1': hp.uniform('pre_scale_f1', 0.2, 0.5),
            # 'pre_scale_f2': hp.uniform('pre_scale_f2', 0.2, 0.6),
            # 'pre_affine_rotate': hp.uniform('pre_affine_rotate', 10, 90),
            # 'pre_affine_shear': hp.uniform('pre_affine_shear', 2, 25),
            # 'pre_affine_translate': hp.uniform('pre_affine_translate', 0.1, 0.4),

            # 'post_fill_holes': hp.choice('post_fill_holes', [False, True]),

            # 'model': hp.choice('model', ['unet']),
        }
    ])

    trials = MongoTrials('mongo://hyper-mongo.devel.kakao.com:10247/curtis_db/jobs', exp_key=exp_key)
    best = fmin(hyperopt_optimizer.objective, space, trials=trials, algo=tpe.suggest, max_evals=4, verbose=1)
    print('--------------')
    print(trials.best_trial['result'])
    # print(best)
    print(hyperopt.space_eval(space, best))

    import json

    # automatic submission to Kaggle
    s = KaggleSubmission(trials.best_trial['result']['model_name'])
    msg, submission = s.submit_result('KakaoAutoML hyperopt %s' % exp_key)
    print('Experiment(%s) : %s\nLB Score: %s' % (exp_key, msg['message'], '' if submission is None else submission.publicScore))
