import hyperopt
from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials

import hyperopt_optimizer
from submission import KaggleSubmission

if __name__ == '__main__':
    exp_key = 'exp_morph2'
    space = hp.choice('parameters', [
        {
            # 'batchsize': hp.choice('batchsize', [16, 32]),
            # 'learning_rate': hp.choice('learning_rate', [0.0005, 0.0001]),
            # 'opt_decay_steps': hp.choice('opt_decay_steps', [450, 700]),
            # 'opt_decay_rate': hp.choice('opt_decay_rate', [0.9, 0.95, 0.99]),
            # 'net_bn_decay': hp.choice('net_bn_decay', [0.9]),
            # 'net_bn_epsilon': hp.choice('net_bn_epsilon', [0.001]),
            # 'net_dropout_keep': hp.choice('net_dropout_keep', [0.9, 0.8, 0.7, 0.5]),
            # 'net_init_stddev': hp.choice('net_init_stddev', [0.02, 0.01, 0.005]),
            # 'unet_base_feature': hp.choice('unet_base_feature', [16, 24, 32]),
            # 'unet_step_size': hp.choice('unet_step_size', [3, 4]),

            'pre_erosion_iter': hp.choice('pre_erosion_iter', [1, 2, 3]),
            'post_dilation_iter': hp.choice('post_dilation_iter', [1, 2, 3, 4]),

            # 'pre_scale_f1': hp.uniform('pre_scale_f1', 0.2, 0.5),
            # 'pre_scale_f2': hp.uniform('pre_scale_f2', 0.2, 0.6),
            # 'pre_affine_rotate': hp.uniform('pre_affine_rotate', 10, 90),
            # 'pre_affine_shear': hp.uniform('pre_affine_shear', 5, 30),
            # 'pre_affine_translate': hp.uniform('pre_affine_translate', 0.1, 0.4),

            # 'post_fill_holes': hp.choice('post_fill_holes', [False, True]),

            'model': hp.choice('model', ['unet']),
        }
    ])
    # 180312T1551410335_unet_lr=0.0001_epoch=500_bs=16, 5940 - 0.4884
    """
    0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466.jpg
    259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4.jpg
    505bc0a3928d8aef5ce441c5a611fdd32e1e8eccdc15cc3a52b88030acb50f81.jpg
    5cee644e5ffbef1ba021c7f389b33bafd3b1841f04d3edd7922d5084c2c4e0c7.jpg
    8922a6ac8fd0258ec27738ca101867169b20d90a60fc84f93df77acd5bf7c80b.jpg
    f5effed21f671bbf4551ecebb7fe95f3be1cf09c16a60afe64d2f0b95be9d1eb.jpg
    """

    trials = MongoTrials('mongo://hyper-mongo.devel.kakao.com:10247/curtis_db/jobs', exp_key=exp_key)
    best = fmin(hyperopt_optimizer.objective, space, trials=trials, algo=tpe.suggest, max_evals=16, verbose=1)
    print('--------------')
    print(trials.best_trial['result'])
    # print(best)
    print(hyperopt.space_eval(space, best))

    import json
    from slackclient import SlackClient
    slack_token = 'xoxp-142308306661-142322097044-329945619031-c8721e6403a7f9cab4b9bd61270060b3'

    sc = SlackClient(slack_token, proxies={
        'https': 'http://10.41.249.28:8080',
        'http': 'http://10.41.249.28:8080',
    })

    sc.api_call(
        "chat.postMessage",
        channel="599-prj-kaggle-gazua",
        text='Experiment(%s) Finished : %s' % (exp_key, json.dumps(trials.best_trial['result']))
    )

    # automatic submission to Kaggle
    s = KaggleSubmission(trials.best_trial['result']['model_name'])
    msg, submission = s.submit_result('KakaoAutoML hyperopt %s' % exp_key)

    sc.api_call(
        "chat.postMessage",
        channel="599-prj-kaggle-gazua",
        text='Experiment(%s) : %s\nLB Score: %s' % (exp_key, msg['message'], '' if submission is None else submission.publicScore)
    )
