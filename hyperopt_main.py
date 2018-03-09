import hyperopt
from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials

import hyperopt_optimizer


if __name__ == '__main__':
    space = hp.choice('parameters', [
        {
            'learning_rate': hp.choice('learning_rate', [0.001, 0.0001]),
            'batchsize': hp.choice('batchsize', [16, 32]),
            'decay_steps': hp.choice('decay_steps', [100, 150, 200, 250, 300, 350, 400, 450, 500]),
            'decay_rate': hp.choice('decay_rate', [0.1, 0.2, 0.3, 0.4, 0.5]),
            'batch_norm_decay': hp.choice('batch_norm_decay', [0.9]),
            'batch_norm_epsilon': hp.choice('batch_norm_epsilon', [0.001]),
            'keep_prob': hp.choice('keep_prob', [0.9, 0.8, 0.7]),
            'stddev': hp.choice('stddev', [0.01]),
            'model': hp.choice('model', ['simple_unet']),
            'epoch': hp.choice('epoch', [100])
        }
    ])
    trials = MongoTrials('mongo://hyper-mongo.devel.kakao.com:10247/bryan_db/jobs', exp_key='exp7')
    best = fmin(hyperopt_optimizer.objective, space, trials=trials, algo=tpe.suggest, max_evals=30)
    print(trials.best_trial['result'])
    # print(best)
    print(hyperopt.space_eval(space, best))
