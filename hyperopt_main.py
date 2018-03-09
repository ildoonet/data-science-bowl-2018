import hyperopt
from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials

import hyperopt_optimizer

if __name__ == '__main__':
    space = hp.choice('parameters', [
        {
            'learning_rate': hp.choice('learning_rate', [0.01, 0.001, 0.0001]),
            'batchsize': hp.choice('batchsize', [16, 32, 64]),
            'model': hp.choice('model', ['simple_unet']),
            'epoch': hp.choice('epoch', [50])
        }
    ])
    trials = MongoTrials('mongo://hyper-mongo.devel.kakao.com:10247/curtis_db/jobs', exp_key='exp1')
    best = fmin(hyperopt_optimizer.objective, space, trials=trials, algo=tpe.suggest, max_evals=10)
    print(trials.best_trial['result'])
    # print(best)
    print(hyperopt.space_eval(space, best))