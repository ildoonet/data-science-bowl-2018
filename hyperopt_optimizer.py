import os

from hyperparams import HyperParams
from train import Trainer
from hyperopt import STATUS_OK, STATUS_FAIL


def objective(args):
    print('------------ STARTED')
    t = Trainer()
    print(args)
    print('------------')
    for k, v in args.items():
        if k not in HyperParams.get().__dict__.keys():
            continue
        HyperParams.get().__dict__[k] = v
    print(HyperParams.get().__dict__)
    print('------------')
    miou, name = t.run('unet')
    print(miou, name)
    print('------------ FINISHED')

    if miou <= 0.0:
        return {
            'loss': 1.0,
            'status': STATUS_FAIL,
        }

    return {
        'loss': 1.0 - miou,
        'miou': miou,
        'model_name': name,
        'task_group_id': os.environ.get('TASK_GROUP_ID', ''),
        'task_group_name': os.environ.get('TASK_GROUP_NAME', ''),
        'task_name': os.environ.get('TASK_NAME', ''),
        'status': STATUS_OK,
    }
