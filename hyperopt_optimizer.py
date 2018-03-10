import os

from hyperparams import HyperParams
from train import Trainer
from hyperopt import STATUS_OK


def objective(args):
    t = Trainer()
    print(args)
    print('------------')
    for k, v in args.items():
        if k not in HyperParams.get().__dict__.keys():
            continue
        HyperParams.get().__dict__[k] = v
    print(HyperParams.get().__dict__)
    print('------------')
    miou = t.run('unet')
    return {
        'loss': 1.0 - miou,
        'miou': miou,
        'task_group_id': os.environ.get('TASK_GROUP_ID', ''),
        'task_group_name': os.environ.get('TASK_GROUP_NAME', ''),
        'task_name': os.environ.get('TASK_NAME', ''),
        'status': STATUS_OK,
    }
