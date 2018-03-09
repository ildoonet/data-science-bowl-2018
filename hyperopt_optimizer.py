import os

from train import Trainer
from hyperopt import STATUS_OK


def objective(args):
    t = Trainer()
    print(args)
    miou = t.run(**args)
    return {
        'loss': 1.0 - miou,
        'miou': miou,
        'task_group_id': os.environ.get('TASK_GROUP_ID', ''),
        'task_group_name': os.environ.get('TASK_GROUP_NAME', ''),
        'task_name': os.environ.get('TASK_NAME', ''),
        'status': STATUS_OK,
    }
