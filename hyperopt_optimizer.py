from train import Trainer
from hyperopt import STATUS_OK


def objective(args):
    t = Trainer()
    print(args)
    miou = t.run(**args)
    return {
        'loss': -miou,
        'miou': miou,
        'status': STATUS_OK,
    }