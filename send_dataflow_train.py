import fire
from tensorpack import send_dataflow_zmq

from network_unet_valid import NetworkUnetValid


class TrainDataSender:
    def run(self, batchsize=16, unet_weight=True):
        network = NetworkUnetValid(batchsize, unet_weight=unet_weight)
        df = network.get_input_flow_train()
        send_dataflow_zmq(df, 'tcp://trainer:8877')


if __name__ == '__main__':
    fire.Fire(TrainDataSender)
