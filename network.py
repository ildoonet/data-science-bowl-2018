import abc


class Network:
    @abc.abstractmethod
    def get_input_flow(self):
        pass

    @abc.abstractmethod
    def get_placeholders(self):
        pass

    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def postprocess(self, output):
        pass
