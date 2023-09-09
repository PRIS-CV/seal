from pprint import pprint


class Metric(object):

    def __init__(self, name, **kwargs) -> None:
        self.name = name
        self._result = {}

    def calculate_metric(self):
        raise NotImplementedError("calculate_metric method is not implemented")

    def get_result(self):
        return self._result

    def reset(self):
        self._result = {}



    