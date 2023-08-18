import time
import json
import os
import os.path as op
from pprint import pprint
from collections import OrderedDict

class Evaluation(object):
    
    def __init__(self, directory) -> None:
        
        self.directory = directory
        if not op.exists(directory):
            os.makedirs(directory)
        self._result = OrderedDict()
        self.build_metrics()

    def build_metrics(self):
        raise NotImplementedError("build_metrics method is not implemented")

    def update_date(self):
        self._result["date"] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    def print_result(self):
        pprint(self._result)

    def save_result(self):
        self.update_date()
        date = self._result['date']
        with open(op.join(self.directory, f"eval-log-{date}.json"), "w") as f:
            json.dump(self._result, f, indent=4)
        print(f"Save result to {op.join(self.directory, f'eval-log-{date}.json')}")

    def __call__(self):
        raise NotImplementedError("__call__ method is not implemented")

    def _clear_result(self):
        self._result = {}
    
