import os
import importlib

from .metric import Metric

__METRICS_DICT__ = {}


def build_metric(name):
    return __METRICS_DICT__[name]


def metric(name):
    
    def register_function_fn(cls):
        if name in __METRICS_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Metric):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Metric))
        __METRICS_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.evaluation.metric.' + module_name)