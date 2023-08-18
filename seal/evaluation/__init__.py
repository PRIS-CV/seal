import os
import importlib

from .evaluation import Evaluation


__EVAL_DICT__ = {}


def build_evaluation(name):
    return __EVAL_DICT__[name]


def evaluation(name):
    
    def register_function_fn(cls):
        if name in __EVAL_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Evaluation):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Evaluation))
        __EVAL_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.evaluation.' + module_name)