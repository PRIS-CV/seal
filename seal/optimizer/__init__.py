__all__ = ["build_optimizer"]

import os
import importlib


__OPTIMIZER_DICT__ = {}


def build_optimizer(name):
    return __OPTIMIZER_DICT__[name]

def optimizer(name):
    
    def register_function_fn(cls):
        if name in __OPTIMIZER_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __OPTIMIZER_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('models.' + module_name)