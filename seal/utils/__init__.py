import os
import importlib


__EVAL_UTILS_DICT__ = {}
__TRAIN_UTILS_DICT__ = {}


def build_eval_util(name):
    return __EVAL_UTILS_DICT__[name]

def build_train_util(name):
    return __TRAIN_UTILS_DICT__[name]

def eval_util(name):
    
    def register_function_fn(cls):
        if name in __EVAL_UTILS_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __EVAL_UTILS_DICT__[name] = cls
        return cls

    return register_function_fn


def train_util(name):
    
    def register_function_fn(cls):
        if name in __TRAIN_UTILS_DICT__:
            raise ValueError("Name %s already registered!" % name)
        __TRAIN_UTILS_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.utils.' + module_name)
