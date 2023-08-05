import os
import importlib

from .dataset import ALDataset

__DATASET_DICT__ = {}


def build_dataset(name):
    return __DATASET_DICT__[name]


def dataset(name):
    
    def register_function_fn(cls):
        if name in __DATASET_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, ALDataset):
            raise ValueError("Class %s is not a subclass of %s" % (cls, ALDataset))
        __DATASET_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.dataset.' + module_name)    
