__all__ = ["build_layer"]


import os
import importlib

from info.models.layer.layer import Layer


__LAYER_DICT__ = {}


def build_layer(name):
    return __LAYER_DICT__[name]

def layer(name):

    def register_function_fn(cls):
        if name in __LAYER_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Layer):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Layer))
        __LAYER_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('info.models.layer.' + module_name)
