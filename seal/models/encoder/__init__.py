import os
import importlib

from .encoder import Encoder


__ENCODER_DICT__ = {}


def build_encoder(name):
    return __ENCODER_DICT__[name]

def encoder(name):

    def register_function_fn(cls):
        if name in __ENCODER_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Encoder):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Encoder))
        __ENCODER_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.models.encoder.' + module_name)
