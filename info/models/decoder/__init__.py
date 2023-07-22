import os
import importlib

from .decoder import Decoder


__DECODER_DICT__ = {}


def build_decoder(name):
    return __DECODER_DICT__[name]

def decoder(name):

    def register_function_fn(cls):
        if name in __DECODER_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Decoder):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Decoder))
        __DECODER_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('info.models.decoder.' + module_name)
        