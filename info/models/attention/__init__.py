__all__ = ["build_attention"]

import os
import importlib

from info.models.attention.attention import Attention


__ATTENTION_DICT__ = {}


def build_attention(name):
    return __ATTENTION_DICT__[name]


def attention(name):
    
    def register_function_fn(cls):
        if name in __ATTENTION_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Attention):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Attention))
        __ATTENTION_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('info.models.attention.' + module_name)
