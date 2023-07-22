import os
import importlib

from .loss import Loss


__LOSS_DICT__ = {}


def build_loss(name):
    return __LOSS_DICT__[name]


def loss(name):

    def register_function_fn(cls):
        if name in __LOSS_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Loss):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Loss))
        __LOSS_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('info.models.loss.' + module_name)
