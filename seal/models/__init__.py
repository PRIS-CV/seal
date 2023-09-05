__all__ = ["build_model", "build_model_config"]


import os
import importlib
from .model import ALModel
from .config import ALModelConfig


__MODEL_DICT__ = {}
__MODEL_CONFIG_DICT__ = {}


def build_model(name):
    return __MODEL_DICT__[name]

def build_model_config(name):
    return __MODEL_CONFIG_DICT__[name]

def almodel(name):
    
    def register_function_fn(cls):
        if name in __MODEL_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, ALModel):
            raise ValueError("Class %s is not a subclass of %s" % (cls, ALModel))
        __MODEL_DICT__[name] = cls
        return cls

    return register_function_fn

def almodel_config(name):
    
    def register_function_fn(cls):
        if name in __MODEL_CONFIG_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, ALModelConfig):
            raise ValueError("Class %s is not a subclass of %s" % (cls, ALModelConfig))
        __MODEL_CONFIG_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):

    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.models.' + module_name)

    elif os.path.isdir(os.path.join(os.path.dirname(__file__), file)):
        module_name = file

        # FIXME monkey patching
        _not_import_dir = ["__pycache__", "backbone", "decoder", "layer", "loss", "attention"]
        if module_name not in _not_import_dir :
            module = importlib.import_module('seal.models.' + module_name)
        
