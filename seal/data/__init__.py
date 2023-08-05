__all__ = ["build_dataset", "build_transform", "build_pipeline", "build_data_encoder"]


import os
import importlib

from seal.data.data import ALDataset, Transform, Pipeline, DataEncoder


__DATASET_DICT__ = {}
__TRANSFORM_DICT__ = {}
__PIPELINE_DICT__ = {}
__DATAENCODER_DICT__ = {}


def build_dataset(name):
    return __DATASET_DICT__[name]

def build_transform(name):
    return __TRANSFORM_DICT__[name]

def build_pipeline(name):
    return __PIPELINE_DICT__[name]

def build_data_encoder(name):
    return __DATAENCODER_DICT__[name]

def aldataset(name):
    
    def register_function_fn(cls):
        if name in __DATASET_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, ALDataset):
            raise ValueError("Class %s is not a subclass of %s" % (cls, ALDataset))
        __DATASET_DICT__[name] = cls
        return cls

    return register_function_fn

def transform(name):
    
    def register_function_fn(cls):
        if name in __TRANSFORM_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Transform):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Transform))
        __TRANSFORM_DICT__[name] = cls
        return cls

    return register_function_fn

def pipeline(name):
    
    def register_function_fn(cls):
        if name in __PIPELINE_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Pipeline):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Pipeline))
        __PIPELINE_DICT__[name] = cls
        return cls

    return register_function_fn

def data_encoder(name):
    
    def register_function_fn(cls):
        if name in __DATAENCODER_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, DataEncoder):
            raise ValueError("Class %s is not a subclass of %s" % (cls, DataEncoder))
        __DATAENCODER_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.data.' + module_name)    
