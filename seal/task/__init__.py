import os
import importlib

from .task import BaseTask

__TASK_DICT__ = {}


def build_task(task_name: str) -> BaseTask:
    if task_name not in __TASK_DICT__:
        raise ValueError("Task type %s not registered!" % task_name)
    return __TASK_DICT__[task_name]

def task(name):
    
    def register_function_fn(cls):
        if name in __TASK_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseTask):
            raise ValueError("Class %s is not a subclass of %s" % (cls, BaseTask))
        __TASK_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('seal.task.' + module_name)    
