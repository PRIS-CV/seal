import os
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..configuration import build_config, BaseConfig, TASK_CONFIG_TYPE


class BaseTask(object):

    name: str = "Base Task"
    project: str = "Example Project"
    
    def __init__(self, d_config: str, mode: str) -> None:
        
        self.d_config = d_config
        self.mode = mode
        self._task_config = {}
        self.initialize_configs()
        self.initialize_device(self.task_settings.get_settings()["device"])
        self.prepare()
        self.distribute_type = ""
        
    def initialize_device(self, device):
        
        self.device = torch.device(f"{device}" if torch.cuda.is_available() else "cpu")

    def initialize_configs(self):
       
        for file in os.listdir(self.d_config):
            if file.endswith('.json'):
                with open(os.path.join(self.d_config, file), "r") as f:
                    content = json.load(f)
                config_type = content["type"]
                if config_type == TASK_CONFIG_TYPE:
                    self.project = content["project"]
                config = build_config(config_type)(content["name"], content["settings"])
                self._task_config[config_type] = config

        self.task_settings = self.get_config("task config")
        self.pipeline_setting = self.get_config("pipeline config")
        self.dataset_setting = self.get_config("dataset config")
        self.model_setting = self.get_config("model config")
        self.train_settings = self.get_config("train config")
        self.eval_settings = self.get_config("eval config")



    def get_config(self, config_type) -> BaseConfig:
        return self._task_config[config_type]
                    
    def save_configs(self):
        for config in self._task_config.values():
            config.save_to_json(os.path.join(self.d_config, config.name + ".json"))
    
    def load_configs(self):
        for config in self._task_config.values():
            config.load_from_json(os.path.join(self.d_config, config.name + ".json"))
    
    def __repr__(self) -> str:
        content = f"\n{self.name}:\n"
        for config in self._task_config.values():
            content += f"{config.__repr__}"
        return content

    def prepare(self):
        pass
    
    def run(self):
        raise NotImplementedError("This method should be implemented in the subclass!")
