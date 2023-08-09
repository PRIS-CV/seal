import os
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ..configuration import build_config, BaseConfig


class BaseTask(object):

    name: str = "Base Task"
    
    def __init__(self, d_config: str) -> None:
        
        self.d_config = d_config
        self._task_config = {}
        self.initialize_configs()
        self.initialize_device(self.task_settings.get_settings()["device"])
        self.prepare()
        
    
    def initialize_device(self, device):
        
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    def initialize_configs(self):
       
        for file in os.listdir(self.d_config):
            if file.endswith('.json'):
                with open(os.path.join(self.d_config, file), "r") as f:
                    content = json.load(f)
                config_type = content["type"]
                config = build_config(config_type)(content["name"], content["settings"])
                self._task_config[config_type] = config

        self.task_settings = self.get_settings("task config")
        self.pipeline_setting = self.get_settings("pipeline config")
        self.dataset_setting = self.get_settings("dataset config")
        self.model_setting = self.get_settings("model config")
        self.train_settings = self.get_settings("train config")
        self.eval_settings = self.get_settings("eval config")

    def setup_data_parallel(rank, world_size):
        # Set up distributed data parallel (DDP) training
        # Initialize the process group
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

        # Set the device to the current GPU
        torch.cuda.set_device(rank)

    def cleanup_data_parallel():
        # Clean up distributed data parallel (DDP) training
        dist.destroy_process_group()

    def get_settings(self, config_type) -> BaseConfig:
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
