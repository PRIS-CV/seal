import os
import json


from ..configuration import build_config


class BaseTask(object):
    
    def __init__(self, d_config: str) -> None:
        
        self.d_config = d_config
        self.task_config = {}
        self.initialize_configs()

    def initialize_configs(self):
       
        for file in os.listdir(self.d_config):
            if file.endswith('.json'):
                with open(os.path.join(self.d_config, file), "r") as f:
                    content = json.load(f)
                config_type = content["type"]
                config = build_config(config_type)(content["name"], content["settings"])
                self.task_config[content["name"]] = config
                    
    def save_configs(self):
        for config in self.task_config.values():
            config.save_to_json(os.path.join(self.d_config, config.name + ".json"))
    
    def load_configs(self):
        for config in self.task_config.values():
            config.load_from_json(os.path.join(self.d_config, config.name + ".json"))
    
    
    def __repr__(self) -> str:
        content = f"\n{self.__class__.__name__}:\n"
        for config in self.task_config.values():
            content += f"{config.__repr__}"
        return
    
    def run(self):
        raise NotImplementedError("This method should be implemented in the subclass!")
