import json


BASE_CONFIG_TYPE = "base config"
DATASET_CONFIG_TYPE = "dataset config"
MODEL_CONFIG_TYPE = "model config"
EVAL_CONFIG_TYPE = "eval config"
TRAIN_CONFIG_TYPE = "train config"
PIPELINE_CONFIG_TYPE = "pipeline config"


class _BaseConfig(object):
    
    type: str = BASE_CONFIG_TYPE
    
    def __init__(self, name:str, settings:dict, *args, **kwargs) -> None:

        self.name: str = name
        self._settings: dict = settings
    
    def get_settings(self):
        return self._settings

    def set_settings(self, **kwargs):
        self._settings.update(**kwargs)

    def __repr__(self) -> str:
        return f"\n{self.name}({self.type}):\n{self._settings}"

    def save_to_json(self, path):
        with open(path, "w") as f:
            save_content = {
                "type": self.type,
                "name": self.name,
                "settings": self._settings
            }
            json.dump(save_content, f, indent=4)

    def load_from_json(self, path):
        with open(path, "r") as f:
            load_content = json.load(f)
            assert self.type == load_content["type"], f"Configuration type mismatch: {self.type} != {load_content['type']}"
            self.name = load_content["name"]
            self._settings = load_content["settings"]
    

class DatasetConfig(_BaseConfig):
    
    type: str = DATASET_CONFIG_TYPE
    

class ModelConfig(_BaseConfig):
        
    type: str = MODEL_CONFIG_TYPE


class EvalConfig(_BaseConfig):
            
    type: str = EVAL_CONFIG_TYPE


class TrainConfig(_BaseConfig):
                    
    type: str = TRAIN_CONFIG_TYPE              


class PipelineConfig(_BaseConfig):

    type: str = PIPELINE_CONFIG_TYPE


