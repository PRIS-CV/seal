import json


BASE_CONFIG_TYPE = "base config"
DATASET_CONFIG_TYPE = "dataset config"
MODEL_CONFIG_TYPE = "model config"
EVAL_CONFIG_TYPE = "eval config"
TRAIN_CONFIG_TYPE = "train config"
PIPELINE_CONFIG_TYPE = "pipeline config"
TASK_CONFIG_TYPE = "task config"

class BaseConfig(object):
    
    type: str = BASE_CONFIG_TYPE
    
    def __init__(self, name:str, settings:dict, *args, **kwargs) -> None:

        self.name: str = name
        self._settings: dict = settings
    
    def get_settings(self):
        return self._settings

    def set_settings(self, **kwargs):
        self._settings.update(**kwargs)

    def __repr__(self) -> str:
        return f"\n({self.type}):\n{self._settings}"

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
    

class DatasetConfig(BaseConfig):
    
    type: str = DATASET_CONFIG_TYPE
    

class ModelConfig(BaseConfig):
        
    type: str = MODEL_CONFIG_TYPE


class EvalConfig(BaseConfig):
            
    type: str = EVAL_CONFIG_TYPE


class TrainConfig(BaseConfig):
                    
    type: str = TRAIN_CONFIG_TYPE              


class PipelineConfig(BaseConfig):

    type: str = PIPELINE_CONFIG_TYPE


class TaskConfig(BaseConfig):
    type: str = TASK_CONFIG_TYPE


CONFIG_TYPES = {
    BASE_CONFIG_TYPE: BaseConfig,
    DATASET_CONFIG_TYPE: DatasetConfig,
    MODEL_CONFIG_TYPE: ModelConfig,
    EVAL_CONFIG_TYPE: EvalConfig,
    TRAIN_CONFIG_TYPE: TrainConfig,
    PIPELINE_CONFIG_TYPE: PipelineConfig,
    TASK_CONFIG_TYPE: TaskConfig
}


def build_config(config_type: str) -> BaseConfig:
    if config_type not in CONFIG_TYPES:
        raise ValueError(f"Unknown config type: {config_type}")
    return CONFIG_TYPES[config_type]
