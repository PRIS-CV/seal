from yacs.config import CfgNode as CN


class VAConfig:
    
    def __init__(self):
        
        self.cfg = CN()
        self.cfg.project = "Visual Attributes Recoginition"
        self.cfg.exp_name = "VAR"

        self.cfg.amp = True
        self.cfg.batch_size = 64
        self.cfg.epochs = 12
        self.cfg.seed = 99
        self.cfg.dataset = "VAWInstanceLevelDataset"
        self.cfg.weight = "/data/liangkongming/code/KGVA/exp/nidc"

        self.cfg.model = "RN50_BCE"
        self.cfg.model_config_name = ""
        self.cfg.model_config = ""

        self.cfg.num_workers = 8
        self.cfg.resize_size = 256
        self.cfg.input_size = 224
        self.cfg.pipeline = "VAWInstancePipeline"
        self.cfg.encoder = "OneHotEncoder"
        self.cfg.eval_util = "vaw_eval_util"
        self.cfg.train_util = "train_one_epoch"
        self.cfg.num_classes = 620

        self.wandb = False

    def get_cfg(self):
        return self.cfg.clone()

    def load(self,config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
    
class ConfigPlaceHolder:
    
    def __init__(self) -> None:
        pass
    
    