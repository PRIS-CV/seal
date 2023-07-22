from yacs.config import CfgNode as CN


class ALModelConfig:
    
    r"""Attribute Learning Model Config.
    """

    def __init__(self, name="Attribute Learning Model Config"):
        
        self.cfg = CN()
        self.cfg.name = name

    def get_cfg(self):
        return self.cfg.clone()

    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
    