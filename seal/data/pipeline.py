from seal.data.data import Pipeline
from seal.data import build_transform, pipeline

import typing as t
from torch import Tensor


@pipeline("GSLInstancePipeline")
class GSLInstancePipeline(Pipeline):
    
    def __init__(self, mode, expand_ratio, input_size, **kwargs):
        
        if mode == "train":
            transforms = {
                "ExpandBox": {"expand_ratio": expand_ratio}, 
                "CropInstance": {}, 
                "Resize": {"w": input_size, "h": input_size}, 
                "ToTensor": {},
                "RandomHorizontalFlip": {}
            }
        else:
            transforms = {
                "ExpandBox": {"expand_ratio": expand_ratio}, 
                "CropInstance": {},
                "Resize": {"w": input_size, "h": input_size}, 
                "ToTensor": {}
            }
        
        transforms = [build_transform(name)(**args) for name, args in transforms.items()]
        super().__init__(transforms)
    
    def __call__(self, image, bboxes, mask) -> t.Tuple[Tensor, Tensor, Tensor]:
        return super().__call__(image, bboxes, mask)


@pipeline("VAWInstancePipeline")
class VAWInstancePipeline(Pipeline):

    name: str = "VAWInstancePipeline"
    mode_choice: list = ["train", "evalu", "visua"]

    def __init__(self, mode, expand_ratio, input_size, **kwargs):
        
        if mode == "train":
            transforms = {
                "ExpandBox": {"expand_ratio": expand_ratio}, 
                "CropInstance": {}, 
                "Resize": {"w": input_size, "h": input_size}, 
                "ToTensor": {},
                "Normalize": {},
                "RandomHorizontalFlip": {}
            }
        elif mode == "evalu":
            transforms = {
                "ExpandBox": {"expand_ratio": expand_ratio}, 
                "CropInstance": {},
                "Resize": {"w": input_size, "h": input_size}, 
                "ToTensor": {},
                "Normalize": {},
            }
        elif mode == "visua":
            transforms = {
                "ExpandBox": {"expand_ratio": expand_ratio}, 
                "CropInstance": {},
            }
        
        else:
            raise ValueError(f"{self.name} only supports mode: {self.mode_choice}.")

        transforms = [build_transform(name)(**args) for name, args in transforms.items()]
        super().__init__(transforms)
    
    def __call__(self, image, bboxes, mask) -> t.Tuple[Tensor, Tensor, Tensor]:
        return super().__call__(image, bboxes, mask)
    

@pipeline("OAGroupInstancePipeline")
class OAGroupInstancePipeline(Pipeline):

    name: str = "OAGroupInstancePipeline"
    mode_choice: list = ["train", "evalu", "visua"]

    def __init__(self, mode, input_size, **kwargs):
        
        if mode == "train":
            transforms = {
                "Resize": {"w": input_size, "h": input_size}, 
                "ToTensor": {},
                "Normalize": {},
                "RandomHorizontalFlip": {}
            }
        elif mode == "evalu":
            transforms = {
                "Resize": {"w": input_size, "h": input_size}, 
                "ToTensor": {},
                "Normalize": {},
            }
        elif mode == "visua":
            transforms = {
            }
        
        else:
            raise ValueError(f"{self.name} only supports mode: {self.mode_choice}.")

        transforms = [build_transform(name)(**args) for name, args in transforms.items()]
        super().__init__(transforms)
    
    def __call__(self, image, bboxes, mask) -> t.Tuple[Tensor, Tensor, Tensor]:
        return super().__call__(image, bboxes, mask)


@pipeline("VAWImagePipeline")
class VAWImagePipeline(Pipeline):
    
    def __init__(self, cfg, mode):

        if mode == "train":
            transforms = {
                "Resize": {"w": cfg.input_size, "h": cfg.input_size}, 
                "ToTensor": {},
                "Normalize": {}, 
                "RandomHorizontalFlip": {}
            }
        else:
            transforms = {
                "Resize": {"w": cfg.input_size, "h": cfg.input_size}, 
                "ToTensor": {},
                "Normalize": {},
            }

        transforms = [build_transform(name)(**args) for name, args in transforms.items()]
        super().__init__(transforms)
    
    def __call__(self, image, bboxes, mask) -> t.Tuple[Tensor, Tensor, Tensor]:
        return super().__call__(image, bboxes, mask)
