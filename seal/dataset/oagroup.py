import os
import os.path as op
import numpy as np
import logging
from PIL import Image
import torch
from torchvision import datasets as datasets
from tqdm import tqdm
from typing import Any


from . import dataset
from .dataset import ALDataset
from .utils import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataset("ObjectAttributeGroupDataset")
class ObjectAttributeGroup(ALDataset):
    
    def __init__(self, image_path, anno_path, mode, transform, **kwargs) -> None:
        super().__init__()

        self.image_path = image_path
        self.anno_path = anno_path
        self.mode = mode
        self.transform = transform

        self.init_combinations()

    def init_combinations(self):
        
        self.objects_images = {}
        self.objects = os.listdir(self.image_path)                             
        self._current_group = self.objects[0]                                   # Default group is the first group
        self.objects_group = {}                                                 # Group name
        self.group_indexes = {}                                                  # Group id and group name mapping 
        self.objects_group_id = {}                                              # Group id for each object

        for o in self.objects:
            attributes = os.listdir(op.join(self.image_path, o))
            self.objects_group[o] = attributes
            self.objects_images[o] = []
            self.objects_group_id[o] = []
            self.group_indexes[o] = {}
            for i, a in enumerate(attributes):
                images = os.listdir(op.join(self.image_path, o, a))
                images = [op.join(self.image_path, o, a, i) for i in images]
                self.objects_images[o].extend(images)
                self.objects_group_id[o].extend([i] * len(images))
                self.group_indexes[o][a] = i
            logger.info(f"Initialize object {o} with attributes {attributes}")        

    @property
    def groups(self):
        return self.objects
    
    def change_group(self, group_name):
        self._current_group = group_name

    def get_group_size(self, group_name):
        return len(self.objects_images[group_name])

    def __getitem__(self, index: Any) -> Any:
        image_path = self.objects_images[self._current_group][index]
        group_id = self.objects_group_id[self._current_group][index]  
        
        image = Image.open(image_path).convert("RGB")
        group_id = torch.tensor(group_id, dtype=torch.long) 
        
        if self.transform is not None:
            image, _, _ = self.transform(image, None, None)

        return {
            'i': image,
            'group_id': group_id 
        }
    
    def get_image_by_index(self, g, index, transform=None) -> Image.Image:
            
        image_path = self.objects_images[g][index]
        image = Image.open(image_path).convert("RGB")
        if transform is not None:
            image, _, _ = transform(image, None, None)

        return image

    def __len__(self):
        return self.get_group_size(self._current_group)

