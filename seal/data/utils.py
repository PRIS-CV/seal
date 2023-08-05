from dataclasses import dataclass
import numpy as np
import PIL
from PIL import Image, ImageDraw
import json
import typing as t
import torch
import torch.nn as nn


def bbox_expand(bbox:t.Sequence[int], image_size:t.Sequence[int], expand_ratio:float) -> t.Sequence[int]:
    r"""Expand the area of bounding box of object to gain more context.

        Args: 
            bbox (t.Sequence[int]): 
                The coordinates of bounding box [xmin, ymin, xmax, ymax].
            image_size (t.Sequence[int]):
                The width and height of image.
            expand_ratio (float):
                The expand ratio of bounding box.
        Return:
            t.Sequence[int, int, int, int]: The coordinates of expanded bounding box [xmin, ymin, xmax, ymax].
    """
    
    xmin, ymin, xmax, ymax = bbox
    wmax, hmax = image_size
    w, h = xmax - xmin, ymax - ymin
    margin = min(w, h) * expand_ratio * 0.5
    
    x1 = max(0, xmin - margin)
    y1 = max(0, ymin - margin)
    x2 = min(wmax, xmax + margin)
    y2 = min(hmax, ymax + margin)

    return x1, y1, x2, y2

def convert_to_relative(bbox:t.Sequence[int], image_size:t.Sequence[int]) -> t.Sequence[int]:
    x, y, w, h = bbox
    w_img, h_img = image_size
    return x / w_img, y / h_img, w / w_img, h / h_img

def xywh_to_xyxy(bbox:t.Sequence[int]) -> t.Sequence[int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h

def resize_bbox(bbox, ratios):
        r"""Resize the bounding box.
            Args:
                bbox (List[float, float, float, float]): 
                    The bounding box which format is [x1, y1, x2, y2].
                ratios (Tuple(float, float)):
                    (w_ratio, h_ratio) $w_ratio = origin_width / target_width$, $h_ratio = origin_height / target_height$
            Return:
                bbox (List[float, float, float, float]): 
                   The resized bounding box which format is [x1, y1, x2, y2].
        """

        xmin, ymin, xmax, ymax = bbox
        ratios_width, ratios_height = ratios
        
        xmin *= ratios_width
        xmax *= ratios_width
        ymin *= ratios_height
        ymax *= ratios_height

        return [xmin, ymin, xmax, ymax]

def polygon_to_mask(w, h, polygon):
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    return mask

def load_json(file:str) -> t.Dict:
    r"""Load json data from file and convert it to dict.
        
        Args: 
            file (str): The path of json file.
        
        Return:
            t.Dict: The json data in a dict format.

    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def save_json(file:str, data:t.Dict, indent=None):
    r"""Load json data from file and convert it to dict.
        
        Args: 
            file (str): The path of json file.
        
        Return:
            t.Dict: The json data in a dict format.

    """
    
    with open(file, 'w') as f:
        json.dump(data, f, indent=indent)

def get_stat(data:t.Dict):
    cnt_attr = {} # Number of attribute occurrences
    cnt_obj = {} # Number of object occurrences
    cnt_pair = {} # Number of pair occurrences
    cooc = {} # co-occurrences of attributes (cooc['red']['blue'] is the number of times red and blue appear together)

    obj_afford = {}
    obj_afford_cooc = {}
    
    n_images = 0

    for ins in data:
        o = ins['object_name']
        box = ins['instance_bbox']
            
        n_images += 1

        if o not in cnt_obj:
            cnt_obj[o] = 0
            obj_afford[o] = {}
            obj_afford_cooc[o] = {}

        cnt_obj[o] += 1

        for a in set(ins['positive_attributes']): # possible duplicates so we use set
            if a not in cnt_attr:
                cnt_attr[a] = 0
                cooc[a] = {}
            cnt_attr[a] += 1

            pair = (a, o)
            if pair not in cnt_pair:
                cnt_pair[pair] = 0
            cnt_pair[pair] += 1

            if a not in obj_afford[o]:
                obj_afford[o][a] = 0
                obj_afford_cooc[o][a] = {}
            obj_afford[o][a] += 1

            for other_a in set(ins['positive_attributes']):
                if a != other_a:
                    if other_a not in cooc[a]:
                        cooc[a][other_a] = 0
                    cooc[a][other_a] += 1

                    if other_a not in obj_afford_cooc[o][a]:
                        obj_afford_cooc[o][a][other_a] = 0
                    obj_afford_cooc[o][a][other_a] += 1
                    
    return cnt_attr, cnt_obj, cnt_pair, cooc, obj_afford, obj_afford_cooc, n_images



class UnNormalize(nn.Module):
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        
        self.mean = torch.tensor(mean).view((-1, 1, 1))
        self.std = torch.tensor(std).view((-1, 1, 1))

    def __call__(self,x):
        
        x = (x * self.std) + self.mean
        return torch.clip(x, 0, None)


@dataclass
class InstanceItem:
    instance: torch.Tensor                          # [1, 3, input_size, input_size]
    label: torch.Tensor = None                      # [1, output_size]
    bbox: torch.Tensor = None                       # [1, 4]


@dataclass
class BatchofInstances:
    instances: torch.Tensor                         # [batch_size, 3, input_size, input_size]
    labels: torch.Tensor = None                     # [batch_size, output_size]
    bboxes: torch.Tensor = None                     # [batch_size, 4]

    @staticmethod
    def collate(items: t.Sequence[InstanceItem]):
        instances = []
        labels = []
        bboxes = []

        for item in items:
            instances.append(item.instance)
            labels.append(item.label)
            bboxes.append(item.bbox)

        instances = torch.cat(instances, dim=0)
        labels = torch.cat(labels, dim=0)
        bboxes = torch.cat(bboxes, dim=0)

        return BatchofInstances(
            instances=instances, 
            labels=labels,
            bboxes=bboxes
        )


@dataclass
class ImageItem:
    image_tensor: torch.Tensor                      # [1, 3, input_size, input_size]
    object_emb: t.List[torch.Tensor]                # [num of objects, emb_size]
    bboxes: torch.Tensor                            # [num of objects, 4]
    labels: torch.Tensor                            # [num of objects, output_size]


@dataclass
class BatchofImages:
    image_tensor: torch.Tensor                      # [batch_size, 3, input_size, input_size]
    object_emb: t.List[torch.Tensor]                # [num of objects in one batch, emb_size]
    labels: torch.Tensor = None                     # [num of objects in one batch, output_size]
    bboxes: torch.Tensor = None                     # [num of objects in one batch, 4]
