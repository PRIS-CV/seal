import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import typing as t
from PIL.Image import Image as PILImage

from info.data import transform
from info.data.data import Transform


@transform("Resize")
class Resize(Transform):
    
    def __init__(self, w, h) -> None:
        self.w = w
        self.h = h

    def __call__(self, image:PILImage, bboxes:t.List[int]=None, mask:PILImage=None):
        r"""Resize image and its all bounding boxes.
            
            Args:
                image (PILImage): 
                    The image.
                bboxes (List[List[int]]): 
                    The bounding boxes of the image whose format is [xmin, ymin, xmax, ymax].
                mask (PILImage):
                    The mask of object in image.
            
            Return:
                (PILImage): 
                    The resized image.
                (List[List[int]]): 
                    The resized bounding boxes which shape is [num of objects, 4].
                (PILImage): 
                    The mask of object in image.
        """
        
        o_w, o_h = image.size

        resized_image = TF.resize(image, (self.w, self.h))
        resized_bboxes = None
        resized_mask = None

        if mask is not None:
            resized_mask = TF.resize(mask, (self.w, self.h))

        if bboxes is not None:
            resized_bboxes = []
            
            for bbox in bboxes:

                xmin, ymin, xmax, ymax = bbox

                ratio_width = self.w / o_w
                ratio_height = self.h / o_h

                xmin = int(xmin * ratio_width)
                xmax = int(xmax * ratio_width)
                ymin = int(ymin * ratio_height)
                ymax = int(ymax * ratio_height)

                resized_bboxes.append([xmin, ymin, xmax, ymax])

        return resized_image, resized_bboxes, resized_mask


@transform("ToTensor")
class ToTensor(Transform):
    r"""Turn the image, bounding boxes, mask to torch.Tensor.
    """
    def __call__(self, image:PILImage, bboxes: t.List[t.List[int]], mask:PILImage):
        r"""Turn the image, bounding boxes, mask to torch.Tensor.
            Args:
                image (PILImage):
                    The input image.
                bboxes (t.List[t.List[int]]):
                    The bounding box of each object in the image.
                mask
                    The mask of object in the image.
        """
        image = TF.to_tensor(image)
        if mask is not None:
            mask = np.array(mask)
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        if bboxes is not None:
            bboxes = torch.tensor(bboxes, dtype=torch.float)
        return image, bboxes, mask


@transform("CropInstance")
class CropInstance(Transform):
    r"""Crop the instance by the bounding box (xmin, ymin, xmax, ymax).
    """
    def __call__(self, image:PILImage, bboxes:t.List[t.List[int]], mask:PILImage=None):
        assert len(bboxes) == 1, "The num of bounding box should be 1 in this transform"
        image = image.crop(bboxes[0])
        if mask is not None:
            mask = mask.crop(bboxes[0])

        return image, bboxes, mask


@transform("ExpandBox")
class ExpandBox(Transform):
    r"""Expand the bounding box by a certain ratio.
    """

    def __init__(self, expand_ratio:float=0.3) -> None:
        super().__init__()
        self.expand_ratio = expand_ratio
        
    def __call__(self, image:PILImage, bboxes:t.List[t.List[int]], mask:PILImage=None):
        
        expanded_bboxes = []

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            wmax, hmax = image.size
            w, h = xmax - xmin, ymax - ymin
            margin = int(min(w, h) * self.expand_ratio * 0.5)
            
            x1 = max(0, xmin - margin)
            y1 = max(0, ymin - margin)
            x2 = min(wmax, xmax + margin)
            y2 = min(hmax, ymax + margin)

        expanded_bboxes.append([x1, y1, x2, y2])

        return image, bboxes, mask


@transform("Normalize")
class Normalize(Transform):
    r"""Normalize the image by mean and std.
    """
    
    def __init__(self, mean:t.List[float]=[0.485, 0.456, 0.406], std:t.List[float]=[0.229, 0.224, 0.225]) -> None:
        self.mean = mean
        self.std = std
    
    def __call__(self, image, bboxes, mask):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, bboxes, mask


@transform("RandomHorizontalFlip")
class RandomHorizontalFlip(Transform):
    
    def __init__(self, prob=0.5) -> None:
        self.prob = prob

    def __call__(self, image:torch.Tensor, bboxes:torch.Tensor, mask:torch.Tensor):
        r"""Resize image and its all bounding boxes.
            
            Args:
                image (torch.Tensor): 
                    The image tensor.
                bboxes (torch.Tensor): 
                    The bounding boxes of the image whose format is [xmin, ymin, xmax, ymax].
            
            Return:
                (PIL.Image): The resized image.
                (torch.Tensor): The resized bounding boxes which shape is [num of objects, 4].
        """

        if random.random() < self.prob:
            
            _, width = image.shape[-2:]
            image = image.flip(-1)
            
            if bboxes is not None:
                bboxes[:, 0] = width - bboxes[:, 2]
                bboxes[:, 2] = width - bboxes[:, 0]
            if mask is not None:
                mask = mask.flip(-1)

        return image, bboxes, mask
