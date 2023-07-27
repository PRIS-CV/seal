import os.path as op
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
import typing as t

from .utils import ImageItem
from .utils import load_json


class LSADataset(Dataset):
    
    def __init__(self, anno_dir:str, img_dir:str, preprocess:T.Compose, split:str, del_vaw:bool=False, sample_ratio:float=0.8):
        
        assert split in ['train', 'val', 'test']
        assert 0 < sample_ratio < 1, "The value of sample ratio must be in (0, 1)"

        self.anno_dir = anno_dir
        self.img_dir = img_dir
        self.split = split
        self.preprocess = preprocess
        self.del_vaw = del_vaw
        self.sample_ratio = sample_ratio

        self.load_annotations()
        if del_vaw:
            self.del_vaw_test()
    
    def encode_attr(self, attr):
        return self.attr2idx[attr]

    def decode_attr(self, idx):
        return self.idx2attr[idx]
    
    def decode_obj(self, idx):
        return self.idx2obj[idx]

    def encode_obj(self, obj):
        return self.obj2idx[obj]

    def __getitem__(self, index:int) -> ImageItem:

        a = self.annos[index]
        objects = []
        bboxes = []
        attrs = []

        object_embs = self.encode_obj(objects)
        image = self.load_img(a['image_id'])
        w, h = image.size
        if self.preprocess is not None:
            image_tensor = self.preprocess(image)

        for o in a["objects"]:
            objects.append(o["object"])
            attrs.append(o["attributes"])
            xmin, ymin, xmax, ymax = list(map(lambda x: float(x), o["box"]))
            if 0 < xmin < 1:
                xmin = xmin * w
                ymin = ymin * h
                xmax = xmax * w
                ymax = ymax * h
            bboxes.append([xmin, ymin, xmax, ymax])

        if self.sample_obj < 1:
            num_of_objects = len(objects)
            sample_num = len(num_of_objects) * self.sample_ratio
            random.sample(range(num_of_objects), )
        
        attrs = self.encode_attr(attrs)

        return ImageItem(image_tensor=image_tensor, object_embs=object_embs, bboxes = bboxes, labels = attrs)

    def __len__(self):
        return len(self.annos)
    
    def __repr__(self) -> str:
        
        return "annotation in " + self.anno_dir + "\n" \
               "img in " + self.img_dir + "\n" \
               "split: " + self.split + "\n" \
               "delete vaw test part: " + self.del_vaw + "\n" \
               "sample obj: " + self.sample_obj


    def load_annotations(self) -> dict:
        file = op.join(self.anno_dir, self.split + '.json')
        self.annos = load_json(file)

    def del_vaw_test(self, vaw_test_list):
        for a in self.annos:
            if a['img'] in vaw_test_list:
                del a

    def load_img(self, img_path):
        return Image.open(img_path)
 


class LSACommonToCommon(LSADataset):
    
    def __init__(self, anno_dir: str, img_dir: str, preprocess: T.Compose, split: str):
        super().__init__(anno_dir, img_dir, preprocess, split)
        self.attr_split_dir = op.join(self.anno_dir, 'common2common')


class LSACommonToRare(LSADataset):
    
    def __init__(self, anno_dir: str, img_dir: str, preprocess: T.Compose, split: str):
        super().__init__(anno_dir, img_dir, preprocess, split)
        self.attr_split_dir = op.join(self.anno_dir, 'common2rare') 

    