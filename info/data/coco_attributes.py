import pickle
import numpy as np
from PIL import Image
import os.path as op
import torch
from torchvision import datasets as datasets
from pycocotools.coco import COCO
from IPython import embed

from info.data import aldataset
from info.data.data import ALDataset
from info.data.utils import *


@aldataset("COCOAttributesInstanceLevelDataset")
class COCOAttributesInstanceLevelDataset(ALDataset):

    def __init__(self, cfg, image_path, anno_path, mode, transform=None) -> None:
        super().__init__(cfg)

        if mode == 'test':
            mode = 'val'
        self.transform = transform
        self.f_attr_anno = '/mnt/sdb/data/wangxinran/dataset/COCO_ATTRIBUTES/cocottributes/data/cocottributes_eccv_version.pkl'
        self.f_coco_anno = f'/mnt/sdb/data/wangxinran/dataset/COCO2014/annotations/instances_{mode}2014.json'
        self.d_coco_image = f'/mnt/sdb/data/wangxinran/dataset/COCO2014/{mode}2014/'
        self.coco = COCO(self.f_coco_anno)
        self.load_attr_anno()
        
        self.data = []
        self.split = mode + '2014'
        
        for patch_id, _ in self.attr_anno['ann_vecs'].items():
            if self.attr_anno['split'][patch_id] == self.split:
                self.data.append(patch_id)

        self.attributes = sorted(self.attr_anno['attributes'], key=lambda x: x['id'])

        self.obj2idx = {item['name']: i for i, item in enumerate(self.coco.cats.values())}
        self.attr2idx = {item['name']: i for i, item in enumerate(self.attributes)}

    def load_attr_anno(self):
        with open(self.f_attr_anno, 'rb') as f:
            self.attr_anno = pickle.load(f, encoding='latin1')

    def encode_attr(self, attr):
        return self.attr2idx[attr]

    def decode_attr(self, idx):
        return self.idx2attr[idx]
    
    def decode_obj(self, idx):
        return self.idx2obj[idx]

    def encode_obj(self, obj):
        return self.obj2idx[obj]

    def __getitem__(self, index):
        
        patch_id = self.data[index]

        pos_attrs = self.attr_anno['ann_vecs'][patch_id]
        pos_attrs = torch.from_numpy(pos_attrs).float()
        target = torch.zeros_like(pos_attrs).fill_(2.).float()
        target[pos_attrs > 0] = 1.

        ann_id = self.attr_anno['patch_id_to_ann_id'][patch_id]
        # coco.loadImgs returns a list
        anno = self.coco.loadAnns(ann_id)[0]
        image = self.coco.loadImgs(anno['image_id'])[0]
        obj_name = self.coco.loadCats(anno['category_id'])[0]['name']
        obj_index = self.encode_obj(obj_name)
        
        bbox = anno["bbox"]
        x, y, w, h = list(map(int, bbox))
        bbox = xywh_to_xyxy(bbox)
        bbox = [list(map(int, bbox))]
        
        try: 
            polygon = anno['segmentation'][0]
            polygon = [(int(polygon[i]), int(polygon[i+1])) for i in range(0, len(polygon), 2)]
            mask = polygon_to_mask(w, h, polygon)
        except:
            mask = Image.new('L', (w, h), 1)

        img = Image.open(op.join(self.d_coco_image, image["file_name"])).convert("RGB")

        

        if self.transform is not None:
            img, bbox, mask = self.transform(img, bbox, mask)
        
        # if img.shape[0] == 1:
        #     print(ann_id)

        instance = {
            'i': img,
            'o': obj_index,
            'm': mask, 
            't': target,
            'b': bbox,
            'h_0': target,
        }

        return instance
    

    def __len__(self):
        return len(self.data)