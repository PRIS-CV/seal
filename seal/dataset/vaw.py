import numpy as np
from PIL import Image
import os.path as op
from tqdm import tqdm
import torch
from torchvision import datasets as datasets

from . import dataset
from .dataset import ALDataset
from seal.dataset.utils import *


@dataset("VAWInstanceLevelDataset")
class VAWInstanceLevelDataset(ALDataset):
    def __init__(self, cfg, image_path, anno_path, mode, transform=None, *args, **kwargs):
        super().__init__()
        self.image_path = image_path
        self.anno_path = anno_path
        self.transform = transform

        assert mode in ['train', 'test', 'val'], 'Dataset only supports train, test, val mode.'
        self.mode = mode
        
        self.annos = load_json(op.join(anno_path, f'{mode}.json'))
        self.attr2idx = load_json(op.join(anno_path, 'attribute_index.json'))
        self.idx2attr = list(self.attr2idx.keys())

        if op.exists(op.join(anno_path, 'object_index.json')):
            self.obj2idx = load_json(op.join(anno_path, 'object_index.json'))
        else:
            self.generate_object_index(anno_path)

        self.idx2obj = list(self.obj2idx.keys())
        self.return_obj_name = False
        
    def generate_object_index(self, anno_path):
        object_list = []
        print("Generating Object Index ..")        
        from tqdm import tqdm
        annos = load_json(op.join(anno_path, "train_part1.json"))
        annos += load_json(op.join(anno_path, "train_part2.json"))
        annos += load_json(op.join(anno_path, "val.json"))
        annos += load_json(op.join(anno_path, "test.json"))
        for anno in tqdm(annos):
            if anno['object_name'] not in object_list:
                object_list.append(anno['object_name'])

        self.obj2idx = {o: i for i, o in enumerate(object_list)}
        save_json(op.join(anno_path, 'object_index.json'), self.obj2idx)
    
    def see_object(self, see:bool):
        self.return_obj_name = see

    def get_object_num(self):
        return len(self.obj2idx)

    def get_attr_num(self):
        return len(self.attr2idx)
    
    def get_attr(self):
        return list(self.attr2idx.keys())

    def encode_attr(self, attr):
        return self.attr2idx[attr]

    def decode_attr(self, idx):
        return self.idx2attr[idx]
    
    def decode_obj(self, idx):
        return self.idx2obj[idx]

    def encode_obj(self, obj):
        return self.obj2idx[obj]

    def get_a_instance(self, idx=None):
        if idx:
            assert idx < len(self), f"NUM of SMAPLE {len(self)}"
            return self.annos[idx]
        else:
            return self.annos[np.random.randint(low=0, high=len(self))]

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        
        anno = self.annos[index]
        
        pos_attr = anno['positive_attributes']
        neg_attr = anno['negative_attributes']
        can_attr = anno['candidate_attributes'] if 'candidate_attributes' in anno else []

        object_name = anno['object_name']
        object_index = self.encode_obj(object_name)
        object_index = torch.tensor(object_index, dtype=torch.long)

        path = op.join(self.image_path, f"{anno['image_id']}.jpg")
        image = Image.open(path).convert('RGB')
        w, h = image.size
        
        try: 
            polygon = anno['instance_polygon'][0]
            polygon = [(int(x), int(y)) for (x, y) in polygon]
            mask = polygon_to_mask(w, h, polygon)
        except:
            mask = Image.new('L', (w, h), 1)
        
        bbox = anno['instance_bbox']
        bbox = xywh_to_xyxy(bbox)
        bbox = [list(map(int, bbox))]
        target = torch.zeros((len(self.idx2attr)), dtype=torch.float).fill_(2)
        
        for a in pos_attr:
            target[self.encode_attr(a)] = 1
        for a in neg_attr:
            target[self.encode_attr(a)] = 0
        for a in can_attr:
            if a in pos_attr or a in neg_attr:
                continue
            # assert(a not in pos_attr and a not in neg_attr)
            target[self.encode_attr(a)] = 3

        if self.transform is not None:
            img, bbox, mask = self.transform(image, bbox, mask)

        instance = {
            'i': img,
            'o': object_index,
            'm': mask,
            't': target,
            'b': bbox,
        }

        return instance


@dataset("VAWImageLevelDataset")
class VAWImageLevelDataset(VAWInstanceLevelDataset):
    
    def __init__(self, cfg, image_path, anno_path, mode, transform=None, *args, **kwargs):
        super().__init__(cfg, image_path, anno_path, mode, transform)
        self._trans_instances_to_image()
        

    def _trans_instances_to_image(self, store=True):
        anno_path = op.join(self.anno_path, f"{self.mode}_image_level.json")
        if op.exists(anno_path):
            self.annos = load_json(anno_path)
        else:
            print("Collecting Image Level Annotation From Instance Level Annotation ...")
            image_level_annos = {}
            
            for anno in tqdm(self.annos):
                
                instance = anno
                image_id = anno["image_id"]
                del instance["image_id"]
                
                try:
                    image_anno = image_level_annos[image_id]
                except:
                    image_anno = {
                        "image_id": image_id,
                        "bboxes": [], 
                        "objects": [], 
                        "polygons": [], 
                        "positive_attributes": [], 
                        "negative_attributes": []
                    }
                finally:
                    image_anno["bboxes"].append(instance["instance_bbox"])
                    image_anno["objects"].append(instance["object_name"])
                    image_anno["polygons"].append(instance["instance_polygon"])
                    image_anno["positive_attributes"].append(instance["positive_attributes"])
                    image_anno["negative_attributes"].append(instance["negative_attributes"])
                    image_level_annos[image_id] = image_anno
            
            self.annos = list(image_level_annos.values())
            if store:
                print(f"Saving Image Level Annotation to {anno_path} ...")
                save_json(anno_path, self.annos)
    
    def __getitem__(self, index):
        image_anno = self.annos[index]
        objects = []
        bboxes = []
        targets = []

        path = op.join(self.image_path, f"{image_anno['image_id']}.jpg")
        image = Image.open(path).convert('RGB')

        for i, o in enumerate(image_anno["objects"]):

            object_index = self.encode_obj(o)
            objects.append(object_index)

            pos_attr = image_anno['positive_attributes'][i]
            neg_attr = image_anno['negative_attributes'][i]
            target = torch.zeros((len(self.idx2attr)), dtype=torch.float).fill_(2)
            for a in pos_attr:
                target[self.encode_attr(a)] = 1
            for a in neg_attr:
                target[self.encode_attr(a)] = 0
            targets.append(target.unsqueeze(0))
            w, h = image.size

            bbox = image_anno['bboxes'][i]
            bbox = xywh_to_xyxy(bbox)                 # [x, y, w, h] -> [x1, y1, x2, y2]
            bbox = list(map(int, bbox))
            bboxes.append(bbox)

        if self.transform is not None:
            image, bboxes, _ = self.transform(image, bboxes, None)
        
        if self.return_obj_name:
            objects = image_anno["objects"]
        else:
            objects = torch.tensor(objects, dtype=torch.long)

        targets = torch.cat(targets, dim=0)
        
        image_data = {
            'i': image,
            'o': objects,
            'b': bboxes,
            't': targets
        }
        
        return image_data

    @staticmethod
    def collate_fn(batch):
        images = torch.cat([item['i'].unsqueeze(0) for item in batch], dim=0)
        bboxes = [item['b'] for item in batch]
        targets = torch.cat([item['t'] for item in batch], dim=0)
        objects = torch.cat([item['o'] for item in batch], dim=0)
            
        return {
            'i': images,
            'o': objects,
            'b': bboxes,
            't': targets
        }
