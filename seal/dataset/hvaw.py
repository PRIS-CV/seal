import os.path as op

from . import dataset
from .dataset import ALDataset
from .utils import *


@dataset("HVAWInstanceLevelDataset")
class VAWInstanceLevelDataset(ALDataset):
    def __init__(self, image_path, anno_path, mode, transform=None, **kwargs):
        super().__init__()
        self.image_path = image_path
        self.anno_path = anno_path
        self.transform = transform

        assert mode in ['train', 'test', 'val'], 'Dataset only supports train, test, val mode.'
        self.mode = mode
        
        self.annos = load_json(op.join(anno_path, f'{mode}.json'))
        self.attr2idx = load_json(op.join(anno_path, 'attribute_index.json'))
        self.idx2attr = list(self.attr2idx.keys())

        self.load_relation(op.join(anno_path, 'hmat.npy'))
            
        self.obj2idx = load_json(op.join(anno_path, 'object_index.json'))
        self.idx2obj = list(self.obj2idx.keys())
    
    def load_relation(self, R):
        R = np.load(R)
        self.R = torch.from_numpy(R).float()

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

    def get_image_id(self, idx):
        return self.annos[idx]['image_id']

    def decode_targets(self, targets_tensor):
        pos_targets = []
        neg_targets = []

        for i in range(len(targets_tensor)):
            if targets_tensor[i] == 1:
                pos_targets.append(self.decode_attr(i))
            elif targets_tensor[i] == 0:
                neg_targets.append(self.decode_attr(i))
        
        return pos_targets, neg_targets


    def complete_target(self, t, compelete_positive=True, complete_negative=True):

        if complete_negative:
            neg = (t == 0).float().unsqueeze(0)
            n_neg = torch.matmul(neg, self.R).squeeze(0)
            t[n_neg > 0] = 0

        if compelete_positive:
            pos = (t == 1).float().unsqueeze(0)
            n_pos = torch.matmul(pos, self.R.t()).squeeze(0)
            t[n_pos > 0] = 1        

        return t

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = self.annos[index]
        pos_attr = anno['positive_attributes']
        neg_attr = anno['negative_attributes']
        
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
    
        target = self.complete_target(target)
                        
        if self.transform is not None:
            img, bbox, mask = self.transform(image, bbox, mask)
        else:
            img = image


        instance = {
            'i': img,
            'o': object_index,
            'm': mask,
            't': target,
            'b': bbox,
        }

        return instance


    