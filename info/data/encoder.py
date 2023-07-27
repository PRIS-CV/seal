import clip
import numpy as np
import os.path as op
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import typing as t

from info.data.data import DataEncoder
from info.data import data_encoder


@data_encoder("GloveEncoder")
class GloveEncoder(DataEncoder):

    embedding_files = {
        '50': 'glove.6B.50d.txt',
        '100': 'glove.6B.100d.txt',
        '200': 'glove.6B.200d.txt',
        '300': 'glove.6B.300d.txt',
    }
    
    def __init__(self, cfg, device) -> None:
        super().__init__(cfg)
        self.root = "/mnt/sdb/data/wangxinran/dataset/GLOVE"
        self.dim = 100
        self.file = op.join(self.root, self.embedding_files[str(self.dim)])
        self.unknown_words = []
        self.load_embedding()
        
    def __call__(self, text: t.List, f_embedding=None) -> Tensor:
        
        self.unknown_words = []
        text_embeddings = self.encode_list(text)
        print(f"Words using <unk> embbeding \n {self.unknown_words}")     
        if f_embedding is not None:
            torch.save(text_embeddings, f_embedding)
        return text_embeddings
    
    def load_embedding(self):

        vocab, embeddings = [], []
        
        with open(self.file, 'rt') as fi:
            full_content = fi.read().strip().split('\n')
        
        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
            vocab.append(i_word)
            embeddings.append(i_embeddings)

        self.vocab = ['<unk>']
        self.embeddings = np.array(embeddings)
        self.vocab.extend(vocab)
        unk_emb = np.mean(self.embeddings, axis=0, keepdims=True)
        self.embeddings = torch.from_numpy(np.vstack((unk_emb, self.embeddings))).float()

    def vectorize(self, word: str) -> torch.Tensor:
        index = self.vocab.index(word.lower())
        return self.embeddings[index]
    
    def encode_single(self, word: str) -> torch.Tensor:
        try:
            return self.vectorize(word)
        except ValueError:

            def encode_split(word_str: str, splitters: t.Collection[str]):
                word_str = [word_str]
                for splitter in splitters:
                    word_out = []
                    for w in word_str:
                        word_out.extend(w.split(splitter))
                    word_str = word_out
                representations = []
                for w in word_str:
                    try:
                        w = self.vectorize(w)
                    except ValueError:
                        self.unknown_words.append(w)
                        w = self.vectorize('<unk>')
                    finally:
                        representations.append(w)
                return sum(representations) / len(representations)

            return encode_split(word, [" ", "-", "'", ',', '"'])

    def encode_list(self, words: t.Sequence[str]) -> torch.Tensor:
        return torch.stack([self.encode_single(word) for word in words])


@data_encoder("BertEncoder")
class BERTEncoder(DataEncoder):
    
    def __init__(self, cfg, device=torch.device('cuda'), pretrained_weight:str=None):
        super().__init__(cfg)
        
        self.dir = dir
        self.device = device
        pretrained_weight = "/mnt/sdb/data/wangxinran/weight/pretraining/bert/bert-base-uncased"
        if pretrained_weight is not None:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_weight)
            self.model = BertModel.from_pretrained(pretrained_weight).embeddings
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()
        
    def __repr__(self) -> str:
        print(self.model.config)
        return super().__repr__()

    def __call__(self, text: t.List, f_embedding=None) -> torch.Tensor:
        r"""
            Encoder list of text to text embedding.

            Args:
                str_list (List[String]): 
                    The texts (e.g. objects, attributes) waited to be embedded.
                file (String):
                    The file path used to store embedding.
                store (Boolean):
                    Whether store the text embedding for next time using.
            
            Return:
                torch.Tensor: The text embedding.
        """
        

        embs = []
        with torch.no_grad():
            for s in tqdm(text):
                input = self.tokenizer(s, return_tensors="pt")
                emb = self.model(input_ids=input["input_ids"], token_type_ids=input["token_type_ids"])[:, 1:-1, : ].mean(dim=1)
                embs.append(emb)
        text_embeddings = torch.cat(embs, dim=0).cpu()
        
        if f_embedding is not None:
            torch.save(text_embeddings, f_embedding)
        return text_embeddings


@data_encoder("CLIPEncoder")
class CLIPEncoder(DataEncoder):
    
    def __init__(self, cfg, backbone='ViT-B/32', device=torch.device('cuda')):
        super().__init__(cfg)
        
        self.dim = 512
        self.device = device
        self.model, _ = clip.load(backbone, device)

    def __call__(self, text:t.List, f_embedding=None) -> torch.Tensor:
        self.model.eval()
        print("Obtaining CLIP Embeddings ...")
        tokens = torch.cat([clip.tokenize(t) for t in text]).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(tokens)
            text_embeddings = text_embeddings.cpu()
        if f_embedding is not None:
            torch.save(text_embeddings, f_embedding)
        return text_embeddings

@data_encoder("OneHotEncoder")
class OneHotEncoder(DataEncoder):
    
    def __init__(self, dim, device) -> None:
        self.dim = dim

    def __call__(self, text: t.List, f_embedding=None):
        self.dim = len(text)
        with torch.no_grad():
            text_embeddings = torch.eye(self.dim)
        if f_embedding is not None:
            torch.save(text_embeddings, f_embedding)
        return text_embeddings
