import pandas as pd
import h5py
import os.path as op
import numpy as np
from tqdm import tqdm

class CNNB:
    """
    A class use to load word embedding from ConceptNet https://github.com/commonsense/conceptnet-numberbatch
    support load from .h5 file or .txt file
    for each word it will return a numpy ndarray word embedding

    Example:
    1. load from .txt 
    cnnb = CNNB("/path/to/numberbatch-en-19.08.txt")
    cnnb("happy")

    cnnb = CNNB("/path/to/mini.h5")
    cnnb("happy")
    """

    def __init__(self, word_embedding_path) -> None:
        self.word_embedding_path = word_embedding_path
        self.file_name = op.basename(self.word_embedding_path)
        assert self.file_name.endswith(".txt") or self.file_name.endswith(".h5"), "The embedding file must be type of .txt or .h5"
        self.load_embedding()

    def load_embedding(self):
        self.url2emb = {}
        print(f"Load Word Embedding File {self.word_embedding_path}")
        if self.file_name.endswith(".h5"):
            data = pd.read_hdf(self.word_embedding_path, key='mat')
            lines = self._filter_lines(data.index.values)
            for url in tqdm(lines):
                self.url2emb[url] = np.array(data.loc[url].tolist())
            self.we_num, self.we_dim = len(self.url2emb), list(self.url2emb.values())[0].shape[0]
            
        else:
            
            with open(self.word_embedding_path, 'r') as wef:
                lines = wef.readlines()
            meta = lines.pop(0).strip('\n').split(' ')
            self.we_num, self.we_dim = int(meta[0]), int(meta[1])

            for line in tqdm(lines):
                line = line.strip('\n').split(" ")
                url, emb = line[0], list(map(float, line[1:]))
                self.url2emb[url] = np.array(emb)

        print(f"Finish Loading, word embedding num: {self.we_num}, dimension: {self.we_dim}")

    def _filter_lines(self, lines):
        new_lines = []
        for line in lines:
            lang = line.split("/")[2]
            if lang == "en":
                new_lines.append(line)            

        return new_lines

    def encode_attr(self, attr, language):
        attr_url = self._standardized_concept_uri(language, attr)
        return self.url2emb[attr_url]

    def __call__(self, attr, language="en") -> np.ndarray:
        return self.encode_attr(attr, language)
        
    def _standardized_concept_uri(self, language, term):
        term = term.replace('_', ' ')
        if self.file_name.endswith(".h5"):
            return '/c/{}/{}'.format(language, term)
        return term

    def __len__(self):
        return self.we_num
    
    def dim(self):
        return self.we_dim