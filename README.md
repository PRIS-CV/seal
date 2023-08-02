# Seal
Semantic Enhanced Attribute Learning

## News
One paper accepted in ACM MM 2023: _Hierarchical Visual Attribute Learning in the Wild_. The relevant Code will be updated soon.

## Paper List
1. Vision-Language Assisted Attribute Learning (NIDC-2023)
2. Attribute Learning with Knowledge Enhanced Partial Annotations (ICIP-2023)

## Preparation

Clone the repo and install the info api lib:

```
git clone https://github.com/GriffinLiang/seal.git
cd /path/to/seal
pip install -e .
```

### Download VAW and VG Dataset
Download VAW dataset at https://github.com/adobe-research/vaw_dataset
Download VG v1.4 dataset at https://visualgenome.org/

### Download Word-embedding
You need to download relevant word embeddings at [Google Drive](https://drive.google.com/drive/folders/18M4F7vA0EOZqlp88E4W9gatQUTcSHYd6?usp=sharing), and put it under `data/embeddings/`.

### Clean The VAW Dataset
Use `clean.ipynb` in `notebook/` to clean the original VAW dataset to get cleaned VAW dataset.



### Run

Train
```
sh train.sh
```

Test
```
sh test.sh
```
