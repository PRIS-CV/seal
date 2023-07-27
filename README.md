# seal
Semantic Enhanced Attribute Learning

This is the code repo for the paper:
1. Vision-Language Assisted Attribute Learning
2. Attribute Learning with Knowledge Enhanced Partial Annotations



## Preparation

Clone the repo and install the info api lib:

```
git clone https://github.com/GriffinLiang/seal.git
cd /path/to/seal
pip install -e .
```

You need to download relevant preparation data (word embeddings and ) at [Google Drive](), and put it under `data/`

### Download VAW and VG Dataset
Download original VAW dataset at 
Download VG dataset at 

### Download Word-embedding


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
