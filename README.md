# SEAL: Semantic Enhanced Attribute Learning
SEAL is a PyTorch-based attribute learning package designed to facilitate the development and evaluation of attribute learning models. SEAL is designed to offer a flexible and modular framework for building attribute learning models. It leverages semantic information and uses state-of-the-art techniques to enhance the accuracy and interpretability of the learned attributes.

## News 🚀
*August 4, 2023*: One paper accepted in ACM MM 2023: **Hierarchical Visual Attribute Learning in the Wild**. The relevant code will be updated soon.


## Attribute Learning Models

| Model Name                                | Project                                                                 | Status    |
|-------------------------------------------|-----------------------------------------------------------------------|-----------|    
| Vision-language Guided Selective Loss     | [Vision-Language Assisted Attribute Learning](projects/gsl/README.md)                       |           |
| Knowledge Enhanced Selective Loss         | [Attribute Learning with Knowledge Enhanced Partial Annotations](projects/kesl/README.md)    |           |
| Object-specific Attribute Relation Net    | [Hierarchical Visual Attribute Learning in the Wild](projects/osarn/README.md)                |           |

## Installation 

Here's how you can get started with SEAL:

1. Clone the repo and install:

```
git clone https://github.com/PRIS-CV/seal.git
cd /path/to/seal
pip install -e .
```


2. Import the models and start building attribute learning pipelines.


## Docs and Tutorial 📚



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

## Citation

If you use SEAL in your research or project, please consider citing the relevant papers.

---
