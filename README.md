# SEAL: Semantic Enhanced Attribute Learning
SEAL is a PyTorch-based attribute learning package designed to facilitate the development and evaluation of attribute learning models. SEAL is designed to offer a flexible and modular framework for building attribute learning models. It leverages semantic information and uses state-of-the-art techniques to enhance the accuracy and interpretability of the learned attributes.

## News 🚀

*August 24, 2023*: One paper accepted in ACM MM 2023: **Hierarchical Visual Attribute Learning in the Wild**. The relevant code is now available. Please see project [osarn](projects/osarn/README.md)

*August 18, 2023*: Add HVAW dataset in `seal/dataset/hvaw.py`. Add new evaluation metric `CV`, `CmAP` and update the evaluation system.


## Plan 📋
*Distributed Mode* We will soon update distributed training and inference.


## Attribute Learning Models

| Model Name                                | Project                                                                 | Status    |
|-------------------------------------------|-----------------------------------------------------------------------|:---------:|    
| Vision-language Guided Selective Loss     | [Vision-Language Assisted Attribute Learning](projects/gsl/README.md)                       | ✅          |
| Knowledge Enhanced Selective Loss         | [Attribute Learning with Knowledge Enhanced Partial Annotations](projects/kesl/README.md)    | 🏗️           |
| Object-specific Attribute Relation Net    | [Hierarchical Visual Attribute Learning in the Wild](projects/osarn/README.md)                | ✅         |

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

A brief [architecture overview](seal/README.md) assists users in quickly grasping the structure of SEAL.


🏗️


## Running

Before running you should check the modular json settings in a project's directory, e.g., `projects/gsl` and see the running instruction in each project's README file:

```bash

python main.py --project projects/gsl --mode train

python main.py --project projects/gsl --mode test

```

## Citation

If you use SEAL in your research or project, please consider citing the relevant papers.

---
