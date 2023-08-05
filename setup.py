from setuptools import setup, find_packages



setup(
    name='seal',
    version='0.1.0',
    description='SEAL is a PyTorch-based attribute learning package designed to facilitate the development and evaluation of attribute learning models. SEAL is designed to offer a flexible and modular framework for building attribute learning models. It leverages semantic information and uses state-of-the-art techniques to enhance the accuracy and interpretability of the learned attributes.',
    author='Xinran Wang, Kongming Liang', 
    author_email='wangxr@bupt.edu.cn',
    url="https://github.com/PRIS-CV/seal",
    maintainer='Xinran Wang',
    packages=find_packages()
)