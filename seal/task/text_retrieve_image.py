import os
import os.path as op
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import logging

from ..dataset import build_dataset
from ..data import build_pipeline
from ..models import build_model
from ..evaluation import build_evaluation
from .utils import add_extra_kwargs_to_dataloader
from ..utils.distributed import is_dist_initialized
from . import task
from .task import BaseTask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@task("TextRetrievalImageTask")
class TextRetrievalImageTask(BaseTask):

    name: str = "InstanceAttributeRecognitionTask"
    project: str = "InstanceAttributeRecognitionProject"
    mode_choice: list = ["test", "retrieval"]

    def __init__(self, d_config: str, mode: str) -> None:
        super().__init__(d_config, mode)

    
    def prepare(self):
        
        if self.mode not in self.mode_choice:
            raise ValueError(f"The {self.name}'s mode must be one of following mode: {self.mode_choice}")
        
        self.model = build_model(self.model_setting.name)(**self.model_setting.get_settings()).to(self.device)

        if is_dist_initialized():
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.evaluation = build_evaluation(self.eval_settings.name)(**self.eval_settings.get_settings())
        
        self.d_weight = self.task_settings.get_settings()['d_weight']

        batch_size = self.eval_settings.get_settings()["batch_size"]

        self.evalu_transforms = build_pipeline(self.pipeline_setting.name)(
            mode="evalu", **self.pipeline_setting.get_settings()
        )

        self.visua_transforms = build_pipeline(self.pipeline_setting.name)(
            mode="visua", **self.pipeline_setting.get_settings()
        )
        
        self.testset = build_dataset(self.dataset_setting.name)(
            mode="test", transform=self.evalu_transforms, **self.dataset_setting.get_settings()
        )
        
        testloader_settings = self.dataset_setting.get_settings()["testloader"]

        testloader_settings = add_extra_kwargs_to_dataloader(testloader_settings, self.testset, batch_size)
        
        self.testloader = DataLoader(**testloader_settings)

        self.classnames = ["banana"]

        self.K = 5
        
        self.save_dir = "projects/clip_attribute_retrieval/retrival/"

    def test(self):
        values, indexs = self.evaluation(self.testloader, self.model, self.classnames, K=self.K)
        return values, indexs
    
    def retrieval(self, visualization=False):
        
        if not op.exists(self.save_dir):
            os.makedirs(self.save_dir)

        _, indexs = self.evaluation(self.testloader, self.model, self.classnames, K=self.K)
        
        assert len(indexs) == len(self.classnames)
        if visualization:
            for i, classname in enumerate(self.classnames):
                logger.info(f"Retrieve {self.K} {classname} images ... ")
                for index in map(int, indexs[i]):
                    image = self.testset.get_image_by_index(index, self.visua_transforms)
                    image.save(op.join(self.save_dir, f'{classname}_{index}.png'))

    def run(self):        
        if self.mode == "test":
            self.test()
        elif self.mode == "retrieval":
            self.retrieval(visualization=True)


@task("GroupTextRetrievalImageTask")
class GroupTextRetrievalImageTask(BaseTask):

    name: str = "InstanceAttributeRecognitionTask"
    project: str = "InstanceAttributeRecognitionProject"
    mode_choice: list = ["test", "retrieval"]

    def __init__(self, d_config: str, mode: str) -> None:
        super().__init__(d_config, mode)

    
    def prepare(self):
        
        if self.mode not in self.mode_choice:
            raise ValueError(f"The {self.name}'s mode must be one of following mode: {self.mode_choice}")
        
        self.model = build_model(self.model_setting.name)(**self.model_setting.get_settings()).to(self.device)

        if is_dist_initialized():
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.evaluation = build_evaluation(self.eval_settings.name)(**self.eval_settings.get_settings())
        
        self.d_weight = self.task_settings.get_settings()['d_weight']

        batch_size = self.eval_settings.get_settings()["batch_size"]

        self.evalu_transforms = build_pipeline(self.pipeline_setting.name)(
            mode="evalu", **self.pipeline_setting.get_settings()
        )

        self.visua_transforms = build_pipeline(self.pipeline_setting.name)(
            mode="visua", **self.pipeline_setting.get_settings()
        )
        
        self.testset = build_dataset(self.dataset_setting.name)(
            mode="test", transform=self.evalu_transforms, **self.dataset_setting.get_settings()
        )
        
        testloader_settings = self.dataset_setting.get_settings()["testloader"]

        testloader_settings = add_extra_kwargs_to_dataloader(testloader_settings, self.testset, batch_size)
        
        self.testloader = DataLoader(**testloader_settings)

        self.groups = self.testset.groups
        
        self.K = 20
        
        self.save_dir = "projects/clip_attribute_retrieval/retrival/"

    def test(self):
        values, indexs = self.evaluation(self.testloader, self.model, self.groups, K=self.K)
        return values, indexs
    
    def retrieval(self, visualization=False):
        
        if not op.exists(self.save_dir):
            os.makedirs(self.save_dir)

        _, group_indexs = self.evaluation(self.testloader, self.model, self.groups, K=self.K)
        
        assert len(group_indexs.keys()) == len(self.groups)
        if visualization:
            for g, indexs in group_indexs.items():
                logger.info(f"Retrieve top {self.K} images from {g} with size {self.testset.get_group_size(g)} ... ")
                for index in map(int, indexs):
                    image = self.testset.get_image_by_index(g, index, self.visua_transforms)
                    image.save(op.join(self.save_dir, f'{g}_{index}.png'))

    def run(self):        
        if self.mode == "test":
            self.test()
        elif self.mode == "retrieval":
            self.retrieval(visualization=True)
            