import os.path as op
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
import logging

from ..dataset import build_dataset
from ..data import build_pipeline
from ..models import build_model
from ..evaluation import build_evaluation
from .utils import add_extra_kwargs_to_dataloader
from ..utils import build_train_util
from ..utils.distributed import is_dist_initialized
from . import task
from .task import BaseTask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@task("InstanceAttributeRecognitionTask")
class InstanceAttributeRecognitionTask(BaseTask):

    name: str = "InstanceAttributeRecognitionTask"
    project: str = "InstanceAttributeRecognitionProject"
    mode_choice: list = ["train", "test"]
    
    def prepare(self):
        
        if self.mode not in self.mode_choice:
            raise ValueError(f"The {self.name}'s mode must be one of following mode: {self.mode_choice}")
        
        self.model = build_model(self.model_setting.name)(**self.model_setting.get_settings()).to(self.device)

        if is_dist_initialized():
            self.model = nn.parallel.DistributedDataParallel(self.model)
        
        self.evaluation = build_evaluation(self.eval_settings.name)(**self.eval_settings.get_settings())
        self.d_weight = self.task_settings.get_settings()['d_weight']
        self.train_util = build_train_util(self.train_settings.name)

        if self.mode == "train":
            
            self.train_transforms = build_pipeline(self.pipeline_setting.name)(
                mode="train", **self.pipeline_setting.get_settings())
            
            self.evalu_transforms = build_pipeline(self.pipeline_setting.name)(
                mode="evalu", **self.pipeline_setting.get_settings())

            self.trainset = build_dataset(self.dataset_setting.name)(
                mode='train', transform=self.train_transforms, **self.dataset_setting.get_settings())
            
            self.valset = build_dataset(self.dataset_setting.name)(
                mode="val", transform=self.evalu_transforms, **self.dataset_setting.get_settings())
            
            self.testset = build_dataset(self.dataset_setting.name)(
                mode="test", transform=self.evalu_transforms, **self.dataset_setting.get_settings())

            
            batch_size = self.train_settings.get_settings()["batch_size"]

            trainloader_settings = self.dataset_setting.get_settings()["trainloader"]            

            trainloader_settings = add_extra_kwargs_to_dataloader(trainloader_settings, self.trainset, batch_size)

            self.trainloader = DataLoader(**trainloader_settings)

            valloader_settings = self.dataset_setting.get_settings()["valloader"]

            valloader_settings = add_extra_kwargs_to_dataloader(valloader_settings, self.valset, batch_size)

            self.valloader = DataLoader(**valloader_settings)

            testloader_settings = self.dataset_setting.get_settings()["testloader"]
            
            testloader_settings = add_extra_kwargs_to_dataloader(testloader_settings, self.testset, batch_size)
            
            self.testloader = DataLoader(**testloader_settings)

        else:

            batch_size = self.eval_settings.get_settings()["batch_size"]

            self.evalu_transforms = build_pipeline(self.pipeline_setting.name)(
                mode="evalu", **self.pipeline_setting.get_settings()
            )
            
            self.testset = build_dataset(self.dataset_setting.name)(
                mode="test", transform=self.evalu_transforms, **self.dataset_setting.get_settings()
            )
            
            testloader_settings = self.dataset_setting.get_settings()["testloader"]

            testloader_settings = add_extra_kwargs_to_dataloader(testloader_settings, self.testset, batch_size)
            
            self.testloader = DataLoader(**testloader_settings)


    def train(self):
        
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            optimizer = self.model.module.get_optimizer()
        else:
            optimizer = self.model.get_optimizer()

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, factor=0.1, threshold=0)
        
        highest_mAP = 0.
        
        for epoch in range(self.train_settings.get_settings()["epochs"]):
            self.train_util(self.model, self.trainloader, optimizer, epoch, self.train_settings.get_settings()["epochs"], self.device, amp=self.train_settings.get_settings()["amp"])
            self.evaluation(model=self.model, dataloader=self.valloader)
            mAP = self.evaluation.get_mAP()
            res_dict = {"validation_mAP": mAP, "epoch": epoch}
            scheduler.step(res_dict["validation_mAP"])

            if res_dict["validation_mAP"] > highest_mAP:
                highest_mAP = res_dict["validation_mAP"]
                try:
                    weight_name = self.project + "-model-highest.pth"
                    torch.save(self.model.state_dict(), op.join(self.d_weight, weight_name))
                except:
                    logger.info("Save Model Weight Failed!")

        logger.info("Finish Train")

        logger.info("Testing the model of highest validation mAP.")
        weight_name = self.project + "-model-highest.pth"
        state_dict = torch.load(op.join(self.d_weight, weight_name), map_location=self.device)
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.evaluation(model=self.model, dataloader=self.testloader)
        res_dict = {"test_mAP": mAP}
        logger.info("Finish Test")
        logger.info(res_dict)


    def eval(self):
        
        try:
            weight_name = self.project + "-model-highest.pth"
            logger.info(f"Loading pretrained weight: {weight_name}")
            state_dict = torch.load(op.join(self.d_weight, weight_name), map_location='cpu')

            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                missing_keys, unexpected_keys = self.model.module.load_state_dict(state_dict, strict=False)
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            logger.info(f"Missing keys: {missing_keys}")
            logger.info(f"Unexpected keys: {unexpected_keys}")
            logger.info("Finish ...")
        except:
            logger.info("Cannot find the weight! Using Initialized Weight")
            
        logger.info("Testing the model of highest validation mAP.")  
        
        self.evaluation(self.testloader, self.model)
        
    def run(self):
        if self.mode == "train":
            self.train()
        else:
            self.eval()