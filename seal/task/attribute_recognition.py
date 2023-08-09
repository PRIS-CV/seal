import os.path as op
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler

from ..dataset import build_dataset
from ..data import build_pipeline
from ..models import build_model
from ..utils import build_train_util, build_eval_util
from ..utils.evaluators import VAWEvaluator
from . import task
from .task import BaseTask


@task("InstanceAttributeRecognitionTask")
class InstanceAttributeRecognitionTask(BaseTask):

    name: str = "Instance Attribute Recognition Task"

    

    

    def prepare(self):
        
        self.model = build_model(self.model_setting.name)().to(self.device)

        if self.task_settings["training"]:
            
            self.train_transforms = build_pipeline(self.pipeline_setting.name)(
                mode="train", **self.pipeline_setting.get_settings())
            
            self.evalu_transforms = build_pipeline(self.pipeline_setting.name)(
                mode="evalu", **self.pipeline_setting.get_settings())

            self.trainset = build_dataset(self.dataset_setting.name)(
                mode='train', transform=self.train_transforms, **self.dataset_setting.get_settings())
            self.valset = build_dataset(self.dataset_setting.name)(
                mode="val", transform=self.evalu_transforms, **self.dataset_setting.get_settings())

            try:
                collate_fn = self.trainset.collate_fn
                print("Using dataset collate function")
            except:
                collate_fn = None
                print("Using dataloader default collate function")
            finally:
                pass

            batch_size = self.train_settings["batch_size"]

            trainloader_settings = self.dataset_setting.get_settings()["trainloader"]            

            trainloader_settings.update({
                "dataset": self.trainset, 
                "batch_size": batch_size, 
                "collate_fn": collate_fn
            })

            self.trainloader = DataLoader(**trainloader_settings)


            valloader_settings = self.dataset_setting.get_settings()["valloader"]

            valloader_settings.update({
                "dataset": self.valset, 
                "batch_size": batch_size, 
                "collate_fn": collate_fn
            })
            
            self.valloader = DataLoader(**valloader_settings)

        else:

            batch_size = self.eval_settings["batch_size"]

            self.evalu_transforms = build_pipeline(self.pipeline.name)(mode="evalu", **self.pipeline_setting.get_settings())
            
            self.testset = build_dataset(self.dataset_setting.name)(
                mode="test", transform=self.evalu_transforms, **self.dataset_setting.get_settings())

            try:
                collate_fn = self.testset.collate_fn
                print("Using dataset collate function")
            except:
                collate_fn = None
                print("Using dataloader default collate function")
            finally:
                pass
            
            testloader_settings = self.dataset_setting.get_settings()["testloader"]
            
            testloader_settings.update({
                "dataset":self.testset, 
                "batch_size": batch_size, 
                "collate_fn": collate_fn
            })
            
            self.testloader = DataLoader(**testloader_settings)
    

    def train(self):

        d_weight = self.task_settings['d_weight']

        optimizer = self.model.get_optimizer(show_detail=False)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.1, threshold=0)
        
        highest_mAP = 0.
        
        train_util = build_train_util(self.train_settings.name)
        
        eval_util = build_eval_util(self.eval_settings.name)
        
        evaluator = VAWEvaluator()
        
        for epoch in range(self.train_settings["epochs"]):
            train_util(self.model, self.trainloader, optimizer, epoch, self.train_settings["epochs"], self.device, amp=self.train_settings["amp"])
            mAP = eval_util(self.model, self.valloader, evaluator, self.device)
            res_dict = {"validation_mAP": mAP, "epoch": epoch}
            scheduler.step(res_dict["validation_mAP"])

            if res_dict["validation_mAP"] > highest_mAP:
                highest_mAP = res_dict["validation_mAP"]
                try:
                    weight_name = self.task_name + "-model-highest.pth"
                    torch.save(self.model.state_dict(), op.join(d_weight, f'{self.name}-model-highest.pth'))
                except:
                    print("Save Model Weight Failed!")

        print("Finish Train")

        print("Testing the model of highest validation mAP.")
        weight_name = self.name + "-model-highest.pth"
        state_dict = torch.load(op.join(d_weight, weight_name), map_location=self.device)
        self.model.load_state_dict(state_dict)
        mAP = eval_util(self.model, self.testloader, evaluator, self.device)
        res_dict = {"test_mAP": mAP}
        print("Finish Test")
        print(res_dict)

    def run(self):
        
        if self.task_settings['training']:
            self.train()