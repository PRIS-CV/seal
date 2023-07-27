import os
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"
import argparse
import os.path as op
import torch
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
import wandb

from info.config import VAConfig
from info.data import *
from info.models import build_model, build_model_config
from info.utils import build_eval_util, build_train_util
from info.utils.utils import *
from info.utils.eval_utils import *
from info.utils.evaluators import *

VAW_DIR = "data/VAWs/data-kn-32"
VG_DIR = "/mnt/sdb/data/wangxinran/dataset/VG"
print(VAW_DIR)

#edge_file = op.join(VAW_DIR, "improve_edges.json")

fpath_attr2idx = op.join(VAW_DIR, 'attribute_index.json')
fpath_attr_type = op.join(VAW_DIR, 'attribute_types.json')
fpath_attr_parent_type = op.join(VAW_DIR, 'attribute_parent_types.json')
fpath_attr_headtail = op.join(VAW_DIR, 'head_tail.json')


def train(cfg, device):

    train_transforms = build_pipeline(cfg.pipeline)(cfg, mode="train")
    evalu_transforms = build_pipeline(cfg.pipeline)(cfg, mode="evalu")

    trainset = build_dataset(cfg.dataset)(cfg, image_path=VG_DIR, anno_path=VAW_DIR, mode='train', transform=train_transforms)
    valset = build_dataset(cfg.dataset)(cfg, image_path=VG_DIR, anno_path=VAW_DIR, mode='val', transform=evalu_transforms)
    testset = build_dataset(cfg.dataset)(cfg, image_path=VG_DIR, anno_path=VAW_DIR, mode='test', transform=evalu_transforms)
    
    try:
        collate_fn = trainset.collate_fn
        print("Using dataset collate function")
    except:
        collate_fn = None
        print("Using dataloader default collate function")
    finally:
        pass

    trainloader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    valloader = DataLoader(
        valset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True, collate_fn=collate_fn)

    testloader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=True, collate_fn=collate_fn)

    model_cfg = build_model_config(cfg.model_config_name)()
    model_cfg.load(cfg.model_config)
    model_cfg = model_cfg.cfg
    print(model_cfg)
    model = build_model(cfg.model)(model_cfg).to(device)
    
    if cfg.wandb:
        wandb.config.update(convert_cfg_to_dict(model_cfg))

    optimizer = model.get_optimizer(show_detail=False)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.1, threshold=0)
    
    highest_mAP = 0.
    train_util = build_train_util(cfg.train_util)
    eval_util = build_eval_util(cfg.eval_util)
    evaluator = VAWEvaluator(fpath_attr2idx=fpath_attr2idx, fpath_attr_headtail=fpath_attr_headtail, fpath_attr_parent_type=fpath_attr_parent_type, fpath_attr_type=fpath_attr_type)
    
    for epoch in range(cfg.epochs):
        train_util(model, trainloader, optimizer, epoch, cfg.epochs, device, amp=cfg.amp, use_wandb=cfg.wandb)
        mAP = eval_util(model, valloader, evaluator, device)
        res_dict = {"validation_mAP": mAP, "epoch": epoch}
        scheduler.step(res_dict["validation_mAP"])

        if cfg.wandb:
            wandb.log(res_dict)
        if res_dict["validation_mAP"] > highest_mAP:
            highest_mAP = res_dict["validation_mAP"]
            try:
                torch.save(model.state_dict(), op.join(cfg.weight, f'{cfg.exp_name}-model-highest.pth'))
            except:
                print("Save Model Weight Failed!")

    print("Finish Train")

    print("Testing the model of highest validation mAP.")
    weight = cfg.exp_name + "-model-highest.pth"
    state_dict = torch.load(op.join(cfg.weight, weight), map_location=device)
    model.load_state_dict(state_dict)
    mAP = eval_util(model, testloader, evaluator, device)
    res_dict = {"test_mAP": mAP}
    if cfg.wandb:
        wandb.log(res_dict)
        wandb.finish()
    print("Finish Test")
    print(res_dict)
    

def test(cfg, device):

    evalu_transforms = build_pipeline(cfg.pipeline)(cfg, mode="evalu")

    # valset = build_dataset(cfg.dataset)(cfg, image_path=VG_DIR, anno_path=VAW_DIR, mode='val', transform=evalu_transforms)
    testset = build_dataset(cfg.dataset)(cfg, image_path=VG_DIR, anno_path=VAW_DIR, mode='test', transform=evalu_transforms)
    
    # valloader = torch.utils.data.DataLoader(
    #     valset, batch_size=cfg.batch_size, shuffle=False,
    #     num_workers=cfg.num_workers, pin_memory=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, drop_last=False)
    
    model_cfg = build_model_config(cfg.model_config_name)()
    model_cfg.load(cfg.model_config)
    model_cfg = model_cfg.cfg
    print(model_cfg)
    model = build_model(cfg.model)(model_cfg).to(device)
    
    try:
        weight = cfg.exp_name + "-model-highest.pth"
        print(f"Loading pretrained weight: {weight}")
        state_dict = torch.load(op.join(cfg.weight, weight), map_location='cpu')

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print("Finish ...")
    except:
        print("Cannot find the weight! Using Initialized Weight")
        

    print("Testing the model of highest validation mAP.")
    eval_util = build_eval_util(cfg.eval_util)
    evaluator = VAWEvaluator(fpath_attr2idx=fpath_attr2idx, fpath_attr_headtail=fpath_attr_headtail, fpath_attr_parent_type=fpath_attr_parent_type, fpath_attr_type=fpath_attr_type)
    # val_mAP_score = eval_util(model, valloader, evaluator, device)
    test_mAP_score = eval_util(model, testloader, evaluator, device)
    # print(f"\n\n VAL mAP: {val_mAP_score:4f}, Test mAP: {test_mAP_score: 4f}")
    print(f"\n\n Test mAP: {test_mAP_score: 4f}")


def main(args):
    config = VAConfig()
    config.load(args.config)
    
    cfg = config.cfg
    cfg.wandb = args.wandb
    print(cfg)
    set_seed(cfg.seed)
    device = torch.device(f"cuda:{args.device}")

    if args.task == "train":
        if cfg.wandb:
            wandb.init(project=cfg.project, name=cfg.exp_name)
            wandb.config.update(convert_cfg_to_dict(cfg))
        train(cfg, device)
    elif args.task == "test":
        test(cfg, device)
    else:
        print("No such task!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--config', type=str, help='the path of configuration file')
    
    parser.add_argument('--device', type=int, default=0, help='the gpu id')
    
    parser.add_argument('--task', type=str, default="train", help='the task to run')

    parser.add_argument('--wandb', action="store_true", help='whether use wandb')

    args = parser.parse_args()

    main(args)
    
