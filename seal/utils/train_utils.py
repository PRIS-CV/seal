from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from seal.utils import train_util
from seal.utils.utils import load_to_device


@train_util("train_one_epoch")
def train_one_epoch(model, dataloader, optimizer, epoch, epoch_num, device, amp=True, use_wandb=False):
    if amp:
        scaler = GradScaler()
    pbar = tqdm(dataloader)
    pbar.set_description(f'Epoch: {epoch + 1} / {epoch_num} Training')

    device = next(model.parameters()).device
    model.train()
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        data = load_to_device(data, device)
        if amp:
            with autocast():
                loss, loss_dict = model(data)
                
            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  
        else:
            loss, loss_dict = model(data)
            loss.backward()
            optimizer.step()
        lr_dict = {f"param-group-{i}": pg['lr'] for i, pg in enumerate(optimizer.state_dict()['param_groups'])}
        loss_dict.update(lr_dict)
        pbar.set_postfix(loss_dict)
        if use_wandb:
            wandb.log(loss_dict)

