import torch


from ..utils.distributed import is_dist_initialized, get_world_size


def add_extra_kwargs_to_dataloader(loader_settings: dict, dataset, batch_size: int):
    
    try:
        collate_fn = dataset.collate_fn
        print("Using dataset collate function")
    except:
        collate_fn = None
        print("Using dataloader default collate function")
    finally:
        pass
    
    if is_dist_initialized():

        distributed_batch_size = batch_size // get_world_size()
        
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=loader_settings["shuffle"],
            drop_last=loader_settings["drop_last"],
        )
        # Shuffle is mutually exclusive with sampler, let DistributedSampler
        # take care of shuffle and pop from main args

        loader_settings.pop("shuffle")
        loader_settings.update({
            "dataset": dataset, 
            "batch_size": distributed_batch_size, 
            "collate_fn": collate_fn,
            "sampler": sampler
        })

    else:
        loader_settings.update({
            "dataset": dataset, 
            "batch_size": batch_size, 
            "collate_fn": collate_fn
        })

    return loader_settings


