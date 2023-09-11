# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from maskrcnn_benchmark, fairseq

import contextlib
import logging
import os
import pickle
import socket
import subprocess
import warnings
from itertools import chain

import torch
from torch import distributed as dist



MAX_SIZE_LIMIT = 65533
BYTE_SIZE = 256


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def get_rank():
    
    if not dist.is_available():
        return 0
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    
    if not dist.is_available():
        return 1
    if not dist.is_nccl_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def gather_tensor_along_batch_with_backward(tensor, dim=0):
    
    world_size = get_world_size()

    if world_size < 2:
        return tensor
    
    tensor_list = GatherLayer.apply(tensor)
    tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list


def is_main():
    return is_master()


def infer_init_method(config):
    
    if config["init_method"] is not None:
        return

    # support torch.distributed.launch
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        config["init_method"] = "env://"
        config["world_size"]= int(os.environ["WORLD_SIZE"])
        config["rank"] = int(os.environ["RANK"])
        config["no_spawn"] = True

    # we can determine the init method automatically for Slurm
    elif config["port"] > 0:
        node_list = os.environ.get("SLURM_STEP_NODELIST")
        if node_list is None:
            node_list = os.environ.get("SLURM_JOB_NODELIST")
        if node_list is not None:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", node_list]
                )
                config["init_method"] = "tcp://{host}:{port}".format(
                    host=hostnames.split()[0].decode("utf-8"),
                    port=config["port"],
                )
                nnodes = int(os.environ.get("SLURM_NNODES"))
                ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
                if ntasks_per_node is not None:
                    ntasks_per_node = int(ntasks_per_node)
                else:
                    ntasks = int(os.environ.get("SLURM_NTASKS"))
                    nnodes = int(os.environ.get("SLURM_NNODES"))
                    assert ntasks % nnodes == 0
                    ntasks_per_node = int(ntasks / nnodes)
                if ntasks_per_node == 1:
                    assert config["world_size"] % nnodes == 0
                    gpus_per_node = config["world_size"] // nnodes
                    node_id = int(os.environ.get("SLURM_NODEID"))
                    config["rank"] = node_id * gpus_per_node
                else:
                    assert ntasks_per_node == config["world_size"] // nnodes
                    config["no_spawn"] = True
                    config["rank"] = int(os.environ.get("SLURM_PROCID"))
                    config.device_id = int(os.environ.get("SLURM_LOCALID"))
            except subprocess.CalledProcessError as e:  # scontrol failed
                raise e
            except FileNotFoundError:  # Slurm is not installed
                pass


def is_master():
    return get_rank() == 0


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def suppress_output(is_main):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)


def distributed_init(config):


    if config["world_size"] == 1:
        logger.info("world size is 1, disable distributed mode!")
        return 

    if dist.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
        config["rank"] = dist.get_rank()
    else:
        logger.info(
            f'Distributed Init (Rank {config["rank"]}): '
            f'{config["init_method"]}'
        )

        nccl_config = config.get("nccl", {})

        if nccl_config.get("nsocks_perthread", None):
            os.environ["NCCL_NSOCKS_PERTHREAD"] = str(nccl_config["nsocks_perthread"])
            # logger.info(f"NCCL_NSOCKS_PERTHREAD: {os.environ['NCCL_NSOCKS_PERTHREAD']}")

        if nccl_config.get("socket_nthreads", None):
            os.environ["NCCL_SOCKET_NTHREADS"] = str(nccl_config["socket_nthreads"])
            # logger.info(f"NCCL_SOCKET_NTHREADS: {os.environ['NCCL_SOCKET_NTHREADS']}")

        dist.init_process_group(
            backend=config["backend"],
            init_method=config["init_method"],
            world_size=config["world_size"],
            rank=config["rank"],
        )

        if is_main(): 
            print(config)

        logger.info(
            f'Initialized Host {socket.gethostname()} as Rank '
            f'{config["rank"]}'
        )

        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            # Set for onboxdataloader support
            split = config["init_method"].split("//")
            assert len(split) == 2, (
                "host url for distributed should be split by '//' "
                + "into exactly two elements"
            )

            split = split[1].split(":")
            assert (
                len(split) == 2
            ), "host url should be of the form <host_url>:<host_port>"
            os.environ["MASTER_ADDR"] = split[0]
            os.environ["MASTER_PORT"] = split[1]

        # perform a dummy all-reduce to initialize the NCCL communicator
        dist.all_reduce(torch.zeros(1).cuda())

        suppress_output(is_main())
        config["rank"] = dist.get_rank()
    return config["rank"]