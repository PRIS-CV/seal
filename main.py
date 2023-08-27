import argparse
import os.path as op
import json
import random
import torch
from pprint import pprint

from seal.task import build_task
from seal.utils.distributed import distributed_init, infer_init_method, get_rank


def distributed_main(device_id, dir_project, task_name, task_setting, mode):
    task_setting["device_id"] = device_id
    if task_setting["rank"] is None:
        task_setting["rank"] = task_setting["start_rank"] + device_id

    main(dir_project, task_name, task_setting, init_distributed=True, mode=mode)

    

def main(dir_project, task_name, task_setting, init_distributed=False, mode=False):

    if torch.cuda.is_available():
        torch.cuda.set_device(task_setting["device_id"])
        torch.cuda.init()

    if init_distributed:
        distributed_init(task_setting)

    task = build_task(task_name)(d_config=dir_project, mode=mode)
    task.run()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--project', type=str, help='the directory of the project')
    parser.add_argument('--mode', type=str, help='the mode of the project')
    
    args = parser.parse_args()

    f_task_config = op.join(args.project, "task.json")

    with open(f_task_config, "r") as f:
        task_config = json.load(f)
        task_name = task_config['name']
        task_setting = task_config['settings']

    task_setting["start_rank"] = 0
    if task_setting["init_method"] is None:
        infer_init_method(task_setting)

    if task_setting["init_method"] is not None:

        if torch.cuda.device_count() > 1 and not task_setting["no_spawn"]:
            task_setting.start_rank = task_setting["rank"]
            task_settingrank = None

            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args.project, task_name, task_setting, args.mode),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(task_setting['rank'], args.project, task_name, task_setting, args.mode)
    
    elif task_setting["world_size"] > 1:
        
        assert task_setting["world_size"] <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        task_setting["init_method"] = f"tcp://localhost:{port}"
        task_setting["rank"] = None
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args.project, task_name, task_setting, args.mode),
            nprocs=task_setting["world_size"],
        )
    else:
        task_setting["device_id"] = 0
        main(args.project, task_name, task_setting, mode=args.mode)
    