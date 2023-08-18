import argparse
import os.path as op
import json

from seal.task import build_task


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--project', type=str, help='the directory of the project')
    parser.add_argument('--mode', type=str, help='the mode of the project')
    
    args = parser.parse_args()
    
    with open(op.join(args.project, "task.json"), "r") as f:
        task_name = json.load(f)["name"]
    
    task = build_task(task_name)(d_config=args.project, mode=args.mode)
    task.run()
