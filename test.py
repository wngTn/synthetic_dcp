from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import __init_paths__
from lib.data.dataloading import ModelNet40
from lib.core.function_dcp import one_epoch
from lib.data.test_data import TestData
from lib.net.dcp import DCP
from lib.utils.util import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pathlib
import shutil
from datetime import datetime

from lib.utils.indices import HEAD
import numpy as np
import argparse
from lib.core.configs import get_cfg_defaults
import open3d as o3d
from json import JSONEncoder
import json

FRAME_ID = 2080


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize in 3D")
    parser.add_argument(
        "--arc", help="architecture name", choices=["dcp", "prnet"], default="dcp"
    )
    args, _ = parser.parse_known_args()
    parser.add_argument(
        "--cfg",
        help="configuration file name",
        type=str,
        default=f"configs/{args.arc}/default.yaml",
    )
    parser.add_argument(
        "--exp_name",
        help="Name of the experiment",
        type=str,
        default=f"{args.arc}_{datetime.now().strftime('%m_%d-%H_%M_%S')}",
    )

    args, _ = parser.parse_known_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    return args, cfg


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    

def store_solution(args, index, solution):
    if not os.path.exists(f"data/solutions/{args.exp_name}"):
        os.makedirs(f"data/solutions/{args.exp_name}")

    print(solution)
    with open(f"data/solutions/{args.exp_name}/{index}.json", "w+") as file:
        solution = [x.tolist() for x in solution]
        json_str = json.dumps(solution)
        file.write(json_str)

def read_solution(args, index):
    with open(f"data/solutions/{args.exp_name}/{index}.json", "r") as file:
        lists = json.loads(file.read())
        solution = [np.array(x, dtype=np.float32) for x in lists]
        return solution

def test(args, cfg, net, test_loader):
    net.eval()

    for it, item in tqdm(enumerate(test_loader), total=len(test_loader)):
        # if it <= 100:
        #     continue
        if it >= 110:
            break

        for src, target in zip(item["source"], item["target"]):
            src = src.cuda()
            target = target.cuda()

            if len(src) == 0 or len(target) == 0:
                continue

            solution = net(src, target)

            solution = [x.cpu().detach().numpy() for x in solution]

            store_solution(args, it, solution)
            
            # check that storing and loading is transparent and correct
            assert all(np.array_equal(x, y) for x, y in zip(solution, read_solution(args, it)))
            
            (
                rotation_ab_pred,
                translation_ab_pred,
                rotation_ba_pred,
                translation_ba_pred,
            ) = solution

            # pcd = o3d.geometry.PointCloud()

            # points = np.concatenate(
            #     (
            #         src[0].T,
            #         target[0].T,
            #         (target[0].T - translation_ab_pred[0]) @ rotation_ab_pred[0],
            #     ),
            #     axis=0,
            # )
            # colors = np.concatenate(
            #     (
            #         np.repeat([[1, 1, 1]], 1024, axis=0),
            #         np.repeat([[1, 0, 0]], 1024, axis=0),
            #         np.repeat([[0, 0, 1]], 1024, axis=0),
            #     )
            # )

            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)

            # o3d.io.write_point_cloud(f"test{it}_{j}.ply", pcd)


def main():
    args, cfg = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)

    test_loader = DataLoader(TestData(1024), num_workers=1)

    net = DCP(cfg).cuda()

    with torch.no_grad():
        test(args, cfg, net, test_loader)

    print("FINISH")


if __name__ == "__main__":
    main()
