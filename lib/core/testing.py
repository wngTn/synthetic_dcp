import os
import argparse
import torch

import __init_paths__
from lib.data.test_data import TestData, collate
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from lib.registration.FilterReg import FilterReg
from lib.registration.ICP import ICP
from lib.registration.Aligner import Aligner
import numpy as np
import argparse
from lib.core.configs import get_cfg_defaults
import open3d as o3d
from json import JSONEncoder
import json

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
    

def store_solution(args, frame_id, dump):
    if not os.path.exists(f"data/solutions/{args.exp_name}"):
        os.makedirs(f"data/solutions/{args.exp_name}")

    # print(solution)
    with open(f"data/solutions/{args.exp_name}/{str(frame_id).zfill(4)}.json", "w+") as file:
        json_str = json.dumps(dump, indent=4)
        file.write(json_str)

def test(args, aligner, test_loader, net = None, is_filterreg = False):
    for item in tqdm(test_loader):
        frame_id = item["frame_id"][0].item()
        dump = {}

        for it, detection in enumerate(zip(item["mesh_vert"], item["mesh_triang"], item["source"], item["target"], item["ids"])):
            
            mesh_vert, mesh_triang, src, target, idx = detection
            
            if len(src) == 0 or len(target) == 0:
                continue
            
            if is_filterreg:
                src = src[0].T
                target = target[0].T
                
                meshes = [o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh_vert[0]), triangles=o3d.utility.Vector3iVector(mesh_triang[0]))]

                tf_matrix = aligner.align_meshes((src, meshes[0]))            

                dump[it] = { "rot" : tf_matrix[:3, :3].tolist(), "trans" : tf_matrix[:3, 3].tolist()}
            else:
                src = src.cuda()
                target = target.cuda()
                
                solution = net(src, target)

                solution = [x.cpu().detach().numpy() for x in solution]
                
                (
                    rotation_ab_pred,
                    translation_ab_pred,
                    rotation_ba_pred,
                    translation_ba_pred,
                ) = solution
                dump[it] = { "rot" : rotation_ab_pred.tolist(), "trans" : translation_ab_pred.tolist()}
                
        store_solution(args, frame_id, dump)