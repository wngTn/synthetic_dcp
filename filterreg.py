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

def test(args, aligner, test_loader):

    for it, item in tqdm(enumerate(test_loader), total=len(test_loader)):
        frame_id = item["frame_id"][0]
        dump = {}

        for mesh_vert, mesh_triang, src, target, idx in zip(item["mesh_vert"], item["mesh_triang"], item["source"], item["target"], item["ids"]):
            if len(src) == 0:
                continue
            src = src[0].T # .cpu().numpy().T
            target = target[0].T #.cpu().numpy().T

            if len(src) == 0:
                continue

            meshes = [o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh_vert[0]), triangles=o3d.utility.Vector3iVector(mesh_triang[0]))]

            tf_matrix = aligner.align_meshes((src, meshes[0]))            

            # pcd = o3d.geometry.PointCloud()

            # points = np.concatenate(
            #     (
            #         src,
            #         target,
            #         (target - tf_matrix[:3, 3]) @ tf_matrix[:3, :3],
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

            # o3d.io.write_point_cloud(f"test{it}.ply", pcd)

            dump[idx[0]] = { "rot" : tf_matrix[:3, :3].tolist(), "trans" : tf_matrix[:3, 3].tolist()}

        store_solution(args, frame_id, dump)
            

def main():
    args, cfg = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)

    test_loader = DataLoader(TestData(1024, None, load=["mesh", "pcd"]), num_workers=1, collate_fn=collate)

    registration_params_filterreg = {}
    registration_params_filterreg["max_iter"] = 1825000000
    registration_params_filterreg["w"] = 0
    registration_params_filterreg["sigma2"] = 0.000285
    registration_params_filterreg["tol"] = 0.05

    registration_params_icp = {}
    registration_params_icp["max_iter"] = 250
    registration_params_icp["threshold"] = 0.05

    icp = ICP(**registration_params_icp)
    filterreg = FilterReg(**registration_params_filterreg)

    args.exp_name = "filterreg"
    aligner = Aligner(voxel_size=0.0125, rigidRegistration=filterreg, icp=None)
    test(args, aligner, test_loader)

    print("FINISH")


if __name__ == "__main__":
    main()
