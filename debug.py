from pathlib import Path
import os
import open3d as o3d

import __init_paths__
from lib.utils.util import create_logger

from lib.registration.FilterReg import FilterReg
from lib.registration.ICP import ICP
from registration.DCPReg import DCPReg
from lib.registration.Aligner import Aligner
from lib.utils.util import load_mesh, add_gear_to_smpl_mesh
from lib.utils.indices import HEAD
import numpy as np
import argparse
from core.configs import get_cfg_defaults

FRAME_ID = 2080

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize in 3D')
    parser.add_argument('--cfg',
                        help='configuration file name',
                        type=str,
                        default='configs/default.yaml')

    args, _ = parser.parse_known_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    return cfg


def main():
    cfg = parse_args()
    logger, _ = create_logger()
    logger.setLevel("INFO")
    logger.info("Starting Registration in debug mode...")

    merged_pcd = o3d.geometry.PointCloud()

    data_dir = Path("data", "trial")

    # Adding point clouds
    for cam in next(os.walk(data_dir))[1]:
        path_to_pcd = data_dir / cam / f"{str(FRAME_ID).zfill(4)}_pointcloud.ply"
        assert os.path.exists(path_to_pcd)

        merged_pcd += o3d.io.read_point_cloud(str(path_to_pcd))
        
    meshes = load_mesh(Path("data", "mesh_files"), FRAME_ID)
    merged_meshes = []

    for mesh in meshes:
        merged_meshes.append(add_gear_to_smpl_mesh(mesh, hat=True))


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
    dcp = DCPReg(cfg=cfg)

    aligner = Aligner(voxel_size=0.0125, rigidRegistration=dcp, icp=icp)
    aligner.align_meshes_debug(merged_meshes, merged_pcd)


if __name__ == "__main__":
    main()
