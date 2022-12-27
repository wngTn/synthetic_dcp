from pathlib import Path
import os
import open3d as o3d

import __init_paths__
from lib.utils.util import create_logger

from lib.registration.FilterReg import FilterReg
from lib.registration.ICP import ICP
from lib.registration.Aligner import Aligner
from lib.utils.util import load_mesh
from lib.utils.indices import HEAD


FRAME_ID = 2080


def main():
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

    for mesh in meshes:
        mesh.remove_vertices_by_index(list(set(range(6890)) - set(HEAD)))

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

    aligner = Aligner(voxel_size=0.0125, rigidRegistration=filterreg, icp=icp)
    aligner.align_meshes_debug(meshes, merged_pcd)


if __name__ == "__main__":
    main()
