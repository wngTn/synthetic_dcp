from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from copy import deepcopy
import matplotlib.pyplot as plt
from glob import glob
import os
import copy

from utils.util import load_mesh, add_gear_to_smpl_mesh
from utils.indices import HEAD
from smplmodel.body_param import load_model


class TestData(Dataset):

    def __init__(self, num_output_points=1024, voxel_size=0.002):
        self.num_output_points = num_output_points

        self.path_to_trial = Path("data") / "trial"
        self.cam_list = ["cn01", "cn02", "cn03", "cn04", "cn05", "cn06"]
        self.voxel_size = voxel_size
        self.data = self.get_frames()
        self.len = len(self.data)

    def __getitem__(self, index):
        frame = self.data[index]
        merged_pcd = o3d.geometry.PointCloud()
        # Adding point clouds
        for cam in self.cam_list:
            path_to_pcd = self.path_to_trial / cam / f"{str(frame).zfill(4)}_pointcloud.ply"
            assert os.path.exists(path_to_pcd)
            merged_pcd += o3d.io.read_point_cloud(str(path_to_pcd))
        
        meshes = load_mesh(Path("data", "mesh_files"), frame)
        
        item_data = {
            "source" : [],
            "target" : []
        }

        for mesh in meshes:
            head_mesh = copy.deepcopy(mesh)
            head_mesh.remove_vertices_by_index(list(set(range(6890)) - set(HEAD)))

            head_mesh_center = head_mesh.get_center()
            head_mesh.translate(-head_mesh_center)

            head_mesh_point_cloud = head_mesh.sample_points_uniformly(number_of_points=self.num_output_points)
            head_mesh_point_cloud_np = np.array(head_mesh_point_cloud.points)
            

            pcd_head = copy.deepcopy(merged_pcd)
            # crop the point cloud
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                head_mesh.vertices
            )
            bbox = bbox.scale(1.05, bbox.get_center())

            bbox.max_bound = bbox.max_bound + np.array([0.15, 0.15, 0.15])
            bbox.min_bound = bbox.min_bound - np.array([0.15, 0.15, 0.15])

            pcd_head.translate(-head_mesh_center)
            pcd_head = pcd_head.crop(bbox)
            pcd_head = pcd_head.voxel_down_sample(self.voxel_size)
            pcd_head_np = np.array(pcd_head.points)[np.random.choice(len(pcd_head.points), 1024)]

            item_data["source"].append(pcd_head_np.T.astype("float32"))
            item_data["target"].append(head_mesh_point_cloud_np.T.astype("float32"))

        item_data["source"] = np.array(item_data["source"])
        item_data["target"] = np.array(item_data["target"])
        return item_data

    def get_frames(self):

        frames = []
        frames_in_cn01 = [int(os.path.basename(x)[:-15]) for x in glob(str(self.path_to_trial / "cn01" / "*.ply"))]
        for f in frames_in_cn01:
            if np.array([os.path.exists(self.path_to_trial / f"cn0{c+1}" / f"{str(f).zfill(4)}_pointcloud.ply") for c in range(6)]).all():
                frames.append(f)
        
        frames = list(sorted(frames))

        return frames


    def __len__(self):
        return self.len