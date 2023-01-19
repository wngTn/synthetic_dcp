import numpy as np
import sys
from pathlib import Path
import open3d as o3d
from copy import deepcopy

sys.path.insert(0, str(Path().resolve() / "lib"))
from data.data_synthetic import SmplSynthetic, SMPLAugmentation
import time

PATH = Path("data") / "smpl_training_poses.pkl"

augmentation = SMPLAugmentation(hat_probability = 1, mask_probability = 1, glasses_probability = 1)
dataset = SmplSynthetic("train", 5024, augmentation)

for i in range(1):
    smpl, head, R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba = dataset[i]


head_copy = deepcopy(head)
head_copy = (head_copy.T @ R_ab).T
head_copy += translation_ab[:, None]



pcds_np = np.concatenate([head], axis=1) # + head_copy # comment out to check whether the inverse rotation and translation actually are the right ones

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/test_dataset1.ply", pcd)

pcds_np = np.concatenate([head_copy], axis=1) # + head_copy # comment out to check whether the inverse rotation and translation actually are the right ones

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/test_dataset2.ply", pcd)

pcds_np = np.concatenate([smpl], axis=1) # + head_copy # comment out to check whether the inverse rotation and translation actually are the right ones

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/test_dataset3.ply", pcd)