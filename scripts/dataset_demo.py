import numpy as np
import sys
from pathlib import Path
import open3d as o3d
from copy import deepcopy

sys.path.insert(0, str(Path().resolve() / "lib"))
from data.data_synthetic import SmplSynthetic, SMPLAugmentation
from data.test_data import TestData
import time

PATH = Path("data") / "smpl_training_poses.pkl"

augmentation = SMPLAugmentation(hat_probability = 1, mask_probability = 1, glasses_probability = 1)
dataset = SmplSynthetic("train", 5024, augmentation, True)

for i in range(1):
    head, smpl, R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba = dataset[i]


head_copy = deepcopy(head)
head_copy = (head_copy.T @ R_ab).T
head_copy += translation_ab[:, None]



pcds_np = np.concatenate([head], axis=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/synthetic_head.ply", pcd)

pcds_np = np.concatenate([head_copy], axis=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/synthetic_head_pred.ply", pcd)

pcds_np = np.concatenate([smpl], axis=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/synthetic_smpl.ply", pcd)



dataset = TestData(target_augmented = True)

i = 15
while True:
    data = dataset[i]
    if len(data["source"]) != 0:
        source = data["source"][0]
        target = data["target"][0]
        print(f"had to skip {i} frames")
        break
    i+=1


pcds_np = np.concatenate([source], axis=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/test_source.ply", pcd)


pcds_np = np.concatenate([target], axis=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/test_target.ply", pcd)