import numpy as np
import sys
from pathlib import Path
import open3d as o3d

sys.path.insert(0, str(Path().resolve() / "lib"))
from smplmodel.body_param import load_model
from utils.util import unpack_poses
from data.data_synthetic import Synthetic_Data, SMPLAugmentation

PATH = Path("data") / "smpl_training_poses.pkl"




augmentation = SMPLAugmentation(hat_probability = 1, mask_probability = 1, glasses_probability = 1)
dataset = Synthetic_Data(100)

pcd = o3d.geometry.PointCloud()

smpl, head, rotation, translation = dataset[0]

arr = np.concatenate([head, augmentation(smpl), (head + translation) @ rotation])
pcd.points = o3d.utility.Vector3dVector(arr)

o3d.io.write_point_cloud("assets/test_dataset.ply", pcd)
