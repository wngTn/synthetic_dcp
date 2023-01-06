import numpy as np
import sys
from pathlib import Path
import open3d as o3d
from copy import deepcopy

sys.path.insert(0, str(Path().resolve() / "lib"))
from data.data_synthetic import SyntheticData, SMPLAugmentation

PATH = Path("data") / "smpl_training_poses.pkl"


augmentation = SMPLAugmentation(hat_probability = 1, mask_probability = 1, glasses_probability = 1)
dataset = SyntheticData(100, 5024, augmentation)

pcd = o3d.geometry.PointCloud()

smpl, head, rotation, translation = dataset[0]

head_copy = deepcopy(head)
head_copy.translate(translation)
head_copy.rotate(rotation, center=(0, 0, 0))


pcds = head + smpl # + head_copy # comment out to check whether the inverse rotation and translation actually are the right ones

o3d.io.write_point_cloud("assets/test_dataset.ply", pcds)
