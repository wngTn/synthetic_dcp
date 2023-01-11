import numpy as np
import sys
from pathlib import Path
import open3d as o3d
from copy import deepcopy

sys.path.insert(0, str(Path().resolve() / "lib"))
from data.data_synthetic import SmplSynthetic, SMPLAugmentation
import time

PATH = Path("data") / "smpl_training_poses.pkl"

t = time.perf_counter_ns()
augmentation = SMPLAugmentation(hat_probability = 1, mask_probability = 1, glasses_probability = 1)
dataset = SmplSynthetic("train", 1024, augmentation)
print(f"{(time.perf_counter_ns()-t)/ 1e9}")


t = time.perf_counter_ns()
for i in range(1):
    smpl, head, rotation, translation, _, _, _, _ = dataset[i]

print(f"{(time.perf_counter_ns()-t)/ 1e9}")

# head_copy = deepcopy(head)
# head_copy.translate(translation)
# head_copy.rotate(rotation, center=(0, 0, 0))


pcds_np = np.concatenate([head, smpl], axis=1) # + head_copy # comment out to check whether the inverse rotation and translation actually are the right ones

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcds_np.T)

o3d.io.write_point_cloud("assets/test_dataset.ply", pcd)
