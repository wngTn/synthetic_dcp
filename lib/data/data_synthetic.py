from torch.utils.data import Dataset
from pathlib import Path
from utils.util import unpack_poses, add_gear_to_smpl_pcd
from utils.indices import HEAD
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from smplmodel.body_param import load_model

PATH = Path("data") / "smpl_training_poses.pkl"


class SMPLAugmentation():
    def __init__(self, hat_probability = 1, mask_probability = 1, glasses_probability = 1) -> None:
        self.hat_probability = hat_probability
        self.mask_probability = mask_probability
        self.glasses_probability = glasses_probability
    
    def __call__(self, pcd):
        """
        """
        add_hat = self.hat_probability > random.random()
        add_mask = self.mask_probability > random.random()
        add_glasses = self.glasses_probability > random.random()

        return add_gear_to_smpl_pcd(pcd, False, add_hat, add_mask, add_glasses)



class Synthetic_Data(Dataset):
    
    def __init__(self, num_poses, transform = lambda x: x):
        smpl_poses = unpack_poses(PATH)
        self.transform = transform
        smpl_poses = np.random.choice(smpl_poses, num_poses)
        
        poses = np.stack([np.array(p['pose_params']) for p in smpl_poses])
        shapes = np.stack([np.array(p['shape_params']) for p in smpl_poses])
        Rh = np.repeat([[1, -1, -1]], len(poses), axis=0)
        
        # [-1.37403257 -1.90787402 -3.6412792  -0.56796243 -0.34575749 -1.35396896 -4.74416233 -1.16888448 -2.11272129 -3.70337418]
        # makes smpl model more thick
        # shapes[23] = np.ones(10) * -1.5

        body_model = load_model(gender="neutral", model_path=Path("data").joinpath("smpl_models"))

        # gets the vertices
        vertices = body_model(poses, shapes, Rh=Rh, Th=None, return_verts=True, return_tensor=False)        
        self.data = vertices
        
    def get_pcd_source_and_rotation_target(self, smpl_pcd):
        """
        return a pointcloud of the head randomly rotated and translated
        return the inverse rotation matrix and negative translation
        """
        head_pcd = smpl_pcd[HEAD]
        
        # get a random rotation matrix
        rotation = R.random(1).as_matrix()[0].T
        
        # get a random translation vector
        translation = np.random.rand(3,1).T
        
        rotation_inverse = np.linalg.inv(rotation)
        
        # rotate the pcd of the head and return the vectors to revert the augmentations (target for training later on)
        return head_pcd @ rotation + translation, rotation_inverse, - translation

    def __getitem__(self, item):
        return self.transform(self.data[item]), *self.get_pcd_source_and_rotation_target(self.data[item])

    def __len__(self):
        return len(self.data)
