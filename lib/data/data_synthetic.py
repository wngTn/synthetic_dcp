from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from copy import deepcopy

from utils.util import unpack_poses, add_gear_to_smpl_mesh
from utils.indices import HEAD
from smplmodel.body_param import load_model


PATH = Path("data") / "smpl_training_poses.pkl"


class SMPLAugmentation():
    def __init__(self, hat_probability = 1, mask_probability = 1, glasses_probability = 1) -> None:
        self.hat_probability = hat_probability
        self.mask_probability = mask_probability
        self.glasses_probability = glasses_probability
    
    def __call__(self, mesh):
        """
        """
        add_hat = self.hat_probability > random.random()
        add_mask = self.mask_probability > random.random()
        add_glasses = self.glasses_probability > random.random()
        
        augmented_mesh = add_gear_to_smpl_mesh(mesh, False, True, add_hat, add_mask, add_glasses)
        return augmented_mesh
    
    
class SyntheticData(Dataset):
    
    def __init__(self, num_poses, num_output_points = 1024, transform = lambda x:x):
        smpl_poses = unpack_poses(PATH)
        self.transform = transform
        self.num_output_points = num_output_points
        smpl_poses = np.random.choice(smpl_poses, num_poses)
        
        poses = np.stack([np.array(p['pose_params']) for p in smpl_poses])
        shapes = np.stack([np.array(p['shape_params']) for p in smpl_poses])
        Rh = np.repeat([[1, -1, -1]], len(poses), axis=0)
        
        # [-1.37403257 -1.90787402 -3.6412792  -0.56796243 -0.34575749 -1.35396896 -4.74416233 -1.16888448 -2.11272129 -3.70337418]
        # makes smpl model more thick
        # shapes[23] = np.ones(10) * -1.5

        self.body_model = load_model(gender="neutral", model_path=Path("data").joinpath("smpl_models"))

        # gets the vertices
        vertices = self.body_model(poses, shapes, Rh=Rh, Th=None, return_verts=True, return_tensor=False)
        self.data = vertices
        
    def gaussian_noise(self, pcd, variance):
        noise = 1 + np.random.normal(0, variance, self.num_output_points * 3).reshape((-1, 3))
        pcd.points = o3d.utility.Vector3dVector(noise * np.array(pcd.points))
        return pcd
        
    def crop_mesh(self, mesh_full, mesh_head):
        # crop the point cloud
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            mesh_head.vertices
        )
        bbox = bbox.scale(1.25, bbox.get_center())

        bbox.max_bound = bbox.max_bound + np.array([0.25, 0.25, 0.25])
        bbox.min_bound = bbox.min_bound - np.array([0.25, 0.25, 0.25])

        return mesh_full.crop(bbox)


    def get_mesh_source_and_rotation_target(self, smpl_mesh):
        """
        return a pointcloud of the head randomly rotated and translated
        return the inverse rotation matrix and negative translation
        """

        head_mesh = deepcopy(smpl_mesh)
        head_mesh.remove_vertices_by_index(list(set(range(6890)) - set(HEAD)))

        # get a random rotation matrix
        rotation = R.random(1).as_matrix()[0]

        # get a random translation vector
        translation = np.random.rand(3,1)

        head_mesh.rotate(rotation, center=(0, 0, 0))
        head_mesh.translate(translation)

        rotation_inverse = np.linalg.inv(rotation)

        # rotate the pcd of the head and return the vectors to revert the augmentations (target for training later on)
        
        return head_mesh, rotation_inverse, - translation

    def __getitem__(self, item):

        # create the smpl mesh
        smpl_mesh = o3d.geometry.TriangleMesh()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(self.data[item])
        smpl_mesh.triangles = o3d.utility.Vector3iVector(self.body_model.faces)

        # get mesh of the smpl head
        original_head_mesh = deepcopy(smpl_mesh)
        original_head_mesh.remove_vertices_by_index(list(set(range(6890)) - set(HEAD)))

        # get the rotated head (problem) and the inverse rotation (solution) of the rigid regestration problem
        translated_head_pcd, rotation_matrix, translation_vector = self.get_mesh_source_and_rotation_target(smpl_mesh)
        
        # add accesouirs to model
        augmented_mesh = self.transform(smpl_mesh)

        # crop mesh with bounding box
        augmented_mesh = self.crop_mesh(augmented_mesh, original_head_mesh)
        
        
        # uniformly downsample the meshes so we can have the dimension the rigid regestration requires
        augmented_pcd = augmented_mesh.sample_points_uniformly(number_of_points=self.num_output_points)
        translated_head_pcd = translated_head_pcd.sample_points_uniformly(number_of_points=self.num_output_points)

        # add some noise to more closely match natural noise of messurements
        augmented_pcd = self.gaussian_noise(augmented_pcd, .01)

        return augmented_pcd, translated_head_pcd, rotation_matrix, translation_vector

    def __len__(self):
        return len(self.data)
