from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from copy import deepcopy
import matplotlib.pyplot as plt
import time

from utils.util import unpack_poses, add_gear_to_smpl_mesh
from utils.indices import HEAD
from smplmodel.body_param import load_model

PATH = Path("data") / "smpl_training_poses.pkl"


class SMPLAugmentation():

    def __init__(self, hat_probability=1, mask_probability=1, glasses_probability=1) -> None:
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


class SmplSynthetic(Dataset):

    def __init__(self, split, num_output_points=1024, transform=lambda x: x):
        self.smpl_poses = unpack_poses(PATH)
        self.transform = transform
        self.num_output_points = num_output_points
        self.factor = 4
        self.body_model = load_model(gender="neutral", model_path=Path("data").joinpath("smpl_models"))
        self.len = 2000 if split == 'train' else 200

    def gaussian_noise(self, pcd, variance):
        noise = 1 + np.random.normal(0, variance, self.num_output_points * 3).reshape((-1, 3))
        pcd.points = o3d.utility.Vector3dVector(noise * np.array(pcd.points))
        return pcd

    def crop_mesh(self, mesh_full, mesh_head):
        # crop the point cloud
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh_head.vertices)
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
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(-0.15, 0.15),
            np.random.uniform(-0.15, 0.15),
        ])
        translation_ba = -R_ba.dot(translation_ab)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        rotation_ab = R.from_euler("zyx", [anglez, angley, anglex])
        head_mesh.vertices = o3d.utility.Vector3dVector(
            (rotation_ab.apply(np.array(head_mesh.vertices)).T + np.expand_dims(translation_ab, axis=1)).T)

        # rotate the pcd of the head and return the vectors to revert the augmentations (target for training later on)

        return head_mesh, R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba

    def __getitem__(self, item):

        # we shouldn't randomize the dataloading here
        # the torch dataloader introduces shuffeling on its own already
        # e.g. len == 200 now actually means only choosing randomly from the first 200 and not 200 from all 90.000 poses

        smpl_pose = self.smpl_poses[item]

        poses = np.array([smpl_pose['pose_params']])
        shapes = np.array([smpl_pose['shape_params']])
        Rh = np.array([[1, -1, -1]])
        vertices = self.body_model(poses, shapes, Rh=Rh, Th=None, return_verts=True, return_tensor=False)[0]

        # create the smpl mesh
        smpl_mesh = o3d.geometry.TriangleMesh()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        smpl_mesh.triangles = o3d.utility.Vector3iVector(self.body_model.faces)

        # get mesh of the smpl head
        original_head_mesh = deepcopy(smpl_mesh)
        original_head_mesh.remove_vertices_by_index(list(set(range(6890)) - set(HEAD)))

        # translate mesh so that the head is in the center for rotation
        head_center = original_head_mesh.get_center()
        smpl_mesh.translate(-head_center)
        original_head_mesh.translate(-head_center)

        # get the rotated head (problem) and the inverse rotation (solution) of the rigid regestration problem
        transformed_head_mesh, R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba = self.get_mesh_source_and_rotation_target(
            smpl_mesh)

        # add accesouirs to model
        augmented_mesh = self.transform(smpl_mesh)

        # crop mesh with bounding box
        augmented_mesh = self.crop_mesh(augmented_mesh, transformed_head_mesh)

        # uniformly downsample the meshes so we can have the dimension the rigid regestration requires
        augmented_pcd = augmented_mesh.sample_points_uniformly(number_of_points=self.num_output_points)
        # augmented_pcd = augmented_mesh.sample_points_poisson_disk(number_of_points=self.num_output_points,
        #                                                           pcl=augmented_pcd)

        transformed_head_pcd = transformed_head_mesh.sample_points_uniformly(number_of_points=self.num_output_points)
        # transformed_head_pcd = transformed_head_mesh.sample_points_poisson_disk(number_of_points=self.num_output_points,
        #                                                                         pcl=transformed_head_pcd)

        # add some noise to more closely match natural noise of messurements
        augmented_pcd = self.gaussian_noise(augmented_pcd, .01)

        pointcloud1 = np.random.permutation(np.array(augmented_pcd.points)).T
        pointcloud2 = np.random.permutation(np.array(transformed_head_pcd.points)).T

        return pointcloud1.astype("float32"), pointcloud2.astype("float32"), R_ab.astype(
            "float32"), translation_ab.astype("float32"), R_ba.astype("float32"), translation_ba.astype(
                "float32"), euler_ab.astype("float32"), euler_ba.astype("float32"),

    def __len__(self):
        return self.len


if __name__ == '__main__':
    data = SmplSynthetic('train', 1024, transform=SMPLAugmentation())

    total_src = []
    total_target = []
    total_rotation_ab = []
    total_translation = []
    for i, (src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab,
            euler_ba) in enumerate(data):
        if i == 10:
            break
        total_src.append(src)
        total_target.append(target)
        total_rotation_ab.append(rotation_ab)
        total_translation.append(translation_ab)

    total_src = np.array(total_src)
    total_target = np.array(total_target)
    total_rotation_ab = np.array(total_rotation_ab)
    total_translation = np.array(total_translation)

    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    fig = plt.figure(figsize=(10, 4))

    for i in list(range(10)):
        _src = total_src[i]
        _target = total_target[i]
        _rotation_ab = total_rotation_ab[i]
        _translation_ab = total_translation[i]
        ax = fig.add_subplot(2, 5, i + 1, projection='3d')
        ax.scatter(_src[0], _src[1], _src[2], alpha=0.2, color='k', marker=".", s=1)
        ax.scatter(_target[0], _target[1], _target[2], alpha=0.2, color='r', marker=".", s=1)
        aligned = ((_target.T - _translation_ab) @ _rotation_ab).T
        ax.scatter(aligned[0], aligned[1], aligned[2], alpha=0.2, color='g', marker=".", s=1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.view_init(azim=-145, elev=20)

        pcd = o3d.geometry.PointCloud()

        points = np.concatenate((_src.T, _target.T, (_target.T - _translation_ab) @ _rotation_ab ), axis=0)
        colors = np.concatenate((np.repeat([[1, 1, 1]], 1024, axis=0), np.repeat([[1, 0, 0]], 1024, axis=0),  np.repeat([[0, 0, 1]], 1024, axis=0)))

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(f"tmp_{i}.ply", pcd)

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)
    # fig.savefig("test.pdf")
