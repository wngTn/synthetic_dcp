from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
from smplmodel.body_param import load_model
import json
import open3d as o3d
import logging
from pathlib import Path
import time
from scipy.spatial.transform import Rotation as R
from utils.indices import HEAD
import copy
import pickle

# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py


def unpack_poses(path):
	poses = []
	with path.open('rb') as fp:
		data = pickle.load(fp)
	for pose in data:
		for j in range(len(pose['pose_params'])):
			pose_data = {
				'pose_params' : pose['pose_params'][j],
				'shape_params' : pose['shape_params'][j]
			}
			poses.append(pose_data)

	return poses

def get_rotation_matrix(vec2, vec1):
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()

def add_gear_to_smpl_pcd(pcd,
                          extract_head=True,
                          hat=True,
                          mask=True,
                          glasses=False):

    # index of vertices on the smpl model
    ear_left = pcd[6887]
    ear_right = pcd[547] 
    mid = 0.5 * (ear_left + ear_right)
    nose = pcd[332]
    top_of_head_vec = pcd[412] * 1

    # extract a new coordinate system fitting the head position
    # compute a rotation matrix for change of basis (standard basis to "head-basis")
    R = get_rotation_matrix(
        np.array([mid - ear_left, top_of_head_vec - mid, nose - mid]).reshape(-1, 3), np.eye(3))

    # Extracts the head
    if extract_head:
        mask = np.arange(len(pcd))
        mask = mask[mask in set(HEAD)]
        pcd = pcd[mask]

    if mask:
        mask_pcd = o3d.io.read_triangle_mesh(str("./data/wearables/medical_mask.obj"))
        mask_pcd.rotate(R, center=(0, 0, 0))
        mask_pcd.translate(nose)
        mask_pcd.scale(1.10, center=mask_pcd.get_center())

        pcd = np.concatenate([pcd, np.asarray(mask_pcd.vertices)], axis=0)

    if hat:
        hat_pcd = o3d.io.read_triangle_mesh(str("./data/wearables/hat.ply"))
        hat_pcd.scale(2.75, center=hat_pcd.get_center())
        hat_pcd.rotate(R, center=(0, 0, 0))
        hat_pcd.translate(mid)

        pcd = np.concatenate([pcd, np.asarray(hat_pcd.vertices)], axis=0)
        

    if glasses:
        glassed_pcd = o3d.io.read_triangle_mesh(str("./data/wearables/glasses.obj"))

        glassed_pcd.scale(1, center=glassed_pcd.get_center())
        glassed_pcd.rotate(R, center=(0, 0, 0))
        glassed_pcd.translate(nose)

        pcd = np.concatenate([pcd, np.asarray(glassed_pcd.vertices)], axis=0)

    return pcd

def add_gear_to_smpl_mesh(mesh,
                          extract_head=True,
                          get_individual=False,
                          hat=True,
                          mask=True,
                          glasses=False):
    meshes = [mesh]

    # index of vertices on the smpl mesh
    # multiply by one to copy them (removing the head changes the referenced values unfortunately)
    ear_left = mesh.vertices[6887] * 1
    ear_right = mesh.vertices[547] * 1
    mid = 0.5 * (ear_left + ear_right)
    nose = mesh.vertices[332] * 1
    top_of_head_vec = mesh.vertices[412] * 1

    # extract a new coordinate system fitting the head position
    # compute a rotation matrix for change of basis (standard basis to "head-basis")
    R = get_rotation_matrix(
        np.array([mid - ear_left, top_of_head_vec - mid, nose - mid]).reshape(-1, 3), np.eye(3))

    # Extracts the head
    if extract_head:
        mesh.remove_vertices_by_index(list(set(range(6890)) - set(HEAD)))

    if mask:
        mask_mesh = o3d.io.read_triangle_mesh(str("./data/wearables/medical_mask.obj"))
        mask_mesh.rotate(R, center=(0, 0, 0))
        mask_mesh.translate(nose)
        mask_mesh.scale(1.10, center=mask_mesh.get_center())

        meshes.append(mask_mesh)

    if hat:
        hat_mesh = o3d.io.read_triangle_mesh(str("./data/wearables/hat.ply"))
        hat_mesh.scale(2.75, center=hat_mesh.get_center())
        hat_mesh.rotate(R, center=(0, 0, 0))
        hat_mesh.translate(mid)

        meshes.append(hat_mesh)

    if glasses:
        glasses_mesh = o3d.io.read_triangle_mesh(str("./data/wearables/glasses.obj"))

        glasses_mesh.scale(1, center=glasses_mesh.get_center())
        glasses_mesh.rotate(R, center=(0, 0, 0))
        glasses_mesh.translate(nose)

        meshes.append(glasses_mesh)

    merged = copy.deepcopy(mesh)
    for wearable_mesh in meshes:
        merged += wearable_mesh

    if get_individual:
        meshes.append(merged)
        return meshes

    return merged


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq="zyx"):
    eulers = []
    for i in range(mats.shape[0]):
        # "In scipy.spatial.Rotation from_dcm was renamed to from_matrix"
        # see https://docs.scipy.org/doc/scipy/release.1.4.0.html?highlight=scipy%20spatial%20transform%20rotation%20from_dcm
        r = Rotation.from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype="float32")


# reads a json file
def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        data = json.load(f)
    return data


# reads a smpl file
def read_smpl(filename):
    datas = read_json(filename)
    outputs = []
    for data in datas:
        for key in ["Rh", "Th", "poses", "shapes", "expression"]:
            if key in data.keys():
                data[key] = np.array(data[key], dtype=np.float32)
        outputs.append(data)
    return outputs


# creates mesh out of vertices and faces
def create_mesh(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def load_joints(data_dir: str, frame_id):
    """
    Loads the body joints from the data_dir

    :param data_dir: The path to the smpl files
    :param frame_ids: The frame id
    :return: Returns a list of the body_joints of the frame id
    """
    # loads the smpl model
    body_model = load_model(gender="neutral", model_path="data/smpl_models")

    data = read_smpl(os.path.join(data_dir, str(frame_id).zfill(6) + ".json"))
    # all the meshes in a frame
    frame_pcds = []
    for i in range(len(data)):
        frame = data[i]
        Rh = frame["Rh"]
        Th = frame["Th"]
        poses = frame["poses"]
        shapes = frame["shapes"]

        # gets the vertices
        vertices = body_model(
            poses,
            shapes,
            Rh,
            Th,
            return_verts=False,
            return_tensor=False,
            return_smpl_joints=True,
        )[0]

        frame_pcds.append(vertices)

    return frame_pcds


def load_mesh(data_dir: str, frame_id):
    """
    Loads the meshes from the data_dir

    :param data_dir: The path to the smpl files
    :param frame_ids: The frame id
    :return: Returns a list of the meshes of the frame id
    """
    # loads the smpl model
    body_model = load_model(gender="neutral", model_path="data/smpl_models")

    data = read_smpl(os.path.join(data_dir, str(frame_id).zfill(6) + ".json"))
    # all the meshes in a frame
    frame_meshes = []
    for i in range(len(data)):
        frame = data[i]
        Rh = frame["Rh"]
        Th = frame["Th"]
        poses = frame["poses"]
        shapes = frame["shapes"]

        # gets the vertices
        vertices = body_model(poses, shapes, Rh, Th, return_verts=True, return_tensor=False)[0]

        # the mesh
        model = create_mesh(vertices=vertices, faces=body_model.faces)

        frame_meshes.append(model)

    return frame_meshes


def create_logger(path=None):
    final_output_dir = Path("")
    if path is not None:
        root_output_dir = Path(path).resolve()
        # set up logger
        if not root_output_dir.exists():
            print("=> creating {}".format(root_output_dir))
            root_output_dir.mkdir()

        name = "Debug_Logging"

        final_output_dir = root_output_dir / name
        print("=> creating {}".format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        log_file = "{}_{}.log".format(name, time_str)
        final_log_file = final_output_dir / log_file
        logging.basicConfig(filename=str(final_log_file))

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = "%(asctime)s | %(name)-20s | %(levelname)-s | %(message)-s"
    time_fmt = "%Y-%m-%d %H:%M:%S"

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(CustomFormatter(fmt, time_fmt))

    logging.getLogger().handlers = []
    logging.getLogger().addHandler(stdout_handler)

    return logger, str(final_output_dir)


class CustomFormatter(logging.Formatter):
    magenta = "\u001b[35m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt, time_fmt=None):
        super().__init__()
        self.fmt = fmt
        self.time_fmt = time_fmt
        self.FORMATS = {
            logging.DEBUG: self.magenta + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.time_fmt)
        return formatter.format(record)
