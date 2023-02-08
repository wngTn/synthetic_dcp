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
from utils.indices import HEAD, HEAD_HIDDEN
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

def get_rotation_matrix_for_basis_vecs(basis_vecs):
    basis_matrix = np.array(basis_vecs).reshape(-1, 3)
    r = R.align_vectors(basis_matrix, np.eye(3))
    return r[0].as_matrix()


# keep in memory for efficiency
MASK_MESH = o3d.io.read_triangle_mesh(str("./data/wearables/medical_mask.obj"))
HAT_MESH = o3d.io.read_triangle_mesh(str("./data/wearables/hat.ply"))
GLASSES_MESH = o3d.io.read_triangle_mesh(str("./data/wearables/glasses.obj"))

# precompute for efficiency
REMOVAL_SET_HEAD = set(range(6890)) - set(HEAD)
REMOVAL_SET_HIDDEN = set(HEAD_HIDDEN)
REMOVAL_SET_HEAD_AND_HIDDEN = REMOVAL_SET_HEAD.union(REMOVAL_SET_HIDDEN)


def rotate_and_translate(mesh, translate, rotation):
    mesh_copy = copy.deepcopy(mesh)
    mesh_copy.rotate(rotation, center=(0, 0, 0))
    mesh_copy.translate(translate)
    return mesh_copy


def add_gear_to_smpl_mesh(mesh,
                          extract_head=True,
                          remove_inner=True,
                          hat=True,
                          mask=True,
                          glasses=False
                          ):
    
    # index of vertices on the smpl mesh
    # multiply by one to copy them (removing the head changes the referenced values unfortunately)
    ear_left = mesh.vertices[6887] * 1
    ear_right = mesh.vertices[547] * 1
    mid = 0.5 * (ear_left + ear_right)
    nose = mesh.vertices[332] * 1
    top_of_head_vec = mesh.vertices[412] * 1

    # extract a new coordinate system fitting the head position
    # compute a rotation matrix for change of basis (standard basis to "head-basis")
    R = get_rotation_matrix_for_basis_vecs([mid - ear_left, top_of_head_vec - mid, nose - mid])
    
    # remove unwanted vertices from smpl mesh
    removal_set = set()
    if extract_head and remove_inner:
        removal_set = REMOVAL_SET_HEAD_AND_HIDDEN
    elif extract_head:
        removal_set = REMOVAL_SET_HEAD
    elif remove_inner:
        removal_set = REMOVAL_SET_HIDDEN

    mesh.remove_vertices_by_index(list(removal_set))

    if mask:
        mesh += rotate_and_translate(MASK_MESH, nose, R)

    if hat:
        mesh += rotate_and_translate(HAT_MESH, mid, R)

    if glasses:
        mesh += rotate_and_translate(GLASSES_MESH, nose, R)

    return mesh


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
    frame_ids = []
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
        frame_ids.append(frame["id"])

    return frame_meshes, frame_ids


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


import os.path as osp
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def rot_trans_to_homogenous(rot, trans):
    """
    Args
        rot: 3x3 rotation matrix
        trans: 3x1 translation vector
    Returns
        4x4 homogenous matrix
    """
    X = np.zeros((4, 4))
    X[:3, :3] = rot
    X[:3, 3] = trans.T
    X[3, 3] = 1
    return X


def homogenous_to_rot_trans(X):
    """
    Args
        x: 4x4 homogenous matrix
    Returns
        rotation, translation: 3x3 rotation matrix, 3x1 translation vector
    """

    return X[:3, :3], X[:3, 3].reshape(3, 1)


def rotation_to_homogenous(vec):
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap

def load_camera_params(dataset_root):
    """Loads the parameters of all the cameras in the dataset_root directory

    Args:
        dataset_root (string): the path to the camera data

    Returns:
        List[dict]: a list of dicts where the parameters are in
    """
    scaling = 1000
    cameras = list(sorted(next(os.walk(dataset_root))[1]))
    camera_params = []
    for cam in cameras:
        ds = {"id": cam}
        intrinsics = osp.join(dataset_root, cam, 'camera_calibration.yml')
        assert osp.exists(intrinsics)
        fs = cv2.FileStorage(intrinsics, cv2.FILE_STORAGE_READ)
        color_intrinsics = fs.getNode("undistorted_color_camera_matrix").mat()
        ds['fx'] = color_intrinsics[0, 0]
        ds['fy'] = color_intrinsics[1, 1]
        ds['cx'] = color_intrinsics[0, 2]
        ds['cy'] = color_intrinsics[1, 2]

        # distortion parameters can be neglected
        dist = fs.getNode("color_distortion_coefficients").mat()
        ds['k'] = np.array(dist[[0, 1, 4, 5, 6, 7]])
        ds['p'] = np.array(dist[2:4])

        depth2color_r = fs.getNode('depth2color_rotation').mat()
        depth2color_t = fs.getNode('depth2color_translation').mat() / scaling

        depth2color = rot_trans_to_homogenous(depth2color_r,
                                              depth2color_t.reshape(3))
        ds["depth2color"] = depth2color

        extrinsics = osp.join(dataset_root, cam, "world2camera.json")
        assert osp.exists(extrinsics)
        with open(extrinsics, 'r') as f:
            ext = json.load(f)
            trans = np.array([x for x in ext['translation'].values()])

            _R = ext['rotation']
            rot = Rotation.from_quat([_R['x'], _R['y'], _R['z'],
                                      _R['w']]).as_matrix()
            ext_homo = rot_trans_to_homogenous(rot, trans)

        # flip coordinate transform back to opencv convention
        yz_flip = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
        YZ_SWAP = rotation_to_homogenous(np.pi / 2 * np.array([1, 0, 0]))

        # first swap into OPENGL convention, then we can apply intrinsics.
        # then swap into our own Z-up prefered format..
        depth2world = YZ_SWAP @ ext_homo @ yz_flip

        ds["depth2world"] = depth2world
        color2world = depth2world @ np.linalg.inv(depth2color)

        ds["color2world"] = color2world

        world2color = np.linalg.inv(color2world)
        ds["world2color"] = world2color
        R, T = homogenous_to_rot_trans(world2color)
        ds["R"] = R
        ds["T"] = T

        camera_params.append(ds)

    return camera_params



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
