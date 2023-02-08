import numpy as np
import pickle
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
from tqdm import tqdm
import open3d as o3d

sys.path.insert(0, str(Path().resolve() / "lib"))
from smplmodel.body_param import load_model
from utils.util import unpack_poses
from data.data_synthetic import SmplSynthetic, SMPLAugmentation

PATH = Path("data") / "smpl_training_poses.pkl"


def create_gif(num_poses=10):
    # the gif doesnt show the augmented smpl model as it is only a point cloud and the functions here
    # only use the faces of the orignal smpl model -> result doesn't contain vertices of accessoires

    body_model = load_model(
        gender="neutral", model_path=Path("data").joinpath("smpl_models")
    )

    smpl_pose = unpack_poses(Path("data") / "smpl_training_poses.pkl")

    images = []

    for _ in tqdm(range(num_poses), desc="Iterating through poses"):
        i = np.random.randint(90000)
        poses = np.array([smpl_pose[i]["pose_params"]])
        shapes = np.array([smpl_pose[i]["shape_params"]])

        vertices = body_model(
            poses,
            shapes,
            Rh=np.array([[1, -1, -1]]),
            Th=None,
            return_verts=True,
            return_tensor=False,
        )[0]

        model_info = {
            "verts": vertices,
            "joints": body_model.J_regressor.cpu().numpy() @ vertices,
        }
        fig = draw_model(
            model_info,
            model_faces=body_model.faces,
            with_joints=True,
            kintree_table=body_model.kintree_table,
        )

        fig.canvas.draw()
        fig.tight_layout(pad=0)
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(500, 500, 3)
        plt.close()

        images.append(img)

    imageio.mimwrite(Path("assets") / "smpl_poses.gif", images, duration=0.2)


def draw_model(
    model_info,
    model_faces=None,
    with_joints=False,
    kintree_table=None,
    ax=None,
    only_joint=False,
):
    """
    Displays mesh batch_idx in batch of model_info, model_info as returned by
    generate_random_model
    """

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection="3d")
    verts = model_info["verts"]
    joints = model_info["joints"]
    if model_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
    elif not only_joint:
        mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if with_joints:
        draw_skeleton(joints, kintree_table=kintree_table, ax=ax, with_numbers=False)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.75, 0.75)
    ax.set_zlim(-1.25, 0.75)
    ax.view_init(azim=-145, elev=20)
    ax.dist = 7

    return fig


def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=True):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = ax

    colors = []
    left_right_mid = ["r", "g", "b"]
    kintree_colors = [2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,0,1,0,1,0,1]
    
    for c in kintree_colors:
        colors += left_right_mid[c]
    # For each 24 joint
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot(
            [joints3D[j1, 0], joints3D[j2, 0]],
            [joints3D[j1, 1], joints3D[j2, 1]],
            [joints3D[j1, 2], joints3D[j2, 2]],
            color=colors[i],
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=5,
        )
        if with_numbers:
            ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    return ax


if __name__ == "__main__":
    create_gif(100)
