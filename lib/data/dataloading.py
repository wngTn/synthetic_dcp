import os
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from urllib import request
from zipfile import ZipFile
from io import BytesIO
import ssl
import open3d as o3d

# Part of the code is referred from: https://github.com/charlesq34/pointnet


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "modelnet40_ply_hdf5_2048")
    if not os.path.exists(DATA_DIR):
        www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        # ignore old expired certificate
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        print("start downloading modelnet40 data")
        http_response = request.urlopen(www, context=ctx)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=BASE_DIR)
        print("finshed downloading modelnet40 data")


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "modelnet40_ply_hdf5_2048")
    all_data = []
    all_label = []
    for h5_name in glob.glob(
        os.path.join(
            DATA_DIR, f"ply_data_{partition}*.h5" 
        )
    ):
        f = h5py.File(h5_name)
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(
        self,
        num_points,
        partition="train",
        gaussian_noise=False,
        unseen=False,
        factor=4,
    ):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == "test":
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == "train":
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        pointcloud = self.data[item][: self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != "train":
            np.random.seed(item)
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
        translation_ab = np.array(
            [
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
            ]
        )
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler("zyx", [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(
            translation_ab, axis=1
        )

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return (
            pointcloud1.astype("float32"),
            pointcloud2.astype("float32"),
            R_ab.astype("float32"),
            translation_ab.astype("float32"),
            R_ba.astype("float32"),
            translation_ba.astype("float32"),
            euler_ab.astype("float32"),
            euler_ba.astype("float32"),
        )

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, "test")
    # for data in train:
    #     print(len(data))
    #     break
    data = ModelNet40(1024, "test", gaussian_noise=True)

    for i, (src, target, rotation_ab, translation_ab, rotation_ba, translation_ba,
            euler_ab,
            euler_ba) in enumerate(data):
        if i == 10:
            break
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(src[0], src[1], src[2], color="r")
        # ax.scatter(target[0], target[1], target[2], color='g')

        # fig.savefig(f"tmp_{i}.pdf")
        pcd = o3d.geometry.PointCloud()

        points = np.concatenate((src.T, target.T, (target.T - translation_ab) @ rotation_ab ), axis=0)
        colors = np.concatenate((np.repeat([[1, 1, 1]], 1024, axis=0), np.repeat([[1, 0, 0]], 1024, axis=0),  np.repeat([[0, 0, 1]], 1024, axis=0)))

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(f"tmp2_{i}.ply", pcd)
