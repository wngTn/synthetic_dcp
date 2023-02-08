import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def visualize_pred_transformation(src, target, rotation_ab, translation_ab, rotation_ab_gt, translation_ab_gt):
    """
    Generates a plot to visualize the rotations

    :param src: (B, 3, 1024)
    :param target: (B, 3, 1024)
    :param rotation_ab (B, 3, 3)
    :param translation_ab (B, 3)
    """
    pcds = []
    for i in list(range(min(10, len(src)))):
        _src = target[i]
        _target = src[i]
        _rotation_ab = rotation_ab[i]
        _translation_ab = translation_ab[i]
        _rotation_ab_gt = rotation_ab_gt[i]
        _translation_ab_gt = translation_ab_gt[i]
        pcd = o3d.geometry.PointCloud()

        # points = np.concatenate((_src.T, _target.T, (_target.T @ _rotation_ab_gt) + _translation_ab_gt, (_target.T @ _rotation_ab) + _translation_ab), axis=0)
        # colors = np.concatenate((np.repeat([[1, 1, 1]], _src.shape[1], axis=0), np.repeat([[1, 0, 0]], _target.shape[1], axis=0),  np.repeat([[0, 1, 0]], _target.shape[1], axis=0), np.repeat([[0, 0, 1]], _target.shape[1], axis=0)))
        
        # only show gt and not aligned target
        points = np.concatenate((_src.T, _target.T, (_target.T @ _rotation_ab) + _translation_ab), axis=0)
        colors = np.concatenate((np.repeat([[1, 1, 1]], _src.shape[1], axis=0), np.repeat([[1, 0, 0]], _target.shape[1], axis=0),  np.repeat([[0, 1, 0]], _target.shape[1], axis=0)))

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcds.append(pcd)

    return pcds