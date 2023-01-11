import matplotlib.pyplot as plt
import numpy as np


def visualize_transformation(src, target, rotation_ab_gt, translation_ab_gt, rotation_ab, translation_ab):
    """
    Generates a plot to visualize the rotations

    :param src: (B, 3, 1024)
    :param target: (B, 3, 1024)
    :param rotation_ab (B, 3, 3)
    :param translation_ab (B, 3)
    """
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    fig = plt.figure(figsize=(10, 4), dpi=300)
    for i in list(range(min(10, len(src)))):
        _src = src[i]
        _target = target[i]
        _rotation_ab_gt = rotation_ab_gt[i]
        _translation_ab_gt = translation_ab_gt[i]
        _rotation_ab = rotation_ab[i]
        _translation_ab = translation_ab[i]
        ax = fig.add_subplot(2, 5, i+1, projection='3d')
        ax.scatter(_src[0], _src[1], _src[2], alpha=0.2, color='k', marker=".", s=1)
        ax.scatter(_target[0], _target[1], _target[2], alpha=0.2, color='r', marker=".", s=1)
        aligned = ((_target.T - _translation_ab) @ _rotation_ab).T
        ax.scatter(aligned[0], aligned[1], aligned[2], alpha=0.2, color='g', marker=".", s=1)
        aligned_gt = ((_target.T - _translation_ab_gt) @ _rotation_ab_gt).T
        ax.scatter(aligned_gt[0], aligned_gt[1], aligned_gt[2], alpha=0.1, color='b', marker=".", s=1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.view_init(azim=-145, elev=20)
        
        # TODO add descriptions on colors in a legend
        # ax.legend()

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig
