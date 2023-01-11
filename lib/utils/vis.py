import matplotlib.pyplot as plt
import numpy as np


def visualize_transformation(src, target, rotation_ab, translation_ab):
    """
    Generates a plot to visualize the rotations

    :param src: (B, 3, 1024)
    :param target: (B, 3, 1024)
    :param rotation_ab (B, 3, 3)
    :param translation_ab (B, 3)
    """
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    fig = plt.figure(figsize=(10, 4))
    for i in list(range(10)):
        _src = src[i]
        _target = target[i]
        _rotation_ab = rotation_ab[i]
        _translation_ab = translation_ab[i]
        ax = fig.add_subplot(2, 5, i+1, projection='3d')
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

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig
