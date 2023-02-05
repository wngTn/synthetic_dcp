from tqdm import tqdm
import open3d as o3d
import numpy as np
from lib.utils.util import npmat2euler

def evaluate_filterreg(aligner, data_loader):
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    eulers_ab = []
    
    for i, data in enumerate(tqdm(data_loader, total = len(data_loader))):

        src, target, target_vertices, target_triangles, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = data
        rotations_ab.append(rotation_ba)
        translations_ab.append(translation_ba)
        eulers_ab.append(euler_ba)
        
        src = src[0].T
        target = target[0].T
        target_vertices = target_vertices[0]
        target_triangles  = target_triangles[0]

        target_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(target_vertices), triangles=o3d.utility.Vector3iVector(target_triangles))

        tf_matrix = aligner.align_meshes((src, target_mesh))
        rotations_ab_pred.append(np.linalg.inv(tf_matrix[:3, :3]))
        translations_ab_pred.append(tf_matrix[:3, 3])
        
        # rotations_ab_pred.append([[0,0,0], [0,0,0], [0,0,0]])
        # translations_ab_pred.append([[0,0,0]])

        # pcd = o3d.geometry.PointCloud()

        # points = np.concatenate(
        #     (
        #         src,
        #         target,
        #         (target @ rotation_ab[0]) + translation_ab[0],
        #         (target @ np.linalg.inv(tf_matrix[:3, :3])) + tf_matrix[:3, 3],
        #     ),
        #     axis=0,
        # )
        # colors = np.concatenate(
        #     (
        #         np.repeat([[1, 1, 1]], 1024, axis=0), 
        #         np.repeat([[1, 0, 0]], 1024, axis=0), # red is head to be aligned
        #         np.repeat([[0, 1, 0]], 1024, axis=0), # green gt
        #         np.repeat([[0, 0, 1]], 1024, axis=0), # blue is pred
        #     )
        # )

        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        # o3d.io.write_point_cloud(f"test{i}.ply", pcd)


    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    
    rotations_ab_pred = np.array(rotations_ab_pred)
    translations_ab_pred = np.array(translations_ab_pred)
    eulers_ab = np.concatenate(eulers_ab, axis=0)
    
    rotations_ab_pred_euler = npmat2euler(rotations_ab_pred)
    r_mse_ab = np.mean((rotations_ab_pred_euler - np.degrees(eulers_ab))**2)
    r_rmse_ab = np.sqrt(r_mse_ab)
    r_mae_ab = np.mean(np.abs(rotations_ab_pred_euler - np.degrees(eulers_ab)))
    t_mse_ab = np.mean((translations_ab - translations_ab_pred)**2)
    t_rmse_ab = np.sqrt(t_mse_ab)
    t_mae_ab = np.mean(np.abs(translations_ab - translations_ab_pred))

    print(
        f"rot_MSE: {r_mse_ab}, rot_RMSE: {r_rmse_ab}, \
        rot_MAE: {r_mae_ab}, trans_MSE: {t_mse_ab}, trans_RMSE: {t_rmse_ab}, trans_MAE: {t_mae_ab}"
    )