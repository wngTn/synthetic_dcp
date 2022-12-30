import os
from pathlib import Path
import open3d as o3d
import open3d.visualization.gui as gui
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path().resolve() / "lib"))
from utils.util import load_mesh, add_gear_to_smpl_mesh


FRAME_ID = 1020
# Whether to only show the head
EXTRACT_HEAD = True
EXTRACT_JOINTS = False


def main():
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True

    data_dir = Path("data", "trial")

    # Adding point clouds
    for cam in next(os.walk(data_dir))[1]:
        path_to_pcd = data_dir / cam / f"{str(FRAME_ID).zfill(4)}_pointcloud.ply"
        assert os.path.exists(path_to_pcd)

        #vis.add_geometry(f"PCD {cam}", o3d.io.read_point_cloud(str(path_to_pcd)))

    meshes = load_mesh(Path("data", "mesh_files"), FRAME_ID)
    

    for i, mesh in enumerate(meshes):
    
        meshes_i = add_gear_to_smpl_mesh(mesh, EXTRACT_HEAD, get_individual = True, glasses = True)
        
        
        
        vis.add_geometry(f"mesh_{i}", meshes_i[0])
        
        pcd_mask = o3d.geometry.PointCloud()
        pcd_mask.points = o3d.utility.Vector3dVector(meshes_i[1].vertices)
        vis.add_geometry(f"mask_{i}", pcd_mask)
    

        pcd_hat = o3d.geometry.PointCloud()
        pcd_hat.points = o3d.utility.Vector3dVector(meshes_i[2].vertices)
        vis.add_geometry(f"hat_{i}", pcd_hat)

        pcd_glasses = o3d.geometry.PointCloud()
        pcd_glasses.points = o3d.utility.Vector3dVector(meshes_i[3].vertices)
        vis.add_geometry(f"glasses_{i}", pcd_glasses)


        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(meshes_i[4].vertices)
        
        
        vis.add_geometry(f"merged_mesh{i}", meshes_i[4])
        vis.add_geometry(f"merged_pcd{i}", merged_pcd)


    app.add_window(vis)
    app.run()


if __name__ == "__main__":
    main()
