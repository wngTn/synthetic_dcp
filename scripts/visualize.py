import os
from pathlib import Path
import open3d as o3d
import open3d.visualization.gui as gui
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path().resolve() / 'lib'))
from utils.util import load_mesh
from utils.indices import HEAD



FRAME_ID = 2010
# Whether to only show the head
EXTRACT_HEAD = True

def main():
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True

    data_dir = Path('data', 'trial')

    # Adding point clouds
    for cam in next(os.walk(data_dir))[1]:
        path_to_pcd = data_dir / cam / f"{str(FRAME_ID).zfill(4)}_pointcloud.ply"
        assert os.path.exists(path_to_pcd)

        vis.add_geometry(f"PCD {cam}", o3d.io.read_point_cloud(str(path_to_pcd)))

    meshes = load_mesh(Path('data', 'mesh_files'), FRAME_ID)

    for i, mesh in enumerate(meshes):
        mesh.paint_uniform_color([0.7, 0.9, 0.9])
        # Extracts the head
        if EXTRACT_HEAD:
            mesh.remove_vertices_by_index(list(set(range(6890)) - set(HEAD)))
        vis.add_geometry(f"mesh_{i}", mesh)

    app.add_window(vis)
    app.run()


if __name__=='__main__':
    main()