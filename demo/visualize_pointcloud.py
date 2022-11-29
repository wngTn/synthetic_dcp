import open3d as o3d
import open3d.visualization.gui as gui
import os

def visualize_pointcloud(vis, frame_id):
    for cam in [f"cn0{i+1}" for i in range(6)]:
        file_id = str(frame_id).zfill(4)
        fpath = os.path.join("demo", "data", cam, f"{file_id}_pointcloud.ply")
        if not os.path.exists(fpath):
            print("File does not exist: ", fpath)
            continue
        ply = o3d.io.read_point_cloud(fpath)
        vis.add_geometry(f"{cam}-ply", ply)
        print(f"Added {f'{file_id}_pointcloud.ply'}")


if __name__ == "__main__":
    # draw ball at point
    frame_id = 2000
    # new extrinsics
    # point = np.array([0.249721, -0.005661, -0.974014])
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    vis.show_settings = True
    visualize_pointcloud(vis, frame_id)
    app.add_window(vis)
    app.run()