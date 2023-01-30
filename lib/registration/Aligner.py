import copy
import open3d as o3d
import numpy as np
from typing import List, Tuple
import multiprocessing
from multiprocessing import get_context
from tqdm import tqdm

from registration.RigidRegistration import RigidRegistration
from registration.ICP import ICP

import logging

logger = logging.getLogger(__name__)

head_mesh = None
head_point_cloud = None
head_mesh_point_cloud = None

transformation_list = []

COUNTER = 0
MAX_ITERATION = 0


def create_bbox(x, y, z, l):
    bbox = (
        np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        * l
    )
    bbox = (bbox + [x, y, z]).astype("float64")
    return bbox


def key_callback(vis):
    """
    Method for debugging purposes
    """
    global COUNTER
    global head_mesh
    global head_mesh_point_cloud
    global transformation_list
    if COUNTER < MAX_ITERATION:
        COUNTER += 1
        head_mesh = head_mesh.transform(transformation_list[COUNTER])
        head_mesh_point_cloud = head_mesh_point_cloud.transform(
            transformation_list[COUNTER]
        )
        vis.update_geometry(head_mesh)
        vis.update_geometry(head_mesh_point_cloud)
        vis.poll_events()
        vis.update_renderer()
        logger.info(f"Showing iteration: {COUNTER}")

        head_mesh = head_mesh.transform(np.linalg.inv(transformation_list[COUNTER]))
        head_mesh_point_cloud = head_mesh_point_cloud.transform(
            np.linalg.inv(transformation_list[COUNTER])
        )


def key_callback_reverse(vis):
    """
    Method for debugging purposes
    """
    global COUNTER
    global head_mesh
    global head_mesh_point_cloud
    global transformation_list
    if COUNTER >= 1:
        COUNTER -= 1
        head_mesh = head_mesh.transform(transformation_list[COUNTER])
        head_mesh_point_cloud = head_mesh_point_cloud.transform(
            transformation_list[COUNTER]
        )
        vis.update_geometry(head_mesh)
        vis.update_geometry(head_mesh_point_cloud)
        vis.poll_events()
        vis.update_renderer()
        logger.info(f"Showing iteration: {COUNTER}")
        head_mesh = head_mesh.transform(np.linalg.inv(transformation_list[COUNTER]))
        head_mesh_point_cloud = head_mesh_point_cloud.transform(
            np.linalg.inv(transformation_list[COUNTER])
        )


def remove_clouds_outliers(pcd):

    cl, ind = pcd.remove_radius_outlier(nb_points=80, radius=0.05)
    pcd = pcd.select_by_index(ind)

    # if ratio > 0:
    #     cl, ind = o3d.statistical_outlier_removal(pcds, 50, ratio)
    #     pcds = o3d.select_down_sample(pcds, ind)

    return pcd


class Aligner:
    def __init__(
        self, voxel_size, rigidRegistration: RigidRegistration, icp: ICP = None
    ):
        """
        Args:
            init_algorithm: the algorithm used for the initial transformation of the mesh -> ['gmmTree', 'ransac', 'fgr', 'l2dist_regs', 'gmmTree']
        """
        self.init_algorithm = rigidRegistration
        self.icp = icp

        self.voxel_size = voxel_size

        global MAX_ITERATION
        if self.icp is not None:
            MAX_ITERATION = icp.max_iter
        else:
            MAX_ITERATION = 0

    def align_meshes(self, zipped_pcds_facesmesh):
        head_points = zipped_pcds_facesmesh[0]
        mesh = zipped_pcds_facesmesh[1]

        head_point_cloud = o3d.geometry.PointCloud()

        head_point_cloud.points = o3d.utility.Vector3dVector(head_points)
        head_point_cloud.estimate_normals()

        head_mesh_copy = copy.deepcopy(mesh)
        head_mesh_point_cloud = head_mesh_copy.sample_points_uniformly(
            number_of_points=10000
        )
        head_mesh_point_cloud = head_mesh_copy.sample_points_poisson_disk(
            number_of_points=1500, pcl=head_mesh_point_cloud
        )

        if self.init_algorithm is not None:
            init_tf_matrix = self.init_algorithm.run(
                head_mesh_point_cloud, head_point_cloud
            )
        else:
            init_tf_matrix = np.eye(4)

        if self.icp is not None:
            tf_matrix = self.icp.run(
                head_mesh_point_cloud,
                head_point_cloud,
                trans_init=init_tf_matrix,
            )
        else:
            tf_matrix = init_tf_matrix

        return tf_matrix

    def align_meshes_debug(
        self,
        list_head_meshes: List[o3d.geometry.TriangleMesh],
        pcd: o3d.geometry.PointCloud,
    ):
        """
        Aligns the meshes, debug version
        Args:
            mesh_nose: dict with mesh : nose (coordinates)
            merged_pcd : the merged point cloud

        Return: transformation matrix
        """

        for i, mesh in enumerate(list_head_meshes):

            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            logger.info(f"Considering person {i}")
            global head_mesh
            global head_point_cloud
            global head_mesh_point_cloud
            global transformation_list
            global PERSON_COUNTER
            global COUNTER
            COUNTER = 0

            transformation_list = []

            transformation_list.append(np.eye(4))

            head_mesh = copy.deepcopy(mesh)
            head_point_cloud = copy.deepcopy(pcd)

            # scales the head down since they are a bit bigger
            head_mesh = head_mesh.scale(0.975, head_mesh.get_center())

            # create point cloud out of head_mesh
            head_mesh_point_cloud = head_mesh.sample_points_uniformly(
                number_of_points=10000
            )
            head_mesh_point_cloud = head_mesh.sample_points_poisson_disk(
                number_of_points=1024, pcl=head_mesh_point_cloud
            )

            # crop the point cloud
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                head_mesh.vertices
            )
            bbox = bbox.scale(1.05, bbox.get_center())

            # a = bbox.extent * np.array([3, 1, 4])
            # bbox.extent = a
            bbox.max_bound = bbox.max_bound + np.array([0.15, 0.15, 0.15])
            bbox.min_bound = bbox.min_bound - np.array([0.15, 0.15, 0.15])
            bbox.color = [0.6, 0.6, 0.6]
            vis.add_geometry(bbox)

            head_point_cloud = head_point_cloud.crop(bbox)
            

            # Preprocessing the point clouds:

            # removes outlier of the point cloud
            # head_point_cloud[i] = remove_clouds_outliers(head_point_cloud[i])

            # downsampling the point cloud
            head_mesh_point_cloud = head_mesh_point_cloud.voxel_down_sample(
                self.voxel_size
            )
            head_mesh_point_cloud.paint_uniform_color([1, 0, 0])
            head_point_cloud = head_point_cloud.voxel_down_sample(self.voxel_size)
            head_mesh_point_cloud = np.random.choice(len(head_mesh_point_cloud), 1024)

            # adds the objects to the visualization window
            # vis.add_geometry(head_mesh)
            vis.add_geometry(head_mesh_point_cloud)
            vis.add_geometry(head_point_cloud)

            # icp_algorithm = O3d_ICP(copy.deepcopy(head_mesh_point_cloud[i]), copy.deepcopy(head_point_cloud[i]), voxel_size)
            head_point_cloud.estimate_normals()

            for j in range(MAX_ITERATION):
                # initial global registration
                if j == 0:
                    init_tf_matrix = self.init_algorithm.run(
                        head_mesh_point_cloud, head_point_cloud
                    )
                    transformation_list.append(init_tf_matrix)
                    logger.info("Added initial transformation")

                else:
                    previous_transformation = transformation_list[j]

                    tf_matrix = self.icp.run_debug(
                        head_mesh_point_cloud,
                        head_point_cloud,
                        trans_init=previous_transformation,
                    )

                    transformation_list.append(tf_matrix)

            vis.register_key_callback(65, key_callback)
            vis.register_key_callback(83, key_callback_reverse)
            vis.register_key_callback(68, exit)

            vis.run()
            vis.destroy_window()
