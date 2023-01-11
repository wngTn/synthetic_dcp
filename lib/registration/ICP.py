import open3d as o3d
import numpy as np
from registration.RigidRegistration import RigidRegistration

import logging

logger = logging.getLogger(__name__)


class ICP(RigidRegistration):
    """
    Class implements a simple ICP algorithm through open 3d's TransformationEstimationPointToPoint
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, source_pcd, target_pcd, trans_init=np.eye(4)):
        if len(target_pcd.points) < 100:
            logger.debug(
                f"The target point cloud has {len(target_pcd.points)} many points -- skipping ICP"
            )
            return self.trans_init

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            self.threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iter
            ),
        )

        logger.debug(f"Done ICP iteration with {self.max_iter} iterations.")

        return reg_p2p.transformation

    def run_debug(self, source_pcd, target_pcd, trans_init=np.eye(4)):
        if len(target_pcd.points) < 100:
            logger.debug(
                f"The target point cloud has {len(target_pcd.points)} many points -- skipping ICP"
            )
            return self.trans_init

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source=source_pcd,
            target=target_pcd,
            max_correspondence_distance=self.threshold,
            init=trans_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1),
        )

        logger.debug("Did one iteration")
        return reg_p2p.transformation
