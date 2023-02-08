import numpy as np
from probreg import filterreg
from registration.RigidRegistration import RigidRegistration

import logging

logger = logging.getLogger(__name__)


class FilterReg(RigidRegistration):
    """
    Class implements Probreg's Gaussian filter regression algorithm
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, source_pcd, target_pcd, trans_init=np.eye(4)):

        tf_param = filterreg.registration_filterreg(
            source_pcd,
            target_pcd,
            objective_type="pt2pt",
            target_normals=np.asarray(target_pcd.normals),
            maxiter=self.max_iter,
            sigma2=self.sigma2,
            tol=self.tol,
        )

        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = tf_param[0].rot
        tf_matrix[:3, 3] = tf_param[0].t

        return tf_matrix
