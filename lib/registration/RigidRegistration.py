import numpy as np


class RigidRegistration:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def run(self, source_pcd, target_pcd, trans_init=np.eye(4)):
        pass

    def run_debug(self, source_pcd, target_pcd, trans_init=np.eye(4)):
        pass
