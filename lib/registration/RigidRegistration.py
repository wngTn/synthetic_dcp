import numpy as np
from abc import ABC, abstractmethod

class RigidRegistration(ABC):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @abstractmethod
    def run(self, source_pcd, target_pcd, trans_init=np.eye(4)):
        pass
