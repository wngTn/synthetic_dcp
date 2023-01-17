import numpy as np
from probreg import filterreg
from registration.RigidRegistration import RigidRegistration
import torch
from net.model import DCP

import logging

logger = logging.getLogger(__name__)


class DCPReg(RigidRegistration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.CFG.SEED)
        torch.cuda.manual_seed_all(self.CFG.SEED)
        np.random.seed(self.CFG.SEED)

        self.net = DCP(self.CFG).cuda()
        self.net.load_state_dict(torch.load(self.cfg.TESTING.MODEL), strict=False)



    def run(self, source_pcd, target_pcd, trans_init=np.eye(4)):
        self.net.eval()
        src = torch.from_numpy(target_pcd)[None, ...].cuda()
        target = torch.from_numpy(source_pcd)[None, ...].cuda()
        (
            rotation_ab_pred,
            translation_ab_pred,
            rotation_ba_pred,
            translation_ba_pred,
        ) = self.net(src, target)
        
        R = rotation_ab_pred[0].cpu().numpy()
        T = translation_ab_pred[0].cpu().numpy()

        tf_matrix = np.eye(4)
        tf_matrix[:3, :3] = R
        tf_matrix[:3, 3] = T
        return tf_matrix

    def run_debug(self, source_pcd, target_pcd, trans_init=np.eye(4)):
        pass
