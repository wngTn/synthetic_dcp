import os
import argparse
import torch
import pathlib
import shutil
from datetime import datetime
import argparse
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter

import __init_paths__
from lib.core.training import train_dcp, train_prnet, IOStream
from lib.core.function_filterreg import evaluate_filterreg
from lib.data.data_synthetic import SmplSynthetic, SMPLAugmentation
from lib.net.dcp import DCP
from lib.net.prnet import PRNet
from lib.data.test_data import TestData, collate
from lib.registration.FilterReg import FilterReg
from lib.registration.ICP import ICP
from lib.registration.Aligner import Aligner
from lib.core.testing import test

from lib.core.configs import get_cfg_defaults

def parse_args():
    parser = argparse.ArgumentParser(description='Run training (only dcp/prnet) or testing')
    parser.add_argument('--arc', help='architecture name', choices=['dcp', 'dcp_global','prnet', 'filterreg', 'filterreg_icp'], default='dcp')
    parser.add_argument('--data', help='synthetic/testing mode or realworld data testing', choices=['synthetic', 'real'], default="synthetic")
    args, _ = parser.parse_known_args()
    parser.add_argument('--cfg', help='configuration file name', type=str, default=f'configs/{args.arc}/default.yaml')
    parser.add_argument('--exp_name',
                        help="Name of the experiment",
                        type=str,
                        default=f"{args.arc}_{args.data}_{datetime.now().strftime('%m_%d-%H_%M_%S')}")

    args, _ = parser.parse_known_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    
    print(args)

    return args, cfg


def _init_(args, cfg):
    if not os.path.exists(f"checkpoints/{args.exp_name}/models"):
        os.makedirs(f"checkpoints/{args.exp_name}/models")
    with open(f"checkpoints/{args.exp_name}/configs.yml", "w") as f:
        f.write(cfg.dump())

    # make it os independent
    for name in ["train", "lib/net/model", "lib/data/data_synthetic"]:
        try:
            from_file = pathlib.Path(f"{name}.py")
            to_file = pathlib.Path(f"checkpoints/{args.exp_name}/{name.split('/')[-1]}.py.backup")
            shutil.copy(from_file, to_file)
        except FileNotFoundError:
            pass


def main():
    args, cfg = parse_args()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    
    _init_(args, cfg)
    
    boardio = SummaryWriter(log_dir=f"checkpoints/{args.exp_name}")
    textio = IOStream(f"checkpoints/{args.exp_name}/run.log")
    if args.data == "real":
        realworld_data_loader = DataLoader(TestData(1024, 1024, load=["mesh", "pcd"]), num_workers=os.cpu_count())

        if args.arc in ["dcp", "dcp_global"]:
            global_feature = args.arc == "dcp_global"
            net = DCP(cfg, global_feature = global_feature).cuda()
            
            net.load_state_dict(torch.load("checkpoints\dcp_trainsynthetic_02_04-17_01_52\models\model.best.t7"))

            with torch.no_grad():
                net.eval()
                test(args, None, realworld_data_loader, net, is_filterreg = False)
                
        elif args.arc in ["filterreg", "filterreg_icp"]:
            registration_params_filterreg = {"max_iter": 1825000000, "w": 0, "sigma2": .000285, "tol": .05}
            registration_params_icp = {"max_iter": 250, "threshold": .05}
            icp = ICP(**registration_params_icp) if args.arc == 'filterreg_icp' else None
            filterreg = FilterReg(**registration_params_filterreg)

            aligner = Aligner(voxel_size=0.0125, rigidRegistration=filterreg, icp=icp)
            test(args, aligner, realworld_data_loader, net = None, is_filterreg = True)

        else:
            raise NotImplementedError()
    else:
        
        train_loader = DataLoader(
            SmplSynthetic(split='train',
                        num_output_points=cfg.TRAINING.NUM_POINTS,
                        transform=SMPLAugmentation(glasses_probability=0.5),
                        target_augmented = False),
            batch_size=cfg.TRAINING.BATCH_SIZE,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            SmplSynthetic(split='test',
                        num_output_points=cfg.TRAINING.NUM_POINTS,
                        transform=SMPLAugmentation(glasses_probability=0.5),
                        target_augmented = False),
            batch_size=cfg.TESTING.BATCH_SIZE,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=False,
        )
        
        if args.arc in ["dcp", "dcp_global"]:
            global_feature = args.arc == "dcp_global"
            net = DCP(cfg, global_feature = global_feature).cuda()
            # for transfer learning comment out
            # net.load_state_dict(torch.load("./pretrained/dcp_v2.t7"), strict=False)
            train_dcp(args, cfg, net, train_loader, test_loader, boardio, textio)
        elif args.arc == "prnet":
            net = PRNet(cfg, args).cuda()
            train_prnet(args, cfg, net, train_loader, test_loader, boardio)
            
        elif args.arc in ["filterreg", "filterreg_icp"]:
            
            data_loader = DataLoader(
            SmplSynthetic(split='train',
                        num_output_points=cfg.TRAINING.NUM_POINTS,
                        transform=SMPLAugmentation(glasses_probability=0.5),
                        target_augmented = False,
                        head_as_mesh = True
                        ),
            batch_size=1,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
            )
            registration_params_filterreg = {"max_iter": 1825000000, "w": 0, "sigma2": .000285, "tol": .05}
            registration_params_icp = {"max_iter": 250, "threshold": .05}
            icp = ICP(**registration_params_icp) if args.arc == 'filterreg_icp' else None
            filterreg = FilterReg(**registration_params_filterreg)

            aligner = Aligner(voxel_size=0.0125, rigidRegistration=filterreg, icp=icp)
            evaluate_filterreg(aligner, data_loader)
            
        else:
            raise NotImplementedError()

    boardio.close()


if __name__ == "__main__":
    main()