from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import __init_paths__
from core.function import one_epoch
from data.data_synthetic import SmplSynthetic, SMPLAugmentation
from lib.net.dcp import DCP
from lib.net.prnet import PRNet, ACPNet
from lib.utils.util import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pathlib
import shutil
from datetime import datetime

from lib.utils.indices import HEAD
import numpy as np
import argparse
from core.configs import get_cfg_defaults

FRAME_ID = 2080


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize in 3D')
    parser.add_argument('--arc', help='architecture name', choices=['dcp', 'prnet', 'apnet (not implemented)'], default='dcp')
    args, _ = parser.parse_known_args()
    parser.add_argument('--cfg', help='configuration file name', type=str, default=f'configs/{args.arc}/default.yaml')
    parser.add_argument('--exp_name',
                        help="Name of the experiment",
                        type=str,
                        default=f"{args.arc}_{datetime.now().strftime('%m_%d-%H_%M_%S')}")

    args, _ = parser.parse_known_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    
    print(args)

    return args, cfg


class IOStream:

    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args, cfg):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + args.exp_name):
        os.makedirs("checkpoints/" + args.exp_name)
    if not os.path.exists("checkpoints/" + args.exp_name + "/" + "models"):
        os.makedirs("checkpoints/" + args.exp_name + "/" + "models")

    with open(os.path.join("checkpoints", args.exp_name, "configs.yml"), "w") as f:
        f.write(cfg.dump())

    # make it os independent
    for name in ["train", "lib/net/model", "lib/data/data_synthetic"]:
        try:
            from_file = pathlib.Path(f"{name}.py")
            to_file = pathlib.Path(f"checkpoints/{args.exp_name}/{name.split('/')[-1]}.py.backup")
            shutil.copy(from_file, to_file)
        except FileNotFoundError:
            pass


def train_dcp(args, cfg, net, train_loader, test_loader, boardio, textio):
    if cfg.TRAINING.USE_SGD:
        print("Use SGD")
        opt = optim.SGD(
            net.parameters(),
            lr=cfg.TRAINING.LR * 100,
            momentum=cfg.TRAINING.LR,
            weight_decay=1e-4,
        )
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=cfg.TRAINING.LR, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    best_test_loss = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    for epoch in range(cfg.TRAINING.EPOCHS):
        (
            train_loss,
            train_cycle_loss,
            train_rotations_ab,
            train_translations_ab,
            train_rotations_ab_pred,
            train_translations_ab_pred,
            train_rotations_ba,
            train_translations_ba,
            train_rotations_ba_pred,
            train_translations_ba_pred,
            train_eulers_ab,
            train_eulers_ba,
        ) = one_epoch(cfg, net, train_loader, opt, boardio, epoch, is_train=True)
        scheduler.step()

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab))**2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred)**2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        textio.cprint("==TRAIN==")
        textio.cprint("A--------->B")
        textio.cprint(
            f"EPOCH:: {epoch}, Loss: {train_loss}, Cycle Loss: {train_cycle_loss}, rot_MSE: {train_r_mse_ab}, rot_RMSE: {train_r_rmse_ab}, \
            rot_MAE: {train_r_mae_ab}, trans_MSE: {train_t_mse_ab}, trans_RMSE: {train_t_rmse_ab}, trans_MAE: {train_t_mae_ab}"
        )
        
        boardio.add_scalar("A->B/train/loss", train_loss, epoch)
        boardio.add_scalar("A->B/train/rotation/MSE", train_r_mse_ab, epoch)
        boardio.add_scalar("A->B/train/rotation/RMSE", train_r_rmse_ab, epoch)
        boardio.add_scalar("A->B/train/rotation/MAE", train_r_mae_ab, epoch)
        boardio.add_scalar("A->B/train/translation/MSE", train_t_mse_ab, epoch)
        boardio.add_scalar("A->B/train/translation/RMSE", train_t_rmse_ab, epoch)
        boardio.add_scalar("A->B/train/translation/MAE", train_t_mae_ab, epoch)

        if not cfg.TRAINING.OVERFIT:
            (
                test_loss,
                test_cycle_loss,
                test_rotations_ab,
                test_translations_ab,
                test_rotations_ab_pred,
                test_translations_ab_pred,
                test_rotations_ba,
                test_translations_ba,
                test_rotations_ba_pred,
                test_translations_ba_pred,
                test_eulers_ab,
                test_eulers_ba,
            ) = one_epoch(cfg, net, test_loader, None, boardio, epoch, is_train=False)

            test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
            test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab))**2)
            test_r_rmse_ab = np.sqrt(test_r_mse_ab)
            test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
            test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred)**2)
            test_t_rmse_ab = np.sqrt(test_t_mse_ab)
            test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

            if best_test_loss >= test_loss:
                best_test_loss = test_loss
                best_test_cycle_loss = test_cycle_loss

                best_test_r_mse_ab = test_r_mse_ab
                best_test_r_rmse_ab = test_r_rmse_ab
                best_test_r_mae_ab = test_r_mae_ab

                best_test_t_mse_ab = test_t_mse_ab
                best_test_t_rmse_ab = test_t_rmse_ab
                best_test_t_mae_ab = test_t_mae_ab

                if torch.cuda.device_count() > 1:
                    torch.save(
                        net.module.state_dict(),
                        "checkpoints/%s/models/model.best.t7" % args.exp_name,
                    )
                else:
                    torch.save(
                        net.state_dict(),
                        "checkpoints/%s/models/model.best.t7" % args.exp_name,
                    )

            textio.cprint("==TEST==")
            textio.cprint("A--------->B")
            textio.cprint(
                f"EPOCH:: {epoch}, Loss: {test_loss}, Cycle Loss: {test_cycle_loss}, rot_MSE: {test_r_mse_ab}, rot_RMSE: {test_r_rmse_ab}, \
                rot_MAE: {test_r_mae_ab}, trans_MSE: {test_t_mse_ab}, trans_RMSE: {test_t_rmse_ab}, trans_MAE: {test_t_mae_ab}"
            )


            textio.cprint("==BEST TEST==")
            textio.cprint("A--------->B")
            textio.cprint(
                f"EPOCH:: {epoch}, Loss: {best_test_loss}, rot_MSE: {best_test_r_mse_ab}, rot_RMSE: {best_test_r_rmse_ab}, \
                rot_MAE: {best_test_r_mae_ab}, trans_MSE: {best_test_t_mse_ab}, trans_RMSE: {best_test_t_rmse_ab}, trans_MAE: {best_test_t_mae_ab}"
            )


            ############TEST
            boardio.add_scalar("A->B/test/loss", test_loss, epoch)
            boardio.add_scalar("A->B/test/rotation/MSE", test_r_mse_ab, epoch)
            boardio.add_scalar("A->B/test/rotation/RMSE", test_r_rmse_ab, epoch)
            boardio.add_scalar("A->B/test/rotation/MAE", test_r_mae_ab, epoch)
            boardio.add_scalar("A->B/test/translation/MSE", test_t_mse_ab, epoch)
            boardio.add_scalar("A->B/test/translation/RMSE", test_t_rmse_ab, epoch)
            boardio.add_scalar("A->B/test/translation/MAE", test_t_mae_ab, epoch)

            ############BEST TEST
            boardio.add_scalar("A->B/best_test/loss", best_test_loss, epoch)
            boardio.add_scalar("A->B/best_test/rotation/MSE", best_test_r_mse_ab, epoch)
            boardio.add_scalar("A->B/best_test/rotation/RMSE", best_test_r_rmse_ab, epoch)
            boardio.add_scalar("A->B/best_test/rotation/MAE", best_test_r_mae_ab, epoch)
            boardio.add_scalar("A->B/best_test/translation/MSE", best_test_t_mse_ab, epoch)
            boardio.add_scalar("A->B/best_test/translation/RMSE", best_test_t_rmse_ab, epoch)
            boardio.add_scalar("A->B/best_test/translation/MAE", best_test_t_mae_ab, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(
                net.module.state_dict(),
                "checkpoints/%s/models/model.latest.t7" % (args.exp_name),
            )
        else:
            torch.save(
                net.state_dict(),
                "checkpoints/%s/models/model.latest.t7" % (args.exp_name),
            )
        gc.collect()

def train_prnet(args, cfg, net, train_loader, test_loader, boardio):
    if cfg.TRAINING.USE_SGD:
        print("Use SGD")
        opt = optim.SGD(
            net.parameters(),
            lr=cfg.TRAINING.LR * 100,
            momentum=cfg.TRAINING.LR,
            weight_decay=1e-4,
        )
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=cfg.TRAINING.LR, weight_decay=1e-4)
        
    epoch_factor = cfg.TRAINING.EPOCHS / 100.0

    scheduler = MultiStepLR(opt,
                            milestones=[int(30*epoch_factor), int(60*epoch_factor), int(80*epoch_factor)],
                            gamma=0.1)

    info_test_best = None

    for epoch in range(cfg.TRAINING.EPOCHS):
        info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt, boardio=boardio)
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader, boardio=boardio)
        scheduler.step()

        if info_test_best is None or info_test_best['loss'] > info_test['loss']:
            info_test_best = info_test
            info_test_best['stage'] = 'best_test'

            net.save('checkpoints/%s/models/model.best.t7' % args.exp_name)
        net.logger.write(info_test_best)

        net.save('checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


def main():
    args, cfg = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)

    if cfg.TRAINING.OVERFIT:
        train_loader = DataLoader(
        SmplSynthetic(split='overfit',
                      num_output_points=cfg.TRAINING.NUM_POINTS,
                      transform=SMPLAugmentation(glasses_probability=0.5)),
        batch_size=cfg.TRAINING.BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=True,
        drop_last=False,
    )
    else:
        train_loader = DataLoader(
            SmplSynthetic(split='train',
                        num_output_points=cfg.TRAINING.NUM_POINTS,
                        transform=SMPLAugmentation(glasses_probability=0.5)),
            batch_size=cfg.TRAINING.BATCH_SIZE,
            num_workers=os.cpu_count(),
            shuffle=True,
            drop_last=True,
        )
    test_loader = DataLoader(
        SmplSynthetic(split='test',
                      num_output_points=cfg.TRAINING.NUM_POINTS,
                      transform=SMPLAugmentation(glasses_probability=0.5)),
        batch_size=cfg.TESTING.BATCH_SIZE,
        num_workers=os.cpu_count(),
        shuffle=False,
        drop_last=False,
    )
    
    _init_(args, cfg)
    
    boardio = SummaryWriter(log_dir="checkpoints/" + args.exp_name)
    textio = IOStream("checkpoints/" + args.exp_name + "/run.log")
    
    if args.arc == "dcp":
        
        # textio.cprint(str(args))
        net = DCP(cfg).cuda()

        train_dcp(args, cfg, net, train_loader, test_loader, boardio, textio)
        boardio.close()

    elif args.arc == "prnet":
        net = PRNet(cfg, args).cuda()
        train_prnet(args, cfg, net, train_loader, test_loader, boardio)
    else:
        raise NotImplementedError()
        
    print("FINISH")
    boardio.close()


if __name__ == "__main__":
    main()