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
from lib.data.dataloading import ModelNet40
from core.function import train_one_epoch, test_one_epoch
from data.data_synthetic import SmplSynthetic, SMPLAugmentation
from lib.net.model import DCP
from lib.utils.util import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pathlib
import shutil


class IOStream:

    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.exists("checkpoints/" + args.exp_name):
        os.makedirs("checkpoints/" + args.exp_name)
    if not os.path.exists("checkpoints/" + args.exp_name + "/" + "models"):
        os.makedirs("checkpoints/" + args.exp_name + "/" + "models")

    # make it os independent
    for name in ["main", "net/model", "lib/data/dataloading"]:
        try:
            from_file = pathlib.Path(f"{name}.py")
            to_file = pathlib.Path(f"checkpoints/{args.exp_name}/{name}.py.backup")
            shutil.copy(from_file, to_file)
        except FileNotFoundError:
            pass


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(
            net.parameters(),
            lr=args.lr * 100,
            momentum=args.momentum,
            weight_decay=1e-4,
        )
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_test_mse_ba = np.inf
    best_test_rmse_ba = np.inf
    best_test_mae_ba = np.inf

    best_test_r_mse_ba = np.inf
    best_test_r_rmse_ba = np.inf
    best_test_r_mae_ba = np.inf
    best_test_t_mse_ba = np.inf
    best_test_t_rmse_ba = np.inf
    best_test_t_mae_ba = np.inf

    for epoch in range(args.epochs):
        (
            train_loss,
            train_cycle_loss,
            train_mse_ab,
            train_mae_ab,
            train_mse_ba,
            train_mae_ba,
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
        ) = train_one_epoch(args, net, train_loader, opt, boardio, epoch)
        scheduler.step()
        (
            test_loss,
            test_cycle_loss,
            test_mse_ab,
            test_mae_ab,
            test_mse_ba,
            test_mae_ba,
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
        ) = test_one_epoch(args, net, test_loader)
        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        train_rmse_ba = np.sqrt(train_mse_ba)
        test_rmse_ba = np.sqrt(test_mse_ba)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab))**2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred)**2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred, "xyz")
        train_r_mse_ba = np.mean((train_rotations_ba_pred_euler - np.degrees(train_eulers_ba))**2)
        train_r_rmse_ba = np.sqrt(train_r_mse_ba)
        train_r_mae_ba = np.mean(np.abs(train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)))
        train_t_mse_ba = np.mean((train_translations_ba - train_translations_ba_pred)**2)
        train_t_rmse_ba = np.sqrt(train_t_mse_ba)
        train_t_mae_ba = np.mean(np.abs(train_translations_ba - train_translations_ba_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab))**2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred)**2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

        test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, "xyz")
        test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba))**2)
        test_r_rmse_ba = np.sqrt(test_r_mse_ba)
        test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
        test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred)**2)
        test_t_rmse_ba = np.sqrt(test_t_mse_ba)
        test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_cycle_loss = test_cycle_loss

            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_test_mse_ba = test_mse_ba
            best_test_rmse_ba = test_rmse_ba
            best_test_mae_ba = test_mae_ba

            best_test_r_mse_ba = test_r_mse_ba
            best_test_r_rmse_ba = test_r_rmse_ba
            best_test_r_mae_ba = test_r_mae_ba

            best_test_t_mse_ba = test_t_mse_ba
            best_test_t_rmse_ba = test_t_rmse_ba
            best_test_t_mae_ba = test_t_mae_ba

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

        textio.cprint("==TRAIN==")
        textio.cprint("A--------->B")
        textio.cprint(
            f"EPOCH:: {epoch}, Loss: {train_loss}, Cycle Loss: {train_cycle_loss}, MSE: {train_mse_ab}, RMSE: {train_rmse_ab}, MAE: {train_mae_ab}, rot_MSE: {train_r_mse_ab}, rot_RMSE: {train_r_rmse_ab}, "
            "rot_MAE: {train_r_mae_ab}, trans_MSE: {train_t_mse_ab}, trans_RMSE: {train_t_rmse_ab}, trans_MAE: {train_t_mae_ab}"
        )

        textio.cprint("B--------->A")
        textio.cprint(
            f"EPOCH:: {epoch}, Loss: {train_loss}, Cycle Loss: {train_mse_ba}, MSE: {train_rmse_ba}, RMSE: {train_rmse_ba}, MAE: {train_mae_ba}, rot_MSE: {train_r_mse_ba}, rot_RMSE: {train_r_rmse_ba}, "
            "rot_MAE: {train_r_mae_ba}, trans_MSE: {train_t_mse_ba}, trans_RMSE: {train_t_rmse_ba}, trans_MAE: {train_t_mae_ba}"
        )

        textio.cprint("==TEST==")
        textio.cprint("A--------->B")
        textio.cprint(
            f"EPOCH:: {epoch}, Loss: {test_loss}, Cycle Loss: {test_cycle_loss}, MSE: {test_mse_ab}, RMSE: {test_rmse_ab}, MAE: {test_mae_ab}, rot_MSE: {test_r_mse_ab}, rot_RMSE: {test_r_rmse_ab}, "
            "rot_MAE: {test_r_mae_ab}, trans_MSE: {test_t_mse_ab}, trans_RMSE: {test_t_rmse_ab}, trans_MAE: {test_t_mae_ab}"
        )

        textio.cprint("B--------->A")
        textio.cprint(
            f"EPOCH:: {epoch}, Loss: {test_loss}, Cycle Loss: {test_mse_ba}, MSE: {test_rmse_ba}, RMSE: {test_rmse_ba}, MAE: {test_mae_ba}, rot_MSE: {test_r_mse_ba}, rot_RMSE: {test_r_rmse_ba}, "
            "rot_MAE: {test_r_mae_ba}, trans_MSE: {test_t_mse_ba}, trans_RMSE: {test_t_rmse_ba}, trans_MAE: {test_t_mae_ba}"
        )

        textio.cprint("==BEST TEST==")
        textio.cprint("A--------->B")
        textio.cprint(
            f"EPOCH:: {epoch}, Loss: {best_test_loss}, Cycle Loss: {best_test_cycle_loss}, MSE: {best_test_mse_ab}, RMSE: {best_test_rmse_ab}, MAE: {best_test_mae_ab}, rot_MSE: {best_test_r_mse_ab}, rot_RMSE: {best_test_r_rmse_ab}, "
            "rot_MAE: {best_test_r_mae_ab}, trans_MSE: {best_test_t_mse_ab}, trans_RMSE: {best_test_t_rmse_ab}, trans_MAE: {best_test_t_mae_ab}"
        )

        textio.cprint("B--------->A")
        textio.cprint(
            f"EPOCH:: {epoch}, Loss: {best_test_loss}, Cycle Loss: {best_test_mse_ba}, MSE: {best_test_rmse_ba}, RMSE: {best_test_rmse_ba}, MAE: {best_test_mae_ba}, rot_MSE: {best_test_r_mse_ba}, rot_RMSE: {best_test_r_rmse_ba}, "
            "rot_MAE: {best_test_r_mae_ba}, trans_MSE: {best_test_t_mse_ba}, trans_RMSE: {best_test_t_rmse_ba}, trans_MAE: {best_test_t_mae_ba}"
        )

        boardio.add_scalar("A->B/train/loss", train_loss, epoch)
        boardio.add_scalar("A->B/train/MSE", train_mse_ab, epoch)
        boardio.add_scalar("A->B/train/RMSE", train_rmse_ab, epoch)
        boardio.add_scalar("A->B/train/MAE", train_mae_ab, epoch)
        boardio.add_scalar("A->B/train/rotation/MSE", train_r_mse_ab, epoch)
        boardio.add_scalar("A->B/train/rotation/RMSE", train_r_rmse_ab, epoch)
        boardio.add_scalar("A->B/train/rotation/MAE", train_r_mae_ab, epoch)
        boardio.add_scalar("A->B/train/translation/MSE", train_t_mse_ab, epoch)
        boardio.add_scalar("A->B/train/translation/RMSE", train_t_rmse_ab, epoch)
        boardio.add_scalar("A->B/train/translation/MAE", train_t_mae_ab, epoch)

        boardio.add_scalar("B->A/train/loss", train_loss, epoch)
        boardio.add_scalar("B->A/train/MSE", train_mse_ba, epoch)
        boardio.add_scalar("B->A/train/RMSE", train_rmse_ba, epoch)
        boardio.add_scalar("B->A/train/MAE", train_mae_ba, epoch)
        boardio.add_scalar("B->A/train/rotation/MSE", train_r_mse_ba, epoch)
        boardio.add_scalar("B->A/train/rotation/RMSE", train_r_rmse_ba, epoch)
        boardio.add_scalar("B->A/train/rotation/MAE", train_r_mae_ba, epoch)
        boardio.add_scalar("B->A/train/translation/MSE", train_t_mse_ba, epoch)
        boardio.add_scalar("B->A/train/translation/RMSE", train_t_rmse_ba, epoch)
        boardio.add_scalar("B->A/train/translation/MAE", train_t_mae_ba, epoch)

        ############TEST
        boardio.add_scalar("A->B/test/loss", test_loss, epoch)
        boardio.add_scalar("A->B/test/MSE", test_mse_ab, epoch)
        boardio.add_scalar("A->B/test/RMSE", test_rmse_ab, epoch)
        boardio.add_scalar("A->B/test/MAE", test_mae_ab, epoch)
        boardio.add_scalar("A->B/test/rotation/MSE", test_r_mse_ab, epoch)
        boardio.add_scalar("A->B/test/rotation/RMSE", test_r_rmse_ab, epoch)
        boardio.add_scalar("A->B/test/rotation/MAE", test_r_mae_ab, epoch)
        boardio.add_scalar("A->B/test/translation/MSE", test_t_mse_ab, epoch)
        boardio.add_scalar("A->B/test/translation/RMSE", test_t_rmse_ab, epoch)
        boardio.add_scalar("A->B/test/translation/MAE", test_t_mae_ab, epoch)

        boardio.add_scalar("B->A/test/loss", test_loss, epoch)
        boardio.add_scalar("B->A/test/MSE", test_mse_ba, epoch)
        boardio.add_scalar("B->A/test/RMSE", test_rmse_ba, epoch)
        boardio.add_scalar("B->A/test/MAE", test_mae_ba, epoch)
        boardio.add_scalar("B->A/test/rotation/MSE", test_r_mse_ba, epoch)
        boardio.add_scalar("B->A/test/rotation/RMSE", test_r_rmse_ba, epoch)
        boardio.add_scalar("B->A/test/rotation/MAE", test_r_mae_ba, epoch)
        boardio.add_scalar("B->A/test/translation/MSE", test_t_mse_ba, epoch)
        boardio.add_scalar("B->A/test/translation/RMSE", test_t_rmse_ba, epoch)
        boardio.add_scalar("B->A/test/translation/MAE", test_t_mae_ba, epoch)

        ############BEST TEST
        boardio.add_scalar("A->B/best_test/loss", best_test_loss, epoch)
        boardio.add_scalar("A->B/best_test/MSE", best_test_mse_ab, epoch)
        boardio.add_scalar("A->B/best_test/RMSE", best_test_rmse_ab, epoch)
        boardio.add_scalar("A->B/best_test/MAE", best_test_mae_ab, epoch)
        boardio.add_scalar("A->B/best_test/rotation/MSE", best_test_r_mse_ab, epoch)
        boardio.add_scalar("A->B/best_test/rotation/RMSE", best_test_r_rmse_ab, epoch)
        boardio.add_scalar("A->B/best_test/rotation/MAE", best_test_r_mae_ab, epoch)
        boardio.add_scalar("A->B/best_test/translation/MSE", best_test_t_mse_ab, epoch)
        boardio.add_scalar("A->B/best_test/translation/RMSE", best_test_t_rmse_ab, epoch)
        boardio.add_scalar("A->B/best_test/translation/MAE", best_test_t_mae_ab, epoch)

        boardio.add_scalar("B->A/best_test/loss", best_test_loss, epoch)
        boardio.add_scalar("B->A/best_test/MSE", best_test_mse_ba, epoch)
        boardio.add_scalar("B->A/best_test/RMSE", best_test_rmse_ba, epoch)
        boardio.add_scalar("B->A/best_test/MAE", best_test_mae_ba, epoch)
        boardio.add_scalar("B->A/best_test/rotation/MSE", best_test_r_mse_ba, epoch)
        boardio.add_scalar("B->A/best_test/rotation/RMSE", best_test_r_rmse_ba, epoch)
        boardio.add_scalar("B->A/best_test/rotation/MAE", best_test_r_mae_ba, epoch)
        boardio.add_scalar("B->A/best_test/translation/MSE", best_test_t_mse_ba, epoch)
        boardio.add_scalar("B->A/best_test/translation/RMSE", best_test_t_rmse_ba, epoch)
        boardio.add_scalar("B->A/best_test/translation/MAE", best_test_t_mae_ba, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(
                net.module.state_dict(),
                "checkpoints/%s/models/model.%d.t7" % (args.exp_name, epoch),
            )
        else:
            torch.save(
                net.state_dict(),
                "checkpoints/%s/models/model.%d.t7" % (args.exp_name, epoch),
            )
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Point Cloud Registration")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp",
        metavar="N",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dcp",
        metavar="N",
        choices=["dcp"],
        help="Model to use, [dcp]",
    )
    parser.add_argument(
        "--emb_nn",
        type=str,
        default="pointnet",
        metavar="N",
        choices=["pointnet", "dgcnn"],
        help="Embedding nn to use, [pointnet, dgcnn]",
    )
    parser.add_argument(
        "--pointer",
        type=str,
        default="transformer",
        metavar="N",
        choices=["identity", "transformer"],
        help="Attention-based pointer generator to use, [identity, transformer]",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="svd",
        metavar="N",
        choices=[
            "mlp",
            "svd",
        ],
        help="Head to use, [mlp, svd]",
    )
    parser.add_argument("--emb_dims", type=int, default=512, metavar="N", help="Dimension of embeddings")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=1,
        metavar="N",
        help="Num of blocks of encoder&decoder",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        metavar="N",
        help="Num of heads in multiheadedattention",
    )
    parser.add_argument(
        "--ff_dims",
        type=int,
        default=1024,
        metavar="N",
        help="Num of dimensions of fc in transformer",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        metavar="N",
        help="Dropout ratio in transformer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="batch_size",
        help="Size of batch)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10,
        metavar="batch_size",
        help="Size of batch)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        metavar="N",
        help="number of episode to train ",
    )
    parser.add_argument("--use_sgd", action="store_true", default=False, help="Use SGD")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001, 0.1 if using sgd)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument("--no_cuda", action="store_true", default=False, help="enables CUDA training")
    parser.add_argument("--seed", type=int, default=1234, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--eval", action="store_true", default=False, help="evaluate the model")
    parser.add_argument(
        "--cycle",
        type=bool,
        default=False,
        metavar="N",
        help="Whether to use cycle consistency",
    )
    parser.add_argument(
        "--gaussian_noise",
        type=bool,
        default=False,
        metavar="N",
        help="Wheter to add gaussian noise",
    )
    parser.add_argument(
        "--unseen",
        type=bool,
        default=False,
        metavar="N",
        help="Wheter to test on unseen category",
    )
    parser.add_argument("--num_points", type=int, default=1024, metavar="N", help="Num of points to use")
    parser.add_argument(
        "--dataset",
        type=str,
        default="smpl_synthetic",
        choices=["modelnet40, smpl_synthetic"],
        metavar="N",
        help="dataset to use",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=4,
        metavar="N",
        help="Divided factor for rotations",
    )
    parser.add_argument("--model_path", type=str, default="", metavar="N", help="Pretrained model path")

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir="checkpoints/" + args.exp_name)
    _init_(args)

    textio = IOStream("checkpoints/" + args.exp_name + "/run.log")
    textio.cprint(str(args))

    if args.dataset == "modelnet40":
        train_loader = DataLoader(
            ModelNet40(
                num_points=args.num_points,
                partition="train",
                gaussian_noise=args.gaussian_noise,
                unseen=args.unseen,
                factor=args.factor,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            ModelNet40(
                num_points=args.num_points,
                partition="test",
                gaussian_noise=args.gaussian_noise,
                unseen=args.unseen,
                factor=args.factor,
            ),
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
        )
    elif args.dataset == "smpl_synthetic":
        train_loader = DataLoader(
            SmplSynthetic(split='train', num_output_points=args.num_points, transform=SMPLAugmentation(glasses_probability=0.5)),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            SmplSynthetic(split='test', num_output_points=args.num_points, transform=SMPLAugmentation(glasses_probability=0.5)),
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False,
        )
    else:
        raise NotImplementedError("not implemented")

    if args.model == "dcp":
        net = DCP(args).cuda()
        if args.eval:
            if args.model_path == "":
                model_path = ("checkpoints" + "/" + args.exp_name + "/models/model.best.t7")
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise NotImplementedError("Not implemented")

    train(args, net, train_loader, test_loader, boardio, textio)

    print("FINISH")
    boardio.close()


if __name__ == "__main__":
    main()