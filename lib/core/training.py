import gc
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import __init_paths__
from lib.core.function_dcp import one_epoch, evaluate_real_data
from lib.utils.util import npmat2euler



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
    scheduler = MultiStepLR(opt, milestones=[10, 20, 30], gamma=0.1)

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

        with torch.no_grad():
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
        _ = net._one_epoch(epoch=epoch, data_loader=train_loader, opt=opt, boardio=boardio, is_train = True)
        
        # this is absurd.... eval takes more memory than training as there are gradients computed regardless.... therefore deactivate it like this
        if not cfg.TRAINING.OVERFIT:
            with torch.no_grad():
                info_test = net._one_epoch(epoch=epoch, data_loader=test_loader, boardio=boardio, is_train = False)
                evaluate_real_data(net)

        scheduler.step()
        if not cfg.TRAINING.OVERFIT:
            if info_test_best is None or info_test_best['loss'] > info_test['loss']:
                info_test_best = info_test
                info_test_best['stage'] = 'best_test'

                net.save('checkpoints/%s/models/model.best.t7' % args.exp_name)
            net.logger.write(info_test_best)

        net.save('checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()
        
class IOStream:

    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()