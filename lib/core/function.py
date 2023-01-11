import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from utils.util import transform_point_cloud, npmat2euler
from utils.vis import visualize_transformation


def train_one_epoch(args, net, train_loader, opt, boardio, epoch):
    net.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for it, (
            src,
            target,
            rotation_ab,
            translation_ab,
            rotation_ba,
            translation_ba,
            euler_ab,
            euler_ba,
    ) in tqdm(enumerate(train_loader)):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        (rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred) = net(src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) + F.mse_loss(
            translation_ab_pred, translation_ab)
        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean(
                (torch.matmul(
                    rotation_ba_pred.transpose(2, 1),
                    translation_ab_pred.view(batch_size, 3, 1),
                ).view(batch_size, 3) + translation_ba_pred)**2,
                dim=[0, 1],
            )
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

        mse_ab += (torch.mean((transformed_src - target)**2, dim=[0, 1, 2]).item() * batch_size)
        mae_ab += (torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size)

        mse_ba += (torch.mean((transformed_target - src)**2, dim=[0, 1, 2]).item() * batch_size)
        mae_ba += (torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size)

        if it % 9 == 0:
            fig = visualize_transformation(src.cpu().numpy(),
                                           target.cpu().numpy(),
                                           rotation_ab.cpu().numpy(),
                                           translation_ab.cpu().numpy(),
                                           rotation_ab_pred.detach().cpu().numpy(),
                                           translation_ab_pred.detach().cpu().numpy(),
                                           )
            boardio.add_figure('predictions', fig, global_step=(epoch + 1) * (it + 1))

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return (
        total_loss * 1.0 / num_examples,
        total_cycle_loss / num_examples,
        mse_ab * 1.0 / num_examples,
        mae_ab * 1.0 / num_examples,
        mse_ba * 1.0 / num_examples,
        mae_ba * 1.0 / num_examples,
        rotations_ab,
        translations_ab,
        rotations_ab_pred,
        translations_ab_pred,
        rotations_ba,
        translations_ba,
        rotations_ba_pred,
        translations_ba_pred,
        eulers_ab,
        eulers_ba,
    )


def test_one_epoch(args, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for (
            src,
            target,
            rotation_ab,
            translation_ab,
            rotation_ba,
            translation_ba,
            euler_ab,
            euler_ba,
    ) in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        (
            rotation_ab_pred,
            translation_ab_pred,
            rotation_ba_pred,
            translation_ba_pred,
        ) = net(src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) + F.mse_loss(
            translation_ab_pred, translation_ab)
        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean(
                (torch.matmul(
                    rotation_ba_pred.transpose(2, 1),
                    translation_ab_pred.view(batch_size, 3, 1),
                ).view(batch_size, 3) + translation_ba_pred)**2,
                dim=[0, 1],
            )
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        total_loss += loss.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

        mse_ab += (torch.mean((transformed_src - target)**2, dim=[0, 1, 2]).item() * batch_size)
        mae_ab += (torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size)

        mse_ba += (torch.mean((transformed_target - src)**2, dim=[0, 1, 2]).item() * batch_size)
        mae_ba += (torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size)

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return (
        total_loss * 1.0 / num_examples,
        total_cycle_loss / num_examples,
        mse_ab * 1.0 / num_examples,
        mae_ab * 1.0 / num_examples,
        mse_ba * 1.0 / num_examples,
        mae_ba * 1.0 / num_examples,
        rotations_ab,
        translations_ab,
        rotations_ab_pred,
        translations_ab_pred,
        rotations_ba,
        translations_ba,
        rotations_ba_pred,
        translations_ba_pred,
        eulers_ab,
        eulers_ba,
    )