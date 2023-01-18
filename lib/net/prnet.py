import os
import sys
import glob
import h5py
import copy
import math
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import open3d as o3d


from lib.utils.util import transform_point_cloud, quat2mat, npmat2euler
from utils.vis import visualize_transformation, visualize_pred_transformation

# this code is referred from: https://github.com/WangYueFt/prnet


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def pairwise_distance(src, tgt):
    inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), tgt)
    xx = torch.sum(src**2, dim=1, keepdim=True)
    yy = torch.sum(tgt**2, dim=1, keepdim=True)
    distances = xx.transpose(2, 1).contiguous() + inner + yy
    return torch.sqrt(distances)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    x = x.view(*x.size()[:3])
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device("cuda")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature


def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity = (
        torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(
        translation_ab, -translation_ba
    )


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(
            self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        )


class Generator(nn.Module):
    def __init__(self, n_emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_emb_dims, n_emb_dims // 2),
            nn.BatchNorm1d(n_emb_dims // 2),
            nn.ReLU(),
            nn.Linear(n_emb_dims // 2, n_emb_dims // 4),
            nn.BatchNorm1d(n_emb_dims // 4),
            nn.ReLU(),
            nn.Linear(n_emb_dims // 4, n_emb_dims // 8),
            nn.BatchNorm1d(n_emb_dims // 8),
            nn.ReLU(),
        )
        self.proj_rot = nn.Linear(n_emb_dims // 8, 4)
        self.proj_trans = nn.Linear(n_emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x), negative_slope=0.2)))


class PointNet(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(n_emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x3)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2).view(
            batch_size, -1, num_points
        )
        return x


class MLPHead(nn.Module):
    def __init__(self, cfg):
        super(MLPHead, self).__init__()
        n_emb_dims = cfg.NET.EMB_DIMS
        self.n_emb_dims = n_emb_dims
        self.nn = nn.Sequential(
            nn.Linear(n_emb_dims * 2, n_emb_dims // 2),
            nn.BatchNorm1d(n_emb_dims // 2),
            nn.ReLU(),
            nn.Linear(n_emb_dims // 2, n_emb_dims // 4),
            nn.BatchNorm1d(n_emb_dims // 4),
            nn.ReLU(),
            nn.Linear(n_emb_dims // 4, n_emb_dims // 8),
            nn.BatchNorm1d(n_emb_dims // 8),
            nn.ReLU(),
        )
        self.proj_rot = nn.Linear(n_emb_dims // 8, 4)
        self.proj_trans = nn.Linear(n_emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.n_emb_dims = cfg.NET.EMB_DIMS
        self.N = cfg.NET.N_BLOCKS
        self.dropout = cfg.NET.DROUPOUT
        self.n_ff_dims = cfg.NET.FF_DIMS
        self.n_heads = cfg.NET.N_HEADS
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.model = EncoderDecoder(
            Encoder(
                EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N
            ),
            Decoder(
                DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout),
                self.N,
            ),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
        )

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class TemperatureNet(nn.Module):
    def __init__(self, cfg):
        super(TemperatureNet, self).__init__()
        self.n_emb_dims = cfg.NET.EMB_DIMS
        self.temp_factor = cfg.NET.TEMP_FACTOR
        self.nn = nn.Sequential(
            nn.Linear(self.n_emb_dims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )
        self.feature_disparity = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src_embedding = src_embedding.mean(dim=2)
        tgt_embedding = tgt_embedding.mean(dim=2)
        residual = torch.abs(src_embedding - tgt_embedding)

        self.feature_disparity = residual

        return (
            torch.clamp(
                self.nn(residual), 1.0 / self.temp_factor, 1.0 * self.temp_factor
            ),
            residual,
        )


class SVDHead(nn.Module):
    def __init__(self, cfg):
        super(SVDHead, self).__init__()
        self.n_emb_dims = cfg.NET.EMB_DIMS
        self.cat_sampler = cfg.NET.CAT_SAMPLER
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.temperature = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.my_iter = torch.ones(1)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size, num_dims, num_points = src.size()
        temperature = input[4].view(batch_size, 1, 1)

        if self.cat_sampler == "softmax":
            d_k = src_embedding.size(1)
            scores = torch.matmul(
                src_embedding.transpose(2, 1).contiguous(), tgt_embedding
            ) / math.sqrt(d_k)
            scores = torch.softmax(temperature * scores, dim=2)
        elif self.cat_sampler == "gumbel_softmax":
            d_k = src_embedding.size(1)
            scores = torch.matmul(
                src_embedding.transpose(2, 1).contiguous(), tgt_embedding
            ) / math.sqrt(d_k)
            scores = scores.view(batch_size * num_points, num_points)
            temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
            scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
            scores = scores.view(batch_size, num_points, num_points)
        else:
            raise Exception("not implemented")

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(
            src_centered, src_corr_centered.transpose(2, 1).contiguous()
        ).cpu()

        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(
                np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, r_det]]).astype("float32")
            ).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)

        R = torch.stack(R, dim=0).cuda()

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(
            dim=2, keepdim=True
        )
        if self.training:
            self.my_iter += 1
        return R, t.view(batch_size, 3)


class KeyPointNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeyPointNet, self).__init__()
        self.num_keypoints = num_keypoints

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        batch_size, num_dims, num_points = src_embedding.size()
        src_norm = torch.norm(src_embedding, dim=1, keepdim=True)
        tgt_norm = torch.norm(tgt_embedding, dim=1, keepdim=True)
        src_topk_idx = torch.topk(src_norm, k=self.num_keypoints, dim=2, sorted=False)[
            1
        ]
        tgt_topk_idx = torch.topk(tgt_norm, k=self.num_keypoints, dim=2, sorted=False)[
            1
        ]
        src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
        tgt_keypoints_idx = tgt_topk_idx.repeat(1, 3, 1)
        src_embedding_idx = src_topk_idx.repeat(1, num_dims, 1)
        tgt_embedding_idx = tgt_topk_idx.repeat(1, num_dims, 1)

        src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
        tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)

        src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
        tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
        return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding


class ACPNet(nn.Module):
    def __init__(self, cfg, args):
        super(ACPNet, self).__init__()
        self.n_emb_dims = cfg.NET.EMB_DIMS
        self.num_keypoints = cfg.NET.N_KEYPOINTS
        self.num_subsampled_points = cfg.NET.N_SUBSAMPLED_POINTS
        self.logger = Logger(args)
        if cfg.NET.EMB_NN == "pointnet":
            self.emb_nn = PointNet(n_emb_dims=self.n_emb_dims)
        elif cfg.NET.EMB_NN == "dgcnn":
            self.emb_nn = DGCNN(n_emb_dims=self.n_emb_dims)
        else:
            raise Exception("Not implemented")

        if cfg.NET.POINTER == "identity":
            self.attention = Identity()
        elif cfg.NET.POINTER == "transformer":
            self.attention = Transformer(cfg=cfg)
        else:
            raise Exception("Not implemented")

        self.temp_net = TemperatureNet(cfg)

        if cfg.NET.HEAD == "mlp":
            self.head = MLPHead(cfg=cfg)
        elif cfg.NET.HEAD == "svd":
            self.head = SVDHead(cfg=cfg)
        else:
            raise Exception("Not implemented")

        if self.num_keypoints != self.num_subsampled_points:
            self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()

    def forward(self, *input):
        (
            src,
            tgt,
            src_embedding,
            tgt_embedding,
            temperature,
            feature_disparity,
        ) = self.predict_embedding(*input)
        rotation_ab, translation_ab = self.head(
            src_embedding, tgt_embedding, src, tgt, temperature
        )
        rotation_ba, translation_ba = self.head(
            tgt_embedding, src_embedding, tgt, src, temperature
        )
        return (
            rotation_ab,
            translation_ab,
            rotation_ba,
            translation_ba,
            feature_disparity,
        )

    def predict_embedding(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        src, tgt, src_embedding, tgt_embedding = self.keypointnet(
            src, tgt, src_embedding, tgt_embedding
        )

        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)

        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(
            *input
        )
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(
            src_embedding.transpose(2, 1).contiguous(), tgt_embedding
        ) / math.sqrt(d_k)
        scores = scores.view(batch_size * num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores


class PRNet(nn.Module):
    def __init__(self, cfg, args):
        super(PRNet, self).__init__()
        self.num_iters = cfg.NET.N_ITERS
        self.logger = Logger(args)
        self.discount_factor = cfg.NET.DISCOUNT_FACTOR
        self.acpnet = ACPNet(cfg, args)
        self.model_path = cfg.NET.MODEL_PATH
        self.feature_alignment_loss = cfg.NET.FEATURE_ALIGNMENT_LOSS
        self.cycle_consistency_loss = cfg.NET.CYCLE_CONSISTENCY_LOSS

        if self.model_path != "":
            self.load(self.model_path)
        if torch.cuda.device_count() > 1:
            self.acpnet = nn.DataParallel(self.acpnet)

    def forward(self, *input):
        (
            rotation_ab,
            translation_ab,
            rotation_ba,
            translation_ba,
            feature_disparity,
        ) = self.acpnet(*input)
        return (
            rotation_ab,
            translation_ab,
            rotation_ba,
            translation_ba,
            feature_disparity,
        )

    def predict(self, src, tgt, n_iters=3):
        batch_size = src.size(0)
        rotation_ab_pred = (
            torch.eye(3, device=src.device, dtype=torch.float32)
            .view(1, 3, 3)
            .repeat(batch_size, 1, 1)
        )
        translation_ab_pred = (
            torch.zeros(3, device=src.device, dtype=torch.float32)
            .view(1, 3)
            .repeat(batch_size, 1)
        )
        for _ in range(n_iters):
            (
                rotation_ab_pred_i,
                translation_ab_pred_i,
                rotation_ba_pred_i,
                translation_ba_pred_i,
                _,
            ) = self.forward(src, tgt)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = (
                torch.matmul(
                    rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)
                ).squeeze(2)
                + translation_ab_pred_i
            )
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        return rotation_ab_pred, translation_ab_pred

    def _one_batch(self, src, tgt, rotation_ab, translation_ab, opt, is_train):
        if is_train:
            opt.zero_grad()
        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = (
            torch.eye(3, device=src.device, dtype=torch.float32)
            .view(1, 3, 3)
            .repeat(batch_size, 1, 1)
        )
        translation_ab_pred = (
            torch.zeros(3, device=src.device, dtype=torch.float32)
            .view(1, 3)
            .repeat(batch_size, 1)
        )

        rotation_ba_pred = (
            torch.eye(3, device=src.device, dtype=torch.float32)
            .view(1, 3, 3)
            .repeat(batch_size, 1, 1)
        )
        translation_ba_pred = (
            torch.zeros(3, device=src.device, dtype=torch.float32)
            .view(1, 3)
            .repeat(batch_size, 1)
        )

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0
        for i in range(self.num_iters):
            (
                rotation_ab_pred_i,
                translation_ab_pred_i,
                rotation_ba_pred_i,
                translation_ba_pred_i,
                feature_disparity,
            ) = self.forward(src, tgt)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = (
                torch.matmul(
                    rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)
                ).squeeze(2)
                + translation_ab_pred_i
            )

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = (
                torch.matmul(
                    rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)
                ).squeeze(2)
                + translation_ba_pred_i
            )

            loss = (
                F.mse_loss(
                    torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab),
                    identity,
                )
                + F.mse_loss(translation_ab_pred, translation_ab)
            ) * self.discount_factor**i
            feature_alignment_loss = (
                feature_disparity.mean()
                * self.feature_alignment_loss
                * self.discount_factor**i
            )
            cycle_consistency_loss = (
                cycle_consistency(
                    rotation_ab_pred_i,
                    translation_ab_pred_i,
                    rotation_ba_pred_i,
                    translation_ba_pred_i,
                )
                * self.cycle_consistency_loss
                * self.discount_factor**i
            )
            scale_consensus_loss = 0
            total_feature_alignment_loss += feature_alignment_loss
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss += loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
            
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
            
        if is_train:
            total_loss.backward()
            opt.step()
            
        return (
            total_loss.item(),
            total_feature_alignment_loss.item(),
            total_cycle_consistency_loss.item(),
            total_scale_consensus_loss,
            rotation_ab_pred,
            translation_ab_pred,
        )

    def _one_epoch(self, epoch, data_loader, boardio = None, opt = None, is_train = False):
        if is_train:
            self.train()
        else:
            self.eval()

        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []
        num_examples = 0
        total_feature_alignment_loss = 0.0
        total_cycle_consistency_loss = 0.0
        total_scale_consensus_loss = 0.0
        for it, data in enumerate(tqdm(data_loader, total=len(data_loader))):
            (
                source,
                target,
                rotation_ab,
                translation_ab,
                rotation_ba,
                translation_ba,
                euler_ab,
                euler_ba,
            ) = [d.cuda() for d in data]
            (
                loss,
                feature_alignment_loss,
                cycle_consistency_loss,
                scale_consensus_loss,
                rotation_ab_pred,
                translation_ab_pred,
            ) = self._one_batch(
                source, target, rotation_ab, translation_ab, opt, is_train=is_train
            )
            batch_size = source.size(0)
            num_examples += batch_size
            total_loss += loss * batch_size
            total_feature_alignment_loss += feature_alignment_loss * batch_size
            
            total_cycle_consistency_loss += cycle_consistency_loss * batch_size
            
            total_scale_consensus_loss += scale_consensus_loss * batch_size
            

            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.cpu().numpy())

            if it == len(data_loader) // 10:
                pcds = visualize_pred_transformation(
                    source.cpu().numpy(),
                    target.cpu().numpy(),
                    rotation_ab_pred.detach().cpu().numpy(),
                    translation_ab_pred.detach().cpu().numpy(),
                )
                for jk, pcd in enumerate(pcds):
                    if not os.path.exists("output_debug"):
                        os.mkdir("output_debug")

                    o3d.io.write_point_cloud(
                        os.path.join("output_debug", f"pointclouds_{jk}.ply"), pcd
                    )

        avg_loss = total_loss / num_examples
        avg_feature_alignment_loss = total_feature_alignment_loss / num_examples
        avg_cycle_consistency_loss = total_cycle_consistency_loss / num_examples
        avg_scale_consensus_loss = total_scale_consensus_loss / num_examples

        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.degrees(np.concatenate(eulers_ab, axis=0))
        eulers_ab_pred = npmat2euler(rotations_ab_pred)
        r_ab_mse = np.mean((eulers_ab - eulers_ab_pred) ** 2)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(np.abs(eulers_ab - eulers_ab_pred))
        t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(np.abs(translations_ab - translations_ab_pred))
        r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)

        stage = "train" if is_train else "test"

        info = {
            "arrow": "A->B",
            "epoch": epoch,
            "stage": stage,
            "loss": avg_loss,
            "feature_alignment_loss": avg_feature_alignment_loss,
            "cycle_consistency_loss": avg_cycle_consistency_loss,
            "scale_consensus_loss": avg_scale_consensus_loss,
            "r_ab_mse": r_ab_mse,
            "r_ab_rmse": r_ab_rmse,
            "r_ab_mae": r_ab_mae,
            "t_ab_mse": t_ab_mse,
            "t_ab_rmse": t_ab_rmse,
            "t_ab_mae": t_ab_mae,
            "r_ab_r2_score": r_ab_r2_score,
            "t_ab_r2_score": t_ab_r2_score,
        }
        self.logger.write(info)

        # tensorboard stuff
        boardio.add_scalar(f"A->B/{stage}/loss", avg_loss, epoch)
        boardio.add_scalar(f"A->B/{stage}/rotation/MSE", r_ab_mse, epoch)
        boardio.add_scalar(f"A->B/{stage}/rotation/RMSE", r_ab_rmse, epoch)
        boardio.add_scalar(f"A->B/{stage}/rotation/MAE", r_ab_mae, epoch)
        boardio.add_scalar(f"A->B/{stage}/translation/MSE", t_ab_mse, epoch)
        boardio.add_scalar(f"A->B/{stage}/translation/RMSE", t_ab_rmse, epoch)
        boardio.add_scalar(f"A->B/{stage}/translation/MAE", t_ab_mae, epoch)

        return info

    def save(self, path):
        if torch.cuda.device_count() > 1:
            torch.save(self.acpnet.module.state_dict(), path)
        else:
            torch.save(self.acpnet.state_dict(), path)

    def load(self, path):
        self.acpnet.load_state_dict(torch.load(path))


class Logger:
    def __init__(self, args):
        self.path = "checkpoints/" + args.exp_name
        self.fw = open(self.path + "/log", "a")
        self.fw.flush()
        with open(os.path.join(self.path, "cfg.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    def write(self, info):
        arrow = info["arrow"]
        epoch = info["epoch"]
        stage = info["stage"]
        loss = info["loss"]
        feature_alignment_loss = info["feature_alignment_loss"]
        cycle_consistency_loss = info["cycle_consistency_loss"]
        scale_consensus_loss = info["scale_consensus_loss"]
        r_ab_mse = info["r_ab_mse"]
        r_ab_rmse = info["r_ab_rmse"]
        r_ab_mae = info["r_ab_mae"]
        t_ab_mse = info["t_ab_mse"]
        t_ab_rmse = info["t_ab_rmse"]
        t_ab_mae = info["t_ab_mae"]
        r_ab_r2_score = info["r_ab_r2_score"]
        t_ab_r2_score = info["t_ab_r2_score"]
        text = (
            f"{arrow}:: Stage: {stage}, Epoch: {epoch}, Loss: {loss}, Feature_alignment_loss: {feature_alignment_loss}, Cycle_consistency_loss: {cycle_consistency_loss}, "
            f"Scale_consensus_loss: {scale_consensus_loss}, Rot_MSE: {r_ab_mse}, Rot_RMSE: {r_ab_rmse}, "
            f"Rot_MAE: {r_ab_mae}, Rot_R2: {r_ab_r2_score}, Trans_MSE: {t_ab_mse}, "
            f"Trans_RMSE: {t_ab_rmse}, Trans_MAE: {t_ab_mae}, Trans_R2: {t_ab_r2_score}\n"
        )

        self.fw.write(text)
        self.fw.flush()
        print(text)

    def close(self):
        self.fw.close()
