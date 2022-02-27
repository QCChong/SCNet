import torch
from torch import nn
from loss.emd import emd_module as emd
from loss.chamfer.champfer_loss import ChamferLoss
from utils.pointnet_utils import gather_points, group_points, farthest_point_sample
from typing import Optional
from knn_cuda import KNN

def fscore(dist1, dist2, threshold=0.0001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    dist1: Batch, N-Points
    dist2: Batch, N-Points
    return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

def calc_cd(output, gt, calc_f1=False):
    chamfer_loss = ChamferLoss()
    dist1, dist2 = chamfer_loss(output, gt)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = dist1.mean(1) + dist2.mean(1)
    if calc_f1:
        d, _ = knnSearch(gt, gt, 2)
        avg_d, _ = torch.min(d[:,:,1], dim=-1, keepdim=True)
        f1, _, _ = fscore(dist1, dist2, avg_d*avg_d)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t

def calc_cd_single_side(output, gt):
    chamfer_loss = ChamferLoss()
    dist1, dist2 = chamfer_loss(output, gt)
    cd_p = torch.sqrt(dist1).mean(1)
    return cd_p

def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, matches = emd_loss(output, gt, eps, iterations)
    emd_p = torch.sqrt(dist).mean(1)
    return emd_p

def knnSearch(point_ref, point_query, k):
    """
    point_ref:   (B, N, C)
    point_query: (B, M, C)
    k: the number of neighbor points
    dist, idx ->  [B,M,K], [B,M,K]
    """
    knn = KNN(k, transpose_mode=True)
    dist, idx = knn(point_ref, point_query)
    return dist, idx

def MLP(channels, batch_norm = False, bias = True, conv_type = '1D', act:Optional=nn.ReLU, last_act=True):
    conv_bn = {'1D':[nn.Conv1d, nn.BatchNorm1d, (1,)],
               '2D':[nn.Conv2d, nn.BatchNorm2d, (1,1)],
               'linear':[nn.Linear, nn.BatchNorm1d, None]}

    conv, bn, kernel = conv_bn[conv_type]
    n = len(channels)
    nn_list = []
    for i in range(1, n):
        if kernel:
            nn_list.append(conv(channels[i-1], channels[i], kernel, bias=bias))
        else:
            nn_list.append(conv(channels[i-1], channels[i], bias=bias))

        if i!=n-1 or last_act:
            if batch_norm:
                nn_list.append(bn(channels[i]))
            nn_list.append(act(inplace=True))

    return nn.Sequential(*nn_list)

def fps(xyz, npoints, BNC=True):
    if BNC:
        idx = farthest_point_sample(xyz, npoints)
        xyz_new = gather_points(xyz.transpose(2, 1).contiguous(), idx)  # (B, 3, npoints)
        xyz_new = xyz_new.transpose(2, 1).contiguous()  # (B, npoints, 3)
    else:
        idx = farthest_point_sample(xyz.transpose(2, 1).contiguous(), npoints)
        xyz_new = gather_points(xyz, idx)  # (B, 3, npoints)

    return xyz_new
