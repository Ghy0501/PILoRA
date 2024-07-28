import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def l2_loss(a):
    l2_dis = torch.pow(a, 2).sum()
    return l2_dis / 2

def pl_loss(distance, labels):
    loss_r = torch.index_select(distance, dim=1, index=labels)
    loss_r = torch.diagonal(loss_r)
    loss_r = torch.mean(loss_r)
    # batch_centers = torch.index_select(centers.transpose(0, 1), 0, labels)
    # loss_r = F.mse_loss(features, batch_centers) / 2
    return loss_r

def compute_distance(x, centers):
    features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
    centers_square = torch.sum(torch.pow(centers, 2), 0, keepdim=True)
    features_into_centers = 2 * torch.matmul(x, (centers))
    dist = features_square + centers_square - features_into_centers
    dist = dist / float(x.shape[1])
    return dist

def DCE(labels, distance, T=1.0, type='train'):
    # if type == 'test':
    #     distance = compute_distance(features, centers)
    # else:
    #     distance = dist
    logits = F.softmax(-distance, dim=1)
    if labels is None: return logits
    loss = F.cross_entropy(-distance / T, labels)
    loss_r = pl_loss(distance, labels)
    loss = loss + 0.001*loss_r
    return logits, loss