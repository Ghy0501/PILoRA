#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from CPN import *
from sklearn.metrics.pairwise import cosine_similarity


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        _, image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size=self.args.local_bs, shuffle=True, num_workers=8)

    def update_weights(self, model, old_model, lr_c, lr_e, current_task, Waq, Wav):
        model.train()

        pg = [p for p in model.parameters() if p.requires_grad]
        train_name = [name for name, param in model.named_parameters() if param.requires_grad]
        network_params = []
        for name, param in model.named_parameters():
            lr = lr_c 
            if name.startswith('encoder'):
                lr = lr_e 
            elif name.startswith('centers'):
                lr = lr_c 
            
            param_group = {'params': [param], 'lr': lr, 'weight_decay': 0.00001}
            network_params.append(param_group)

        optimizer = torch.optim.Adam(network_params)
        label = []
        all_classes = set(range(10)) 
        label_feature_dict = {cls: [] for cls in all_classes}
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                label.extend(labels.tolist())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                model.zero_grad()
                pred, feature, dist, centers, feat_layer4 = model(images)
                logits, loss_dce = DCE(labels, dist)

                l1_loss = torch.tensor(0., dtype=torch.float32, device=self.args.device)  
                ort_loss = torch.tensor(0., dtype=torch.float32, device=self.args.device)  

                for name, param in model.named_parameters():
                    if 'linear_a_q_{}'.format(current_task) in name:
                        l1_loss += torch.norm(param, p=1)
                    if 'linear_a_v_{}'.format(current_task) in name:
                        l1_loss += torch.norm(param, p=1)

                
                if old_model is None:
                    loss = loss_dce + 0.01 * l1_loss
                else:
                    for name, param in model.named_parameters():
                        if 'linear_a_q_{}'.format(current_task) in name:
                            for pre_param in Waq:
                                ort_loss += torch.abs(torch.mm(pre_param, param.T)).sum()
                        if 'linear_a_v_{}'.format(current_task) in name:
                            for pre_param in Wav:
                                ort_loss += torch.abs(torch.mm(pre_param, param.T)).sum()
                    loss = loss_dce + 0.01 * l1_loss + 0.5 * ort_loss 
                
                loss.backward()
                optimizer.step()

                for i, lbl in enumerate(labels):
                    lbl = lbl.item()  
                    label_feature_dict[lbl%10].append(feature[i].detach().cpu().numpy())
        averages = {}
        for cls, values in label_feature_dict.items():
            if not values:  
                averages[cls] = []  
            else:  
                average_value = sum(values) / len(values)
                averages[cls] = average_value
        return model.state_dict(), averages
