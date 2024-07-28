#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import collections
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from sampling import *
from iCIFAR100 import iCIFAR100
import random
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy.spatial.distance import cdist
from data_manager_tiny import *
import torch.nn.functional as F


def get_trainable_param_names(model):
    return [name for name, param in model.named_parameters() if param.requires_grad]

def get_frozen_param_names(model):
    return [name for name, param in model.named_parameters() if not param.requires_grad]

def build_continual_dataset(args, class_order):
    class_mask = list() if args.task_inc or args.train_mask else None

    # trans_test = transforms.Compose([
    #                         transforms.Resize(224),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #                     ])

    class_mask = split_single_dataset(args, class_order)

    # for classes in class_mask:
    #     class_task = [classes[0], classes[-1]+1]
    #     dataset_val = iCIFAR100('../../data', train=False, download=True,
    #                             test_transform=trans_test)
    #     dataset_val.getTestData(class_task)
    #     test_loader = DataLoader(dataset_val, batch_size=100,
    #                                 shuffle=False, num_workers=8)
    #     test_dataloader.append(test_loader)
    return class_mask


def get_trainand_test_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = '../../data'
    trans_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trans_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
    #                                transform=trans_train)

    # test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
    #                               transform=trans_test)     

    train_dataset = iCIFAR100(data_dir, train=True, download=True,
                                transform=trans_train)

    test_dataset = iCIFAR100(data_dir, train=False, download=True,
                                test_transform=trans_test)  
    all_classes = [0, args.total_classes]
    test_dataset.getTestData(all_classes)
    train_dataset.getTrainData(all_classes)
    return train_dataset, test_dataset

def get_dataset(args, trans_train, m, class_set, task_num):
    # sample training data amongst users
    data_manager = DataManager()
    if args.iid:
        current_class = random.sample([x for x in range(class_set[0], class_set[-1]+1)], task_num)
        train_dataset = data_manager.get_dataset(trans_train, index=current_class, train=True)
        user_groups = cifar_iid(train_dataset, m)
    else:
        if args.niid_type == "Q":
            current_class = random.sample([x for x in range(class_set[0], class_set[-1]+1)], task_num)
            train_dataset = data_manager.get_dataset(trans_train, index=current_class, train=True)
            user_groups = quantity_based_label_skew(train_dataset, m, alpha=args.alpha)
            # current_class = random.sample([x for x in range(start, end)], args.alpha)
            # train_dataset.getTrainData(current_class)
            # user_groups = cifar_iid(train_dataset, m)
        else:
            current_class = random.sample([x for x in range(class_set[0], class_set[-1]+1)], task_num)
            train_dataset = data_manager.get_dataset(trans_train, index=current_class, train=True)
            user_groups = distribution_based_label_skew(train_dataset, m, beta=args.beta)


    return train_dataset, user_groups

def split_single_dataset(args, class_order):
    nb_classes = args.total_classes
    assert nb_classes % (args.task_num+1) == 0
    classes_per_task = nb_classes // (args.task_num+1)

    labels = [i for i in range(nb_classes)]
    
    mask = list()

    # if args.shuffle:
    #     random.shuffle(labels)
    class_till_now = classes_per_task
    for _ in range(args.task_num+1):
        
        # scope = class_order[:class_till_now]
        # class_till_now += classes_per_task
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]

        mask.append(scope)
    
    return mask


def compute_weight(centers_list, feature_list):
    weight = []
    for idx in range(0, 20):
        non_empty_indices = [i for i, a_item in enumerate(feature_list[idx]) if len(a_item) > 0]
        if non_empty_indices != []:
            non_empty_a = [feature_list[idx][i] for i in non_empty_indices]
            non_empty_a = np.array(non_empty_a)
            non_empty_b = np.array(centers_list[idx])
            distances_matrix = cdist(non_empty_b, non_empty_a, metric='euclidean')

            total_distance = np.sum(distances_matrix, axis=1, keepdims=True)
            total_distance = [item for sublist in total_distance.tolist() for item in sublist]

            reciprocal_data = [1 / value for value in total_distance]

            min_val = np.min(reciprocal_data)
            max_val = np.max(reciprocal_data)
            normalized_data = [(value - min_val) / (max_val - min_val) for value in reciprocal_data]
            softmax_data = F.softmax(torch.tensor(normalized_data)  / 0.2, dim=0)
        else:
            softmax_data = torch.tensor([0.1]*20)
        weight.append(softmax_data)
    return weight

def average_weights(weights_list, model, classes, niid_type, feature_list, backbone_weight):
    centers_list = [[] for i in range(0, 20)]
    weight = []
    trainable_params = get_trainable_param_names(model)
    idx = 0
    for _, name in enumerate(trainable_params):
        if name.startswith('centers'):
            for w in weights_list:
                centers_list[idx].append(w[name].squeeze().cpu().numpy())
            idx += 1
    weight = compute_weight(centers_list, feature_list)
    
    avg_weights = collections.OrderedDict()

    weight_names = weights_list[0].keys()
    index=0
    for name in weight_names:

        if name not in trainable_params:
            if name in model.state_dict():
                avg_weights[name] = model.state_dict()[name]
        else:
            if name.startswith('centers'):
                aggregated_weight_tensor = torch.stack(
                    [w[name] * weight[index][i] for i, w in enumerate(weights_list)]).sum(dim=0)
                avg_weights[name] = aggregated_weight_tensor
                index += 1
            else:
                avg_weights[name] = torch.stack([w[name] * backbone_weight[i] for i, w in enumerate(weights_list)]).sum(dim=0)
    return avg_weights

def global_server(model, global_model, Waq, Wav, Wbq, Wbv, current_task):
    para_aq = []    
    para_bq = []
    para_av = []
    para_bv = []
    centers = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'linear_a_q_{}'.format(current_task) in name:
                para_aq.append(param.data)
            if 'linear_a_v_{}'.format(current_task) in name:
                para_av.append(param.data)
            if 'linear_b_q_{}'.format(current_task) in name:
                para_bq.append(param.data)
            if 'linear_b_v_{}'.format(current_task) in name:
                para_bv.append(param.data)
            if 'centers' in name:
                centers.append(param.data)
        idx = 0
        num = 0
        if current_task == 0:
            for name, param in global_model.named_parameters():
                if 'linear_a_q_{}'.format(current_task) in name:
                    param.data[idx:idx+4, :] = para_aq[0]
                if 'linear_a_v_{}'.format(current_task) in name:
                    param.data[idx:idx+4, :] = para_av[0]
                if 'linear_b_q_{}'.format(current_task) in name:
                    param.data[:, idx:idx+4] = para_bq[0]
                if 'linear_b_v_{}'.format(current_task) in name:
                    param.data[:, idx:idx+4] = para_bv[0]
                if 'centers' in name:
                    param.data = centers[num]
                    num += 1
        else:
            for name, param in global_model.named_parameters():
                if 'linear_a_q_{}'.format(current_task) in name:
                    param.data[idx:idx+4, :] = Waq[0] + para_aq[0]
                if 'linear_a_v_{}'.format(current_task) in name:
                    param.data[idx:idx+4, :] = Wav[0] + para_av[0]
                if 'linear_b_q_{}'.format(current_task) in name:
                    param.data[:, idx:idx+4] = Wbq[0] + para_bq[0]
                if 'linear_b_v_{}'.format(current_task) in name:
                    param.data[:, idx:idx+4] = Wbv[0] + para_bv[0]
                if 'centers' in name:
                    param.data = centers[num]
                    num += 1
    return global_model


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.centers_lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Users in one epoch  : {args.client_local}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
