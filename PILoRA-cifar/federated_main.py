#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import copy
import os
import shutil
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate
from ResNet import resnet18
from utils import get_trainand_test_dataset, get_dataset, average_weights, exp_details, build_continual_dataset
from VLT import *
from VITLORA import vitlora
import torch.nn.functional as F

def inference(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for setp, (indexs, data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            #data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    return acc, test_loss


def prepare_folders(cur_path):
    folders_util = [
        os.path.join(cur_path + '/logs-VIT-LoRA', args.store_name),
        os.path.join(cur_path + '/checkpoints', args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.makedirs(folder, exist_ok=True)


def save_checkpoint(state, is_best):
    filename = '{}/{}/ckpt.pth.tar'.format(os.path.abspath(os.path.dirname(os.getcwd())) + '/checkpoints',
                                           args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    task_size = int((args.total_classes - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    args.type = 'iid' if args.iid == 1 else 'non-iid'
    args.store_name = '_'.join(
        [args.dataset, args.model, args.type, 'lr-' + str(args.centers_lr)])
    cur_path = os.path.abspath(os.path.dirname(os.getcwd()))
    prepare_folders(cur_path)
    exp_details(args)

    seed = 2023  
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # BUILD MODEL
    file_name = args.store_name
    class_set = list(range(args.total_classes))
    model = VLT(modelname='vit_base_patch16_224_dino',
                num_classes=args.total_classes,
                pretrained=True,
                r = 4,
                lora_layer = [0])
    model = model.to(args.device)

    global_model = vitlora(args, file_name, model, task_size, args.device)
    global_model.setup_data(shuffle=True, seed=0)

    num_params = np.sum([p.numel() for p in model.parameters() if p.requires_grad or not p.requires_grad])
    learnable_params = np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
    print('# of learnable params: {}; total params: {}; {:.4f}%'.format(learnable_params, num_params, (learnable_params/num_params)*100))

    for i in range(args.task_num+1):

        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])

        filename = 'log_task{}.txt'.format(i)
        logger_file = open(os.path.join(cur_path + '/logs-VIT-LoRA', args.store_name, filename), 'w')
        tf_writer = SummaryWriter(log_dir=os.path.join(cur_path + '/logs-VIT-LoRA', args.store_name))

        global_model.beforeTrain(i)
        global_model.train(i, old_class=old_class, tf_writer=tf_writer, logger_file=logger_file)
        global_model.afterTrain(i)

        