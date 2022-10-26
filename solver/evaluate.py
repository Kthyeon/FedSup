#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pdb

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_img(net_g, datatest, args, return_probs=False, user_idx=-1):
    # net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    probs = []
    
    if args.architecture_option == 'max_supernet':
        net_g.sample_max_subnet()
    elif args.architecture_option == 'random_supernet':
        net_g.sample_active_subnet()
    elif args.architecture_option == 'min_supernet':
        net_g.sample_min_subnet()

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)

        
        log_probs = net_g(data)
        probs.append(log_probs.clone().detach().cpu())

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').detach().cpu().item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1].detach().cpu()
        correct += y_pred.eq(target.cpu().data.view_as(y_pred)).long().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        if user_idx < 0:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        else:
            print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                user_idx, test_loss, correct, len(data_loader.dataset), accuracy))

    if return_probs:
        return accuracy, test_loss, torch.cat(probs)
    return accuracy, test_loss


def test_img_local(net_g, dataset, args, user_idx=-1, idxs=None):
    # net_g.eval()
    net_g.train()
    # testing
    test_loss = 0
    correct = 0

    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.bs, shuffle=False)
    l = len(data_loader)

    
    try:
        if user_idx < 25:
            net_g.sample_min_subnet()
        elif user_idx < 50:
            net_g.sample_max_subnet()
        elif user_idx < 75:
            tmp = get_params(net_g, target_min = 0.75, target_max = 1.0)
            net_g.set_active_subnet(
            resolution =tmp['resolution'],
            width =tmp['width'],
            depth =tmp['depth'],
            kernel_size =tmp['kernel_size'],
            expand_ratio =tmp['expand_ratio'],
            )
        else:
            tmp = get_params(net_g, target_min = 2.0, target_max = 2.75)
            net_g.set_active_subnet(
            resolution =tmp['resolution'],
            width =tmp['width'],
            depth =tmp['depth'],
            kernel_size =tmp['kernel_size'],
            expand_ratio =tmp['expand_ratio'],
            )
    except:
        pass
    
    
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    if args.verbose:
        print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            user_idx, test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss

def test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False):
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        net_local.eval()
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])

        acc_test_local[idx] = a
        loss_test_local[idx] = b

    if return_all:
        return acc_test_local, loss_test_local
    return acc_test_local.mean(), loss_test_local.mean()

def test_img_avg_all(net_glob, net_local_list, args, dataset_test, return_net=False):
    net_glob_temp = copy.deepcopy(net_glob)
    w_keys_epoch = net_glob.state_dict().keys()
    w_glob_temp = {}
    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        w_local = net_local.state_dict()

        if len(w_glob_temp) == 0:
            w_glob_temp = copy.deepcopy(w_local)
        else:
            for k in w_keys_epoch:
                w_glob_temp[k] += w_local[k]

    for k in w_keys_epoch:
        w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)
    net_glob_temp.load_state_dict(w_glob_temp)
    acc_test_avg, loss_test_avg = test_img(net_glob_temp, dataset_test, args)

    if return_net:
        return acc_test_avg, loss_test_avg, net_glob_temp
    return acc_test_avg, loss_test_avg

criterion = nn.CrossEntropyLoss()

def get_params(net, target_min, target_max):
    params = 9999
    while (params > target_max) or (params < target_min):
        net_tmp = copy.deepcopy(net)
        tmp = net_tmp.sample_active_subnet()
        net_tmp.set_active_subnet(
        resolution =tmp['resolution'],
        width =tmp['width'],
        depth =tmp['depth'],
        kernel_size =tmp['kernel_size'],
        expand_ratio =tmp['expand_ratio'],
        )
        net_tmp = net_tmp.get_active_subnet()

        params = sum(p.numel() for p in net_tmp.parameters() if p.requires_grad)
        params = params / 1e6
    
    return tmp