#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb
from solver.build import build_optimizer
# import utils.loss_ops as loss_ops 
import utils.loss_ops as loss_ops
import copy

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, arch_sampler=None, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).cuda(args.device)
        self.soft_loss_func = loss_ops.KLLossSoft().cuda(args.device)
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain
        self.arch_sampler = arch_sampler

    def train(self, net, idx=-1, lr=0.1):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)
        optimizer = build_optimizer(self.args, net)
        
        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_ep
        
        sandwich_rule = getattr(self.args, 'sandwich_rule', True)
        if self.args.num_arch_training ==1:
            if sandwich_rule == True:
                net.sample_max_subnet()
            else:
                net.sample_active_subnet()
        
        if idx < 25:
            net.sample_min_subnet()
        elif idx < 50:
            net.sample_max_subnet()
        elif idx < 75:
            tmp = get_params(net, target_min = 0.75, target_max = 1.0)
            net.set_active_subnet(
            resolution =tmp['resolution'],
            width =tmp['width'],
            depth =tmp['depth'],
            kernel_size =tmp['kernel_size'],
            expand_ratio =tmp['expand_ratio'],
            )
        else:
            tmp = get_params(net, target_min = 2.0, target_max = 2.75)
            net.set_active_subnet(
            resolution =tmp['resolution'],
            width =tmp['width'],
            depth =tmp['depth'],
            kernel_size =tmp['kernel_size'],
            expand_ratio =tmp['expand_ratio'],
            )
        
        
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                num_subnet_training = max(1, getattr(self.args, 'num_arch_training', 1))
                # print(labels)
                optimizer.zero_grad()
                drop_connect_only_last_two_stages = getattr(self.args, 'drop_connect_only_last_two_stages', True)
                   


                net.set_dropout_rate(self.args.dropout, self.args.drop_connect, drop_connect_only_last_two_stages) #dropout for supernet
                output = net(images)
    

                loss = self.loss_func(output, labels)
                loss.backward()
                
                with torch.no_grad():
                    soft_logits = output.clone().detach()

                #step 2. sample the smallest network and several random networks
                net.set_dropout_rate(0, 0, drop_connect_only_last_two_stages)  #reset dropout rate
                for arch_id in range(1, num_subnet_training):
                    if arch_id == num_subnet_training-1 and sandwich_rule:
                        net.sample_min_subnet()
                    else:
                        # attentive sampling with training loss as the surrogate performance metric 
                        if self.arch_sampler is not None:
                            sampling_method = self.args.sampler.method
                            if sampling_method in ['bestup', 'worstup']:
                                target_flops = arch_sampler.sample_one_target_flops()
                                candidate_archs = arch_sampler.sample_archs_according_to_flops(
                                    target_flops, n_samples=args.sampler.num_trials
                                )
                                my_pred_accs = []
                                for arch in candidate_archs:
                                    net.set_active_subnet(**arch)
                                    with torch.no_grad():
                                        my_pred_accs.append(-1.0 * criterion(net(images), labels))

                                if sampling_method == 'bestup':
                                    idx, _ = max(enumerate(my_pred_accs), key=operator.itemgetter(1))                          
                                else:
                                    idx, _ = min(enumerate(my_pred_accs), key=operator.itemgetter(1))                          
                                net.set_active_subnet(**candidate_archs[idx])  #reset
                            else:
                                raise NotImplementedError
                        else:
                            net.sample_active_subnet()

                    # calcualting loss
                    output = net(images)

                    if self.soft_loss_func:
                        loss = self.soft_loss_func(output, soft_logits)
                    else:
                        assert not args.inplace_distill
                        loss = self.loss_func(output, labels)

                    loss.backward()

                    #clip gradients if specfied
                if getattr(self.args, 'grad_clip_value', None):
                    torch.nn.utils.clip_grad_value_(net.parameters(), self.args.grad_clip_value)
                    
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
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
    
