#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch
from copy import deepcopy
import argparse
import random
import os

from fedutils.train_utils import get_data

from utils.options import args_parser, build_args_and_env, renewal_args
from utils.config import setup

from models.model_factory import create_model

from solver.local_update import LocalUpdate
from solver.evaluate import test_img
from solver.build import build_lr_scheduler

import pdb



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedSuperNet Training')
    parser.add_argument('--config-file', default=None, type=str, 
                    help='training configuration')
    
    parser = args_parser(parser)
    run_args = parser.parse_args()
    args = build_args_and_env(run_args)
    args = renewal_args(args, run_args)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.diri:
        base_dir = './save_pn/{}/{}_iid{}_num{}_C{}_le{}_ls{}_mt{}_wd{}_idis{}_nat{}_dc{}_dr{}/beta{}/{}/'.format(
            args.dataset, args.exp_name, args.iid, args.num_users, args.frac, args.local_ep, args.label_smoothing,
            args.optimizer.momentum, args.weight_decay_weight, args.inplace_distill, args.num_arch_training,
            args.drop_connect, args.dropout, args.beta, args.results_save)
    else:
        base_dir = './save_pn/{}/{}_iid{}_num{}_C{}_le{}_ls{}_mt{}_wd{}_idis{}_nat{}_dc{}_dr{}/shard{}/{}/'.format(
            args.dataset, args.exp_name, args.iid, args.num_users, args.frac, args.local_ep, args.label_smoothing, 
            args.optimizer.momentum, args.weight_decay_weight, args.inplace_distill, args.num_arch_training, 
            args.drop_connect, args.dropout, args.shard_per_user, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'fed')):
        os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)
    
    # build model
    net_glob = create_model(args).to(args.device)
    net_glob.train()
    
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # training
    results_save_path = os.path.join(base_dir, 'fed/results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None
    
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.optimizer.momentum)
    args.n_iters_per_epoch = int(len(dataset_train) // args.local_bs // (args.num_users*args.frac))
    lr_scheduler = build_lr_scheduler(args, optimizer)
    lr = optimizer.param_groups[0]['lr']
    

    results = []

    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        for idx in idxs_users:
            local = LocalUpdate(args=args, arch_sampler=None, 
                                dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)
            
            w_local, loss = local.train(net=net_local.to(args.device), idx=idx)
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            # net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))


            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, 
                                         columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

        if (iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, 'fed/best_{}.pt'.format(iter + 1))
            model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)
            
        

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))