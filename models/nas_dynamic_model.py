import copy
import random
import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.dynamic_layers import DynamicMBConvLayer, DynamicConvBnActLayer, DynamicLinearLayer, DynamicShortcutLayer
from .modules.static_layers import MobileInvertedResidualBlock
from .modules.nn_utils import make_divisible, int2list
from .modules.nn_base import MyNetwork
from .nas_static_model import FedStaticModel


class FedDynamicModel(MyNetwork):
    
    def __init__(self, supernet, n_classes = 10, bn_param = (0., 1e-5)):
        super(FedDynamicModel, self).__init__()
        
        self.supernet = supernet
        self.n_classes = n_classes
        self.stage_names = ['first_conv', 'mb1', 'mb2', 'mb3', 'mb4']
        
        self.width_list, self.depth_list, self.ks_list, self.expand_ratio_list = [], [], [], []
        for name in self.stage_names:
            block_cfg = getattr(self.supernet, name)
            self.width_list.append(block_cfg.c)
            if name.startswith('mb'):
                self.depth_list.append(block_cfg.d)
                self.ks_list.append(block_cfg.k)
                self.expand_ratio_list.append(block_cfg.t)
        self.resolution_list = self.supernet.resolutions
        
        self.cfg_candidates = {
            'resolution': self.resolution_list ,
            'width': self.width_list,
            'depth': self.depth_list,
            'kernel_size': self.ks_list,
            'expand_ratio': self.expand_ratio_list
        }
        
        #first conv layer, including conv, bn, act
        out_channel_list, act_func, stride = \
            self.supernet.first_conv.c, self.supernet.first_conv.act_func, self.supernet.first_conv.s
        self.first_conv = DynamicConvBnActLayer(
            in_channel_list=int2list(3), out_channel_list=out_channel_list, 
            kernel_size=3, stride=stride, act_func=act_func,
        )
        
        
        # inverted residual blocks
        self.block_group_info = []
        blocks = []
        _block_index = 0
        feature_dim = out_channel_list
        for stage_id, key in enumerate(self.stage_names[1:]):
            block_cfg = getattr(self.supernet, key)
            width = block_cfg.c
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            ks = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else 1
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim, 
                    out_channel_list=output_channel, 
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list, 
                    stride=stride, 
                    act_func=act_func, 
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet, 'channels_per_group', 1)
                )
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=stride)
                blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks = nn.ModuleList(blocks)

        last_channel, act_func = self.supernet.mb4.c, self.supernet.mb4.act_func
        
        #final layer
        self.classifier = DynamicLinearLayer(
            in_features_list=last_channel, out_features=n_classes, bias=True
        )        
        
        
        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0
        self.active_resolution = 32
        
        
    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, MobileInvertedResidualBlock):
                    if isinstance(m.mobile_inverted_conv, DynamicMBConvLayer) and m.shortcut is not None:
                        m.mobile_inverted_conv.point_linear.bn.rmsnorm.weight.zero_()
    
    
    @staticmethod
    def name():
        return 'FedDynamicModel'
    
    
    def forward(self, x):
        # resize input to target resolution first
        if x.size(-1) != self.active_resolution:
            x = torch.nn.functional.interpolate(x, size=self.active_resolution, mode='bicubic')

        # first conv
        x = self.first_conv(x)
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = F.adaptive_avg_pool2d(x,1)
        x = torch.squeeze(x)

        if self.active_dropout_rate > 0 and self.training:
            x = torch.nn.functional.dropout(x, p = self.active_dropout_rate)

        x = self.classifier(x)
        return x
    
    
    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str        
        
    @property
    def config(self):
        return {
            'name': FedDynamicModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
            'resolution': self.active_resolution
        }
    
    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')


    def set_active_subnet(self, resolution=32, width=None, depth=None, kernel_size=None, expand_ratio=None, **kwargs):
        assert len(depth) == len(kernel_size) == len(expand_ratio) == len(width) - 1
        self.active_resolution = resolution

        # first conv
        self.first_conv.active_out_channel = width[0] 

        for stage_id, (c, k, e, d) in enumerate(zip(width[1:-1], kernel_size, expand_ratio, depth)):
            start_idx, end_idx = min(self.block_group_info[stage_id]), max(self.block_group_info[stage_id])
            for block_id in range(start_idx, start_idx+d):
                block = self.blocks[block_id]
                #block output channels
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c

                #dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k

                #dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e

        #IRBlocks repated times
        for i, d in enumerate(depth):
            self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        # #last conv
        # if not self.use_v3_head:
        #     self.last_conv.active_out_channel = width[-1]
        # else:
        #     # default expansion ratio: 6
        #     self.last_conv.final_expand_layer.active_out_channel = width[-2] * 6
        #     self.last_conv.feature_mix_layer.active_out_channel = width[-1]
            

        
    def get_active_subnet_settings(self):
        r = self.active_resolution
        width, depth, kernel_size, expand_ratio= [], [], [],  []

        #first conv
        width.append(self.first_conv.active_out_channel)
        for stage_id in range(len(self.block_group_info)):
            start_idx = min(self.block_group_info[stage_id])
            block = self.blocks[start_idx]  #first block
            width.append(block.mobile_inverted_conv.active_out_channel)
            kernel_size.append(block.mobile_inverted_conv.active_kernel_size)
            expand_ratio.append(block.mobile_inverted_conv.active_expand_ratio)
            depth.append(self.runtime_depth[stage_id])
        
        return {
            'resolution': r,
            'width': width,
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'depth': depth,
        }        
    
    def set_dropout_rate(self, dropout=0, drop_connect=0, drop_connect_only_last_two_stages=True):
        self.active_dropout_rate = dropout
        for idx, block in enumerate(self.blocks):
            if drop_connect_only_last_two_stages:
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            this_drop_connect_rate = drop_connect * float(idx) / len(self.blocks)
            block.drop_connect_rate = this_drop_connect_rate


    def sample_min_subnet(self):
        return self._sample_active_subnet(min_net=True)


    def sample_max_subnet(self):
        return self._sample_active_subnet(max_net=True)
    

    def sample_active_subnet(self, compute_flops=False):
        cfg = self._sample_active_subnet(
            False, False
        ) 
        if compute_flops:
            cfg['flops'] = self.compute_active_subnet_flops()
        return cfg
    
    
    def sample_min_subnet(self):
        return self._sample_active_subnet(min_net = True)
    
    
    def sample_max_subnet(self, compute_flops=False):
        cfg = self._sample_active_subnet(
            False, False
        ) 
        if compute_flops:
            cfg['flops'] = self.compute_active_subnet_flops()
        return cfg
    
    def _sample_active_subnet(self, min_net=False, max_net=False):

        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))

        cfg = {}
        # sample a resolution
        cfg['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = []
            for vv in self.cfg_candidates[k]:
                cfg[k].append(sample_cfg(int2list(vv), min_net, max_net))

        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg
    
        
    
    def mutate_and_reset(self, cfg, prob=0.1, keep_resolution=False):
        cfg = copy.deepcopy(cfg)
        pick_another = lambda x, candidates: x if len(candidates) == 1 else random.choice([v for v in candidates if v != x])
        # sample a resolution
        r = random.random()
        if r < prob and not keep_resolution:
            cfg['resolution'] = pick_another(cfg['resolution'], self.cfg_candidates['resolution'])

        # sample channels, depth, kernel_size, expand_ratio
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            for _i, _v in enumerate(cfg[k]):
                r = random.random()
                if r < prob:
                    cfg[k][_i] = pick_another(cfg[k][_i], int2list(self.cfg_candidates[k][_i]))

        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        
        return cfg
    
    def crossover_and_reset(self, cfg1, cfg2, p=0.5):
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if random.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        cfg = {}
        cfg['resolution'] = cfg1['resolution'] if random.random() < p else cfg2['resolution']
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = _cross_helper(cfg1[k], cfg2[k], p)

        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio']
        )
        return cfg 
    
    def get_active_subnet(self, preserve_weight=True):
        with torch.no_grad():
            first_conv = self.first_conv.get_active_subnet(3, preserve_weight)

            blocks = []
            input_channel = first_conv.out_channels
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                        self.blocks[idx].shortcut.get_active_subnet(input_channel, preserve_weight) if self.blocks[idx].shortcut is not None else None
                    ))
                    input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks += stage_blocks

            
            in_features = stage_blocks[-1].mobile_inverted_conv.out_channels
            classifier = self.classifier.get_active_subnet(in_features, preserve_weight)
            
            _subnet = FedStaticModel(
                first_conv, blocks, classifier, self.active_resolution
            )
            # _subnet.set_bn_param(**self.get_bn_param())
            return _subnet


    def get_active_net_config(self):
        raise NotImplementedError


    def compute_active_subnet_flops(self):

        def count_conv(c_in, c_out, size_out, groups, k):
            kernel_ops = k**2
            output_elements = c_out * size_out**2
            ops = c_in * output_elements * kernel_ops / groups
            return ops

        def count_linear(c_in, c_out):
            return c_in * c_out

        total_ops = 0

        c_in = 3
        size_out = self.active_resolution // self.first_conv.stride
        c_out = self.first_conv.active_out_channel

        total_ops += count_conv(c_in, c_out, size_out, 1, 3)
        c_in = c_out

        # mb blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                block = self.blocks[idx]
                c_middle = make_divisible(round(c_in * block.mobile_inverted_conv.active_expand_ratio), 8)
                # 1*1 conv
                if block.mobile_inverted_conv.inverted_bottleneck is not None:
                    total_ops += count_conv(c_in, c_middle, size_out, 1, 1)
                # dw conv
                stride = 1 if idx > active_idx[0] else block.mobile_inverted_conv.stride
                if size_out % stride == 0:
                    size_out = size_out // stride
                else:
                    size_out = (size_out +1) // stride
                total_ops += count_conv(c_middle, c_middle, size_out, c_middle, block.mobile_inverted_conv.active_kernel_size)
                # 1*1 conv
                c_out = block.mobile_inverted_conv.active_out_channel
                total_ops += count_conv(c_middle, c_out, size_out, 1, 1)
                #se
                if block.mobile_inverted_conv.use_se:
                    num_mid = make_divisible(c_middle // block.mobile_inverted_conv.depth_conv.se.reduction, divisor=8)
                    total_ops += count_conv(c_middle, num_mid, 1, 1, 1) * 2
                if block.shortcut and c_in != c_out:
                    total_ops += count_conv(c_in, c_out, size_out, 1, 1)
                c_in = c_out

        # n_classes
        total_ops += count_linear(c_out, self.n_classes)
        return total_ops / 1e6


    def load_weights_from_pretrained_models(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        assert isinstance(checkpoint, dict)
        pretrained_state_dicts = checkpoint['state_dict']
        for k, v in self.state_dict().items():
            name = 'module.' + k if not k.startswith('module') else k
            v.copy_(pretrained_state_dicts[name])