import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

from .modules.static_layers import set_layer_from_config, MBInvertedConvLayer, ConvBnActLayer, ShortcutLayer, LinearLayer, MobileInvertedResidualBlock, IdentityLayer
from .modules.nn_utils import make_divisible
from .modules.nn_base import MyNetwork


class FedStaticModel(MyNetwork):
    
    def __init__(self, first_conv, blocks, classifier, resolution):
        super(FedStaticModel, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.resolution = resolution
        self.classifier = classifier
        
        
    def forward(self, x):
        # logit which is an input vector of softmax function
        if x.size(-1) != self.resolution:
            x = torhx.nn.functional.interpolate(x, size=self.resolution, mode = 'bicubic')
        
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.squeeze(x)
        x = self.classifier(x)
            
        return x
    
    def extract_features(self, x):
        # penultimate layer's feature representation (vector)
        if x.size(-1) != self.resolution:
            x = torhx.nn.functional.interpolate(x, size=self.resolution, mode = 'bicubic')
        
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(out,1)
        x = torch.squeeze(x)
        
        return x
    
    
    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        for block in self.blocks:
            _str += block.module_str + '\n'
        _str += self.classifier.module_str
        
        return _str
    
    
    @property
    def config(self):
        return {
            'name': FedStaticModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.confg,
            'blocks': [
                blcok.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
            'resolution': self.resolution
        }
    
    
    def weight_initialization(self):
        # weight parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        
        
    @staticmethod
    def build_from_config(config):
        raise NotImplementedError
        
    def reset_running_stats_for_calibration(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.SyncBatchNorm):
                m.training = True
                m.momentum = None # culumatvie moving average
                m.reset_running_sstats()