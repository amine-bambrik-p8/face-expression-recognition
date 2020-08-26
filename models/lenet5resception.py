import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import InceptionBlock
from torch.nn import *
import torch

from models.layers.avg_decoder import AvgDecoder
from models.layers.net_in_net_decoder import NetInNetDecoder
from models.layers.stack_block import stack_block
from torch.nn import *

class ResCeptionBlock(ResNetBasicBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_block(self.in_channels,self.in_channels,kernel_size=2,activation=nn.ReLU(),stride=2) if self.downsampling==2 else nn.Identity(),
            InceptionBlock(self.in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            InceptionBlock(self.out_channels, self.expanded_channels),
            nn.BatchNorm2d(self.expanded_channels),
            nn.ReLU(),
        )
class LeNetResCeptionNet(nn.Module):
    def __init__(self,config,in_channels=1, n_classes=7, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(config.in_channels, block=ResCeptionBlock,blocks_sizes=config.encoder_channels, deepths=config.encoder_depths,activation=globals()[config.encoder_fn](*config.encoder_fn_params))
        self.decoder = globals()[config.decoder](config)
        self.class_fn = globals()[config.class_fn](dim=1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.class_fn(x)

