import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.inception_block import InceptionBlock
from models.layers.stack_block import stack_block
from torch.nn import *

class ResCeptionBlock(ResNetBasicBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            InceptionBlock(in_channels, out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            InceptionBlock(out_channels, out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2,2) if self.downsampling==2 else nn.Identity()
        )
class LeNetResCeptionNet(nn.Module):
    def __init__(self,config,in_channels=1, n_classes=7, *args, **kwargs):
        super().__init__()
        self.gate = stack_block(
              in_f=config.in_channels,
              out_f=config.encoder_channels[0],
              kernel_size=7,
              block=ResNetBasicBlock,
              depth=2
              )
        self.encoder = ResNetEncoder(config.encoder_channels[0], block=ResCeptionBlock,blocks_sizes=config.encoder_channels[1:], deepths=config.encoder_depths)
        self.decoder = globals()[config.decoder](config)
        self.class_fn = globals()[config.class_fn](dim=1)
        
    def forward(self, x):
        x = self.gate(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return F.log_softmax(x,dim=1)

