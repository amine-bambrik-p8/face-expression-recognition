import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.layers.conv_auto import Conv2dAuto
from models.layers.res_block import *
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResCeptionBlock(ResNetBasicBlock):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            InceptionBlock(self.in_channels, self.out_channels),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False,stride=self.downsampling),
        )

class InceptionBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionBlock,self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels,out_channels,kernel_size=1)

        self.branch5x5_1 =  nn.Conv2d(in_channels,out_channels//2,kernel_size=1)
        self.branch5x5_2 = Conv2dAuto(out_channels//2,out_channels,kernel_size=5)

        self.branch3x3_1 =  nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.branch3x3_2 = Conv2dAuto(out_channels,out_channels*2,kernel_size=3)

        self.compress = nn.Conv2d(out_channels*4,out_channels=out_channels,kernel_size=1)
    def forward(self,x):
        out_branch1x1 = self.branch1x1(x)
        
        out_branch5x5 = self.branch5x5_1(x)
        out_branch5x5 = self.branch5x5_2(out_branch5x5)

        out_branch3x3 = self.branch3x3_1(x)
        out_branch3x3 = self.branch3x3_2(out_branch3x3)

        outputs = [out_branch1x1,out_branch3x3,out_branch5x5]
        outputs = torch.cat(outputs,1)
        out = self.compress(outputs)
        return out




