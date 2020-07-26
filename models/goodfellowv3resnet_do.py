import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.basic_decoder import BasicDecoder
from functools import partial


from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block

class ResNetBasicBlockDO(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels,dropout=0.2, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            same_conv_block(self.out_channels, self.expanded_channels, conv_block=conv_block,activation=nn.Identity, bias=False,stride=self.downsampling),
            same_conv_block(self.in_channels, self.expanded_channels, conv_block=conv_block, bias=False,dropout=dropout),
        )

class GoodFellowV3ResNetDO(nn.Module):
    def __init__(self, in_channels=1, n_classes=7, *args, **kwargs):
        super().__init__()
        self.gate = ResNetEncoder(in_channels,blocks_sizes=[64], deepths=[2])
        self.encoder = ResNetEncoder(64, block=ResNetBasicBlockDO,blocks_sizes=[64,128], deepths=[2, 2])
        self.decoder = BasicDecoder([128*12*12,1024,1024],7,dropout=0.5)

    def forward(self, x):
        x = self.gate(x)
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        x = self.decoder(x)
        return F.log_softmax(x,dim=1)

