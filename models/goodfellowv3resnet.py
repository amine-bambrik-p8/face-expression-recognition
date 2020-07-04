import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.basic_decoder_bn_do_v3 import BasicDecoderBNDO

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.in_out_block_sizes = list(zip(blocks_sizes[:-1], blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(in_channels, blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class GoodFellowV3ResNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=7, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, block=ResNetBasicBlock,blocks_sizes=[64, 64,128], deepths=[2, 2, 2])
        self.decoder = BasicDecoderBNDO([128*12*12,1024,1024],7)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0),-1)
        x = self.decoder(x)
        return F.log_softmax(x,dim=1)

