import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.basic_decoder import BasicDecoder

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features,n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return F.log_softmax(x,dim=1)

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        # self.gate = nn.Sequential(
        #     nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(self.blocks_sizes[0]),
        #     activation_func(activation),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            # ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
            #             block=block,*args, **kwargs),
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

class LeNetResNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=7, *args, **kwargs):
        super().__init__()
        in_channels = 1
        n_classes = 7
        self.encoder = ResNetEncoder(in_channels, block=ResNetBasicBlock,blocks_sizes=[16, 32, 64], deepths=[3, 3, 3])
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

