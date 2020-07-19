import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *

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
        return F.log_softmax(x,dim=1)

