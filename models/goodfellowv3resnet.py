import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.basic_decoder import BasicDecoder

class GoodFellowV3ResNet(nn.Module):
    def __init__(self,config, in_channels=1, n_classes=7, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, block=ResNetBasicBlock,blocks_sizes=config.encoder_channels, deepths=[2, 2, 2])
        self.decoder = BasicDecoder(config)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.log_softmax(x,dim=1)

