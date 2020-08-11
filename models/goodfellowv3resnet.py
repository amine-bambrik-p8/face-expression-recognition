import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.basic_decoder import BasicDecoder
from models.layers.avg_decoder import AvgDecoder
from models.layers.net_in_net_decoder import NetInNetDecoder
from torch.nn import *
class GoodFellowV3ResNet(nn.Module):
    def __init__(self,config, in_channels=1, n_classes=7, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, block=ResNetBasicBlock,blocks_sizes=config.encoder_channels, deepths=config.encoder_depths)
        self.decoder = globals()[config.decoder](config)
        self.class_fn = globals()[config.class_fn](dim=1)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.class_fn(x)

