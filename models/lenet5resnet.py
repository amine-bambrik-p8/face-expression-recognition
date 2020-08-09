import torch.nn as nn
import torch.nn.functional as F
from models.layers.res_block import *
from models.layers.basic_decoder import BasicDecoder
class LeNetResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder(1,blocks_sizes=[64, 128, 256], deepths=[2,2,2])
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return F.log_softmax(x,dim=1)

