
import torch
from torch import nn
from torch.nn import *
from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block
class NetInNetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, config):
        super().__init__()
        
        self.net_in_net = nn.Sequential(*[stack_block(
              in_f=in_c,
              out_f=out_c,
              kernel_size=1,
              block=conv_block,
              depth=1,
              activation=globals()[config.decoder_fn]
              ) for in_c,out_c in zip(config.decoder_channels[:-1],config.decoder_channels[1:]) ])
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.decoder = nn.Linear(config.decoder_channels[-1],config.n_classes)

    def forward(self, x):
        x = self.net_in_net(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x