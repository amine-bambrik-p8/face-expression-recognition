import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import XceptionBlock
from torch.nn import *
import torch

from models.layers.avg_decoder import AvgDecoder
from models.layers.net_in_net_decoder import NetInNetDecoder

from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block
def block(in_f, out_f,dropout=0.0,depth=2,activation=nn.ReLU()):
    l = []
    for i in range(depth-1):
      b=nn.BatchNorm2d(out_f)
      m=nn.Sequential(
          XceptionBlock(out_f, out_f),
          b,
          activation
          )
      l.append(m)
    b=nn.BatchNorm2d(out_f)
    return nn.Sequential(
        XceptionBlock(in_f, out_f),
        b,
        activation,
        *l,
        conv_block(out_f,out_f,kernel_size=2,activation=activation,stride=2),
        nn.Dropout2d(dropout) if(dropout>0.0) else nn.Identity(),
    )
class GoodFellowV3XceptionDown(nn.Module):
  def __init__(self,config):
    super(GoodFellowV3XceptionDown,self).__init__()
    self.gate = stack_block(
              in_f=config.in_channels,
              out_f=config.encoder_channels[0],
              kernel_size=7,
              block=same_conv_block,
              depth=config.encoder_depths[0],
              conv_block=conv_block
              )
    self.encoder = nn.Sequential(*[block(in_c,out_c,dropout=config.encoder_dropout,depth=depth,activation=globals()[config.encoder_fn](*config.encoder_fn_params)) for in_c,out_c,depth in zip(config.encoder_channels[:-1],config.encoder_channels[1:],config.encoder_depths)])
    self.decoder = globals()[config.decoder](config)
    self.class_fn = globals()[config.class_fn](dim=1)

  def forward(self,x):
    x = self.gate(x)
    x = self.encoder(x)
    x = self.decoder(x)
    return self.class_fn(x)