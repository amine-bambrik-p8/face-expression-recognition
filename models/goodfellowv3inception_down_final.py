import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import InceptionBlock
from torch.nn import *
import torch

from models.layers.avg_decoder import AvgDecoder
from models.layers.net_in_net_decoder import NetInNetDecoder
from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block
def net_conv_block(in_f, out_f,kernel_size,batch_norm=True,dropout=0.0,activation=nn.ReLU(),*args, **kwargs):
    c=nn.Conv2d(in_f, out_f,kernel_size=kernel_size,bias=False, *args, **kwargs)
    b=nn.BatchNorm2d(out_f) if(batch_norm) else nn.Identity()
    torch.nn.init.xavier_normal_(c.weight)
    return nn.Sequential(
        c,
        b,
        nn.LocalResponseNorm(out_f) if(batch_norm) else nn.Identity(),
        activation,
        nn.Dropout2d(dropout) if(dropout > 0.0) else nn.Identity()
    )
def block(in_f, out_f,dropout=0.0,depth=2,activation=nn.ReLU()):
    l = []
    for i in range(depth-1):
      b=nn.BatchNorm2d(out_f)
      m=nn.Sequential(
          InceptionBlock(out_f, out_f),
          b,
          activation
          )
      l.append(m)
    b=nn.BatchNorm2d(out_f)
    return nn.Sequential(
        conv_block(in_f,in_f,kernel_size=2,activation=activation,stride=2),
        InceptionBlock(in_f, out_f),
        b,
        activation,
        *l,
        nn.Dropout2d(dropout) if(dropout>0.0) else nn.Identity()
    )
class GoodFellowV3InceptionDownFinal(nn.Module):
  def __init__(self,config):
    super(GoodFellowV3InceptionDownFinal,self).__init__()
    self.gate = stack_block(
              in_f=config.in_channels,
              out_f=config.encoder_channels[0],
              kernel_size=7,
              block=same_conv_block,
              depth=config.encoder_depths[0],
              conv_block=conv_block
              )
    self.encoder = nn.Sequential(*[block(in_c,out_c,dropout=config.encoder_dropout,depth=depth,activation=globals()[config.encoder_fn](*config.encoder_fn_params)) for in_c,out_c,depth in zip(config.encoder_channels[:-1],config.encoder_channels[1:],config.encoder_depths)])
    self.net_in_net = stack_block(
              in_f=config.encoder_channels[-1],
              out_f=config.encoder_channels[-1],
              kernel_size=1,
              block=net_conv_block,
              depth=1,
              activation=globals()[config.decoder_fn](*config.decoder_fn_params),
              dropout=config.encoder_dropout,
              batch_norm=config.encoder_batch_norm,
    )
    self.avg = nn.AdaptiveAvgPool2d((1,1))
    self.decoder = globals()[config.decoder](config)
    self.class_fn = globals()[config.class_fn](dim=1)

  def forward(self,x):
    x = self.gate(x)
    x = self.encoder(x)
    x = self.net_in_net(x)
    x = self.avg(x)
    x = self.decoder(x)
    return self.class_fn(x)