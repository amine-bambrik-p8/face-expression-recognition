import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import InceptionBlock

from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block
def block(in_f, out_f,dropout=0.0,depth=2,activation=nn.ReLU()):
    return nn.Sequential(
        InceptionBlock(in_f, out_f),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        *[
          nn.Sequential(
          InceptionBlock(out_f, out_f),
          nn.BatchNorm2d(out_f),
          nn.ReLU())
        for i in range(depth-1)],
        nn.MaxPool2d(2,2),
        nn.Dropout2d(dropout) if(dropout>0.0) else nn.Identity()
    )
class GoodFellowV3Inception(nn.Module):
  def __init__(self,config):
    super(GoodFellowV3Inception,self).__init__()
    self.gate = stack_block(
              in_f=config.in_channels,
              out_f=config.encoder_channels[0],
              kernel_size=7,
              block=same_conv_block,
              depth=config.encoder_depths[0],
              conv_block=conv_block
              )
    self.encoder = nn.Sequential(*[block(in_c,out_c,dropout=config.encoder_dropout,depth=depth,activation=globals()[config.encoder_fn](*config.encoder_fn_params)) for in_c,out_c,depth in zip(config.encoder_channels[:-1],config.encoder_channels[1:],config.encoder_depths[1:])])
    self.decoder = globals()[config.decoder](config)
    self.class_fn = globals()[config.class_fn](dim=1)

  def forward(self,x):
    x = self.gate(x)
    x = self.encoder(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return self.class_fn(x)