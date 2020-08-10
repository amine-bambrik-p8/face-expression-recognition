import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import InceptionBlock

from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block
def block(in_f, out_f,dropout=0.0):
    return nn.Sequential(
        InceptionBlock(in_f, out_f),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        InceptionBlock(out_f, out_f),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
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
              depth=2,
              conv_block=conv_block
              )
    self.encoder = *[block(in_c,out_c) for in_c,out_c in zip(config.encoder_channels[:-1],config.encoder_channels[1:])]
    self.decoder = BasicDecoder(config)

  def forward(self,x):
    x = self.gate(x)
    x = self.encoder(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)