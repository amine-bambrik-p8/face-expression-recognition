import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
from models.layers.basic_decoder import BasicDecoder
from models.layers.avg_decoder import AvgDecoder 
from models.layers.net_in_net_decoder import NetInNetDecoder 
import torch
from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block
class EncoderBNDO(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.enc_blocks = nn.Sequential(
            *[stack_block(
              in_f=in_c,
              out_f=out_c,
              kernel_size=5,
              block=same_conv_block,
              depth=2,
              out_gate=nn.Sequential(
                  nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                  ),
                  nn.Dropout2d(config.encoder_dropout) if config.encoder_dropout > 0.0 else nn.Identity()
                ),
              conv_block=conv_block
              ) for in_c,out_c in zip(config.encoder_channels[:-1],config.encoder_channels[1:])],
        )
    def forward(self, x):
        return self.enc_blocks(x)
class GoodFellowV3(nn.Module):
  def __init__(self,config):
    super(GoodFellowV3,self).__init__()
    self.gate = stack_block(
              in_f=config.in_channels,
              out_f=config.encoder_channels[0],
              kernel_size=7,
              block=same_conv_block,
              depth=2,
              out_gate=nn.MaxPool2d(
                kernel_size=2,
                stride=2
                ),
              conv_block=conv_block
              )
    self.encoder = EncoderBNDO(config)
    #1,[64,64,128]
    self.decoder = globals()[config.decoder](config)
    #[128*6*6,1024,1024],7
    self.class_fn = globals()[config.class_fn](dim=1)
  def forward(self,x):
    x = self.gate(x)    
    x = self.encoder(x)
    x = self.decoder(x)
    return self.class_fn(x)
  

class GoodFellowV3Inference(nn.Module):
  def __init__(self):
    super(GoodFellowV3Inference,self).__init__()
    
    self.encoder = EncoderBNDO(1,[64,64,128])
    self.decoder = BasicDecoder([128*6*6,1024,1024],7)

  def forward(self,x):
    x = x.reshape(48, 48, 4,-1)
    x = torch.narrow(x, dim=2, start=3, length=1)
    x = x.reshape(-1,1, 48, 48)
    x = x / 255
    x = (x - 0.5) / 0.5
    x = self.encoder(x)
    x = torch.flatten(x,1)
    x = self.decoder(x)
    return F.softmax(x,dim=1)