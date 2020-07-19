import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder

from models.layers.conv_block import conv_block
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block

class EncoderBNDO(nn.Module):
    def __init__(self,in_c,channels):
        super().__init__()
        self.enc_blocks = nn.Sequential(
            stack_block(
              in_f=in_c,
              out_f=channels[0],
              kernel_size=3,
              block=same_conv_block,
              depth=1,
              out_gate=nn.MaxPool2d(
                kernel_size=2,
                stride=2
                ),
              conv_block=conv_block
              ),
            *[stack_block(
              in_f=in_f,
              out_f=out_f,
              kernel_size=3,
              block=same_conv_block,
              depth=1,
              out_gate=nn.Sequential(
                  nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                  ),
                  nn.Dropout(0.25)
                ),
              conv_block=conv_block
              ) for in_f,out_f in zip(channels[:-1],channels[1:])],
        )
    def forward(self, x):
        return self.enc_blocks(x)
class GoodFellowV2(nn.Module):
  def __init__(self):
    super(GoodFellowV2,self).__init__()
    self.encoder = EncoderBNDO(1,[64,128,512,512])
    self.decoder = BasicDecoder([512*3*3,256,512],7,dropout=0.25)

  def forward(self,x):
    x = self.encoder(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)