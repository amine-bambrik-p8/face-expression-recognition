import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
import torch
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
              kernel_size=4,
              block=same_conv_block,
              depth=2,
              out_gate=nn.MaxPool2d(
                kernel_size=2,
                stride=2
                ),
              conv_block=conv_block
              ),
            *[stack_block(
              in_f=in_f,
              out_f=out_f,
              kernel_size=4,
              block=same_conv_block,
              depth=2,
              out_gate=nn.Sequential(
                  nn.MaxPool2d(
                    kernel_size=2,
                    stride=2
                  ),
                  nn.Dropout(0.1)
                ),
              conv_block=conv_block
              ) for in_f,out_f in zip(channels[:-1],channels[1:])],
        )
    def forward(self, x):
        return self.enc_blocks(x)
class GoodFellowV3LandmarksDO(nn.Module):
  def __init__(self):
    super(GoodFellowV3Landmarks,self).__init__()
    self.encoder = EncoderBNDO(1,[64,64,128])
    self.decoder = BasicDecoder([128*7*7+68*2,1024,1024],7,dropout=0.1)

  def forward(self,x):
    x,landmarks = x[0],x[1]
    landmarks = landmarks.view(landmarks.size(0),-1)
    x = self.encoder(x)
    x = x.view(x.size(0),-1)
    x = torch.cat([x,landmarks],dim=1)
    x = F.batch_norm(x)
    x = F.dropout(x,training=self.train,p=0.3)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)