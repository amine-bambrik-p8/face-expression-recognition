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
              in_f=1,
              out_f=64,
              kernel_size=7,
              block=same_conv_block,
              depth=2,
              out_gate=nn.MaxPool2d(
                kernel_size=2,
                stride=2
                ),
              conv_block=conv_block
              )
    self.block1 = block(64,128)
    self.block2 = block(128,256)
    self.block3 = block(256,512)
    self.decoder = BasicDecoder([512*6*6,1024,1024],7)

  def forward(self,x):
    x = self.gate(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)