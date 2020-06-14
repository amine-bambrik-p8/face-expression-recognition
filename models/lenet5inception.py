import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import InceptionBlock

def block(in_f, out_f):
    return nn.Sequential(
        InceptionBlock(in_f, out_f),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
class LeNetInception(nn.Module):
  def __init__(self):
    super(LeNetInception,self).__init__()
    self.block1 = block(1,16)
    self.block2 = block(16,32)
    self.block3 = block(32,64)
    self.decoder = BasicDecoder([64*6*6,256,512],7);

  def forward(self,x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)