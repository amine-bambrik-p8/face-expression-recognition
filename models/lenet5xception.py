import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import XceptionBlock


def block(in_f, out_f,dropout=0.0):
    return nn.Sequential(
        XceptionBlock(in_f, out_f,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        XceptionBlock(out_f, out_f,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Dropout2d(dropout) if(dropout>0.0) else nn.Identity()
    )
class LeNetInception(nn.Module):
  def __init__(self):
    super(LeNetInception,self).__init__()
    self.block1 = block(1,64)
    self.block2 = block(64,128)
    self.block3 = block(128,256)
    self.decoder = BasicDecoder([256*6*6,1024,1024],7);

  def forward(self,x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)