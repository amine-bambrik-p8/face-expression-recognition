import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import InceptionBlock

def block(in_f, out_f,dropout=0.0):
    return nn.Sequential(
        InceptionBlock(in_f, out_f,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        InceptionBlock(out_f, out_f,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        InceptionBlock(out_f, out_f,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.Conv2d(out_f,out_f,kernel_size=2,stride=2),
        nn.Dropout2d(dropout) if(dropout>0.0) else nn.Identity()
    )
class GoodFellowV3InceptionDown(nn.Module):
  def __init__(self):
    super(GoodFellowV3InceptionDown,self).__init__()
    self.block1 = block(1,64)
    self.block2 = block(64,64)
    self.block3 = block(64,128,dropout=0.1)
    self.decoder = BasicDecoder([128*6*6,1024,1024],7,dropout=0.5)

  def forward(self,x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)