import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.inception_block import InceptionBlock

def block(in_f, out_f,with_dropout=False):
    return nn.Sequential(
        InceptionBlock(in_f, out_f,kernel_size=4,padding=2),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        InceptionBlock(out_f, out_f),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.MaxPool2d(2,2,kernel_size=4,padding=2),
        nn.Dropout2d(0.1) if(with_dropout) else nn.Identity()
    )
class GoodFellowV3Inception(nn.Module):
  def __init__(self):
    super(GoodFellowV3Inception,self).__init__()
    self.block1 = block(1,64)
    self.block2 = block(64,64)
    self.block3 = block(64,128)
    self.decoder = BasicDecoder([128*6*6,1024,1024],7,dropout=0.1);

  def forward(self,x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)