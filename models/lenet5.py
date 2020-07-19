import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder

class LeNet5(nn.Module):
  def __init__(self):
    super(LeNet5,self).__init__()
    self.conv1 = nn.Conv2d(1,6,5)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6,16,5)
    self.decoder = BasicDecoder([16*9*9,256,512],7,batch_norm=False);

  def forward(self,x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)