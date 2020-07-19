import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
from models.layers.conv_block import conv_block
from models.layers.same_conv import same_conv_block
def stack_block(in_f, out_f,kernel_size,*args, **kwargs):
    return nn.Sequential(
        same_conv_block(in_f,out_f,kernel_size,*args,**kwargs),
        conv_block(out_f,out_f,kernel_size),
        nn.MaxPool2d(2,2)
    )
class LeNetVGGBN(nn.Module):
  def __init__(self):
    super(LeNetVGGBN,self).__init__()
    # self.pool = nn.MaxPool2d(2,2)
    # self.conv1_1 = nn.Conv2d(1,16,3,padding=1)
    # self.conv1_2 = nn.Conv2d(16,16,3)
    self.conv1 = stack_block(1,16,3,conv_block=conv_block)
    # self.conv2_1 = nn.Conv2d(16,32,3,padding=1)
    # self.conv2_2 = nn.Conv2d(32,32,3)
    self.conv2 = stack_block(16,32,3,conv_block=conv_block)
    # self.conv3_1 = nn.Conv2d(32,64,3,padding=1)
    # self.conv3_2 = nn.Conv2d(64,64,3)
    self.conv3 = stack_block(32,64,3,conv_block=conv_block)

    self.decoder = BasicDecoder([64*4*4,256,512],7);

  def forward(self,x):
    # 48 x 48 x 1
    # x = F.relu(self.conv1_1(x))
    # x = self.pool(F.relu(self.conv1_2(x)))
    x = self.conv1(x)
    # 23 x 23 x 16
    # x = F.relu(self.conv2_1(x))
    # x = self.pool(F.relu(self.conv2_2(x)))
    x = self.conv2(x)
    # 10 x 10 x 32
    # x = F.relu(self.conv3_1(x))
    # x = self.pool(F.relu(self.conv3_2(x)))
    x = self.conv3(x)
    # 4 x 4 x 64
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)
  