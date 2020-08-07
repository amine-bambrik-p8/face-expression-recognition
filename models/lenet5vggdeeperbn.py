import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder

from models.layers.conv_block import conv_block
from models.layers.same_conv import same_conv_block

def stack_block(in_f, out_f,kernel_size,*args, **kwargs):
    return nn.Sequential(
        same_conv_block(in_f,out_f,kernel_size,*args,**kwargs),
        same_conv_block(out_f,out_f,kernel_size,*args,**kwargs),
        same_conv_block(out_f,out_f,kernel_size,*args,**kwargs),
        same_conv_block(out_f,out_f,kernel_size,*args,**kwargs),
        nn.MaxPool2d(2,2)
    )
class LeNetVGGDeeperBN(nn.Module):
  def __init__(self):
    super(LeNetVGGDeeperBN,self).__init__()
    # self.pool = nn.MaxPool2d(2,2)
    # self.conv1_1 = nn.Conv2d(1,64,3,padding=1)
    # self.conv1_3 = nn.Conv2d(64,64,3,padding=1)
    # self.conv1_3 = nn.Conv2d(64,64,3,padding=1)
    # self.conv1_3 = nn.Conv2d(64,64,3,padding=1)
    self.conv1 = stack_block(1,64,3,conv_block=conv_block)
    # self.conv2_1 = nn.Conv2d(64,128,3,padding=1)
    # self.conv2_3 = nn.Conv2d(128,128,3,padding=1)
    # self.conv2_3 = nn.Conv2d(128,128,3,padding=1)
    # self.conv2_3 = nn.Conv2d(128,128,3,padding=1)
    self.conv2 = stack_block(64,128,3,conv_block=conv_block)
    # self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
    # self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
    # self.conv3_3 = nn.Conv2d(256,256,3,padding=1)
    # self.conv3_3 = nn.Conv2d(256,256,3,padding=1)
    self.conv3 = stack_block(128,256,3,conv_block=conv_block)
    # self.conv3_1 = nn.Conv2d(256,512,3,padding=1)
    # self.conv3_1 = nn.Conv2d(512,512,3,padding=1)
    # self.conv3_1 = nn.Conv2d(512,512,3,padding=1)
    # self.conv3_1 = nn.Conv2d(512,512,3,padding=1)
    self.conv4 = stack_block(256,512,3,conv_block=conv_block)
    # self.conv3_1 = nn.Conv2d(256,512,3,padding=1)
    # self.conv3_1 = nn.Conv2d(512,512,3,padding=1)
    # self.conv3_1 = nn.Conv2d(512,512,3,padding=1)
    # self.conv3_1 = nn.Conv2d(512,512,3,padding=1)
    self.conv5 = stack_block(512,512,3,conv_block=conv_block)

    self.decoder = BasicDecoder([512,1024,1024],7);

  def forward(self,x):
    # 48 x 48 x 1
    x = self.conv1(x)
    # 23 x 23 x 16
    x = self.conv2(x)
    # 10 x 10 x 32
    x = self.conv3(x)
    # 4 x 4 x 64
    x = self.conv4(x)
    x = self.conv5(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)
  