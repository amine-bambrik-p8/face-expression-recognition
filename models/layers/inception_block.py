import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.layers.same_conv import same_conv_block

from models.layers.res_block import *
class XceptionBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,padding=0):
        super(XceptionBlock,self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels,out_channels//2,kernel_size=1,bias=False)
        torch.nn.init.kaiming_normal_(self.branch1x1.weight)
        self.branch3x3_1 = nn.Conv2d(in_channels,out_channels//8,kernel_size=1,bias=False)
        torch.nn.init.kaiming_normal_(self.branch3x3_1.weight)
        self.branch3x3_2 = same_conv_block(out_channels//8,out_channels//2,kernel_size=3,conv_block=nn.Conv2d,bias=False)
        torch.nn.init.kaiming_normal_(self.branch3x3_2.weight)

    def forward(self,x):
        out_branch1x1 = self.branch1x1(x)
        out_branch3x3 = self.branch3x3_1(x)
        out_branch3x3 = self.branch3x3_2(out_branch3x3)
        outputs = [out_branch1x1,out_branch3x3]
        outputs = torch.cat(outputs,1)
        return outputs

class InceptionBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionBlock,self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels,out_channels//4,kernel_size=1,bias=False)
        torch.nn.init.kaiming_normal_(self.branch1x1.weight)

        self.branch5x5_1 =  nn.Conv2d(in_channels,out_channels//8,kernel_size=1,bias=False)
        torch.nn.init.kaiming_normal_(self.branch5x5_1.weight)
        self.branch5x5_2 = same_conv_block(out_channels//8,out_channels//4,kernel_size=5,conv_block=nn.Conv2d,bias=False)
        torch.nn.init.kaiming_normal_(self.branch5x5_2.weight)

        self.branch3x3_1 =  nn.Conv2d(in_channels,out_channels//4,kernel_size=1,bias=False)
        torch.nn.init.kaiming_normal_(self.branch3x3_1.weight)
        self.branch3x3_2 = same_conv_block(out_channels//4,out_channels//2,kernel_size=3,conv_block=nn.Conv2d,bias=False)
        torch.nn.init.kaiming_normal_(self.branch3x3_2.weight)
    def forward(self,x):
        out_branch1x1 = self.branch1x1(x)
        
        out_branch5x5 = self.branch5x5_1(x)
        out_branch5x5 = self.branch5x5_2(out_branch5x5)

        out_branch3x3 = self.branch3x3_1(x)
        out_branch3x3 = self.branch3x3_2(out_branch3x3)

        outputs = [out_branch1x1,out_branch3x3,out_branch5x5]
        outputs = torch.cat(outputs,1)
        return outputs




