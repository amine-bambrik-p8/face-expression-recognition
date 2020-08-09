import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.conv_block import conv_block
from models.layers.same_conv import same_conv_block

def stack_block(in_f, out_f,kernel_size,depth=1,*args, **kwargs):
    return nn.Sequential(
        same_conv_block(in_f,out_f,kernel_size,*args,**kwargs),
        *[same_conv_block(out_f,out_f,kernel_size,*args,**kwargs) for i in range(1,depth)],
        nn.MaxPool2d(2,2)
    )
class VGG19(nn.Module):
    def __init__(self,config):
        super(VGG19, self).__init__()
        stack = [1,64,128,256,512,512]
        depth = [2,2,4,4,4]
        self.features = nn.Sequential(
            *[
                stack_block(in_c,out_c,3,depth=dp,conv_block=conv_block) for dp,in_c,out_c in zip(depth,stack[:-1],stack[1:])
            ],
            nn.AvgPool2d(kernel_size=1, stride=1)
        )
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return F.log_softmax(out,dim=1)