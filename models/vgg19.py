import torch
import torch.nn as nn
import torch.nn.functional as F


def same_conv_block(in_f,out_f,kernel_size=(3,3),*args,**kwargs):
    padding=1
    if isinstance(kernel_size,tuple):
      padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
      padding = kernel_size // 2
    return conv_block(in_f,out_f,kernel_size=kernel_size,padding=padding)

def conv_block(in_f, out_f,kernel_size,*args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f,kernel_size=kernel_size, *args, **kwargs),
        nn.ReLU(inplace=True)
    )
def stack_block(in_f, out_f,kernel_size,depth=1,*args, **kwargs):
    return nn.Sequential(
        same_conv_block(in_f,out_f,kernel_size,*args,**kwargs),
        *[same_conv_block(out_f,out_f,kernel_size,*args,**kwargs) for i in range(1,depth)],
        nn.MaxPool2d(2,2)
    )
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        stack = [1,64,128,256,512,512]
        depth = [2,2,4,4,4]
        self.features = nn.Sequential(
            *[
                stack_block(in_c,out_c,3,depth=dp) for dp,in_c,out_c in zip(depth,stack[:-1],stack[1:])
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