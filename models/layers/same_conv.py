import torch.nn as nn
def same_conv_block(in_f,out_f,kernel_size=(3,3),conv_block=nn.Conv2d,*args,**kwargs):
    padding=1
    if isinstance(kernel_size,tuple):
      padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
      padding = kernel_size // 2
    return conv_block(in_f,out_f,kernel_size=kernel_size,padding=padding,*args,**kwargs)
