import torch.nn as nn
def conv_block(in_f, out_f,kernel_size,batch_norm=True,dropout=0.0,activation=nn.ReLU,*args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f,kernel_size=kernel_size, *args, **kwargs),
        nn.BatchNorm2d(out_f) if(batch_norm) else nn.Identity(),
        activation(),
        nn.Dropout2d(dropout) if(dropout > 0.0) else nn.Identity()
    )