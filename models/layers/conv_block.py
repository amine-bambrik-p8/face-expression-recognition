import torch.nn as nn
import torch
def conv_block(in_f, out_f,kernel_size,batch_norm=True,dropout=0.0,activation=nn.ReLU(inplace=True),*args, **kwargs):
    c = nn.Conv2d(in_f, out_f,kernel_size=kernel_size,bias=False, *args, **kwargs)
    b = nn.BatchNorm2d(out_f) if(batch_norm) else nn.Identity()
    torch.nn.init.xavier_normal_(c.weight)
    return nn.Sequential(
        c,
        b,
        activation,
        nn.Dropout2d(dropout) if(dropout > 0.0) else nn.Identity()
    )