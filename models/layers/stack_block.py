import torch.nn as nn
def stack_block(in_f, out_f,kernel_size,block,depth=1,out_gate=nn.Identity(),in_gate=nn.Identity(),*args, **kwargs):
    return nn.Sequential(
        #in_gate,
        block(in_f,out_f,kernel_size,*args, **kwargs),
        *[block(out_f,out_f,kernel_size,*args, **kwargs) for i in range(1,depth)],
        out_gate
    )