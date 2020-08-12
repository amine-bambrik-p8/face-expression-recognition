
import torch
from torch import nn
from torch.nn import *
from models.layers.stack_block import stack_block
from models.layers.same_conv import same_conv_block
def conv_block(in_f, out_f,kernel_size,batch_norm=True,dropout=0.0,activation=nn.ReLU(),*args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f,kernel_size=kernel_size, *args, **kwargs),
        nn.BatchNorm2d(out_f) if(batch_norm) else nn.Identity(),
        nn.LocalResponseNorm(out_f) if(batch_norm) else nn.Identity(),
        activation,
        nn.Dropout2d(dropout) if(dropout > 0.0) else nn.Identity()
    )
class NetInNetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, config):
        super().__init__()
        
        self.net_in_net = nn.Sequential(*[stack_block(
              in_f=in_c,
              out_f=out_c,
              kernel_size=1,
              block=conv_block,
              depth=1,
              activation=globals()[config.decoder_fn](*config.decoder_fn_params),
              dropout=config.decoder_dropout,
              batch_norm=config.decoder_batch_norm,
              ) for in_c,out_c in zip(config.decoder_channels[:-1],config.decoder_channels[1:]) ])
        self.decoder = nn.Sequential(
            nn.Conv2d(config.decoder_channels[-1],config.n_classes,kernel_size=1),
            nn.Softmax2d(),
            nn.Conv2d(config.n_classes,config.n_classes,kernel_size=6)
            )
        self.avg = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.net_in_net(x)
        x = self.decoder(x)
        print(x.shape)
        x = self.avg(x)
        x = torch.flatten(x,1)
        return x