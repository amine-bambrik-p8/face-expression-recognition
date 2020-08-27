
import torch
from torch import nn
from torch.nn import *
from models.layers.stack_block import stack_block
from models.layers.conv_block import conv_block
from models.layers.same_conv import same_conv_block
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
              activation=globals()[config.encoder_fn](*config.encoder_fn_params),
              dropout=config.encoder_dropout,
              batch_norm=config.encoder_batch_norm,
              ) for in_c,out_c in zip(config.decoder_channels[:-1],config.decoder_channels[1:])])
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.decoder = nn.Linear(config.decoder_channels[-1],config.n_classes,bias=False)
        self.dropout = nn.Dropout(config.decoder_dropout) if config.decoder_dropout>0.0 else nn.Identity()
        torch.nn.init.kaiming_normal_(self.decoder.weight)
        

    def forward(self, x):
        x = self.net_in_net(x)
        x = self.avg(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.decoder(x)
        return x