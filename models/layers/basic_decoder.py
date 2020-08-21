import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
import torch 
def dec_block(in_f, out_f,dropout=0.0,activation=nn.ReLU(True),batch_norm=True):
    n=nn.Linear(in_f, out_f,bias=False)
    b=nn.BatchNorm1d(out_f) if(batch_norm) else nn.Identity()
    torch.nn.init.kaiming_normal_(n.weight)
    return nn.Sequential(
        n,
        b,
        activation,
        nn.Dropout(dropout) if(dropout > 0.0) else nn.Identity()
    )
class BasicDecoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        dec_sizes=config.decoder_channels
        n_classes=config.n_classes
        dropout=config.decoder_dropout
        batch_norm=config.decoder_batch_norm
        self.dropout = nn.Dropout(dropout)
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f,dropout=dropout,batch_norm=batch_norm,activation=globals()[config.decoder_fn](*config.decoder_fn_params)) 
                    for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes,bias=False)
        torch.nn.init.kaiming_normal_(self.last.weight)
    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.dec_blocks(x)
        return self.last(x)