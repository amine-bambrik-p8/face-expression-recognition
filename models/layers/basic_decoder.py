import torch.nn as nn
import torch.nn.functional as F
def dec_block(in_f, out_f,dropout=None,batch_norm=True):
    return nn.Sequential(
        nn.Linear(in_f, out_f),        
        nn.BatchNorm1d(out_f) if(batch_norm) else nn.Identity(),
        nn.ReLU(),
        nn.Dropout(dropout) if(dropout in not None) else nn.Identity(),
    )
class BasicDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes,dropout=None,batch_norm=True):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f,dropout,batch_norm) 
                    for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)
    def forward(self, x):
        x = self.dec_blocks(x)
        return self.last(x)