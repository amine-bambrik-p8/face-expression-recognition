import torch.nn as nn
import torch.nn.functional as F
def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Dropout(0.25),
    )
class BasicDecoderBNDO(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f) 
                    for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)
    def forward(self, x):
        x = self.dec_blocks(x)
        return self.last(x)