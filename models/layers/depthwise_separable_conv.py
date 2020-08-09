from torch import nn
class DSConv2d(nn.Module):
    def __init__(self, nin, nout,kernel_size=3):
        super(DSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=kernel_size//2, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out