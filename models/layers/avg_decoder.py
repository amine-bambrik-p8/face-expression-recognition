from torch import nn
class AvgDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, config):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.decoder = nn.Linear(config.decoders_channels[0],config.n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x