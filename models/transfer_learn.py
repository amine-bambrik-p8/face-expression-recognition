
import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
import torchvision
class ResNet18(nn.Module):
    def __init__(self,config):
        super(ResNet18,self).__init__()
        self.model_conv = torchvision.models.resnet18(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Sequential(
            nn.Linear(num_ftrs,7)
        )
    def forward(self,x):
        x = self.model_conv(x)
        return F.log_softmax(x,dim=1)
    def parameters(self):
        return self.model_conv.fc.parameters()
