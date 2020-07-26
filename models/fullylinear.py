
import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder import BasicDecoder
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.decoder = BasicDecoder([48*48,1024,2048,1024,512,64],7,batch_norm=False);
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.decoder(x)
        return F.log_softmax(x,dim=1)
