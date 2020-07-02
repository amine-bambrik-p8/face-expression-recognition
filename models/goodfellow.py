import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder_bn_do import BasicDecoderBNDO
def conv_block(in_f, out_f,kernel_size,*args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f,kernel_size=kernel_size, *args, **kwargs),
        nn.ReLU(),
        nn.BatchNorm2d(out_f)
    )
def same_conv_block(in_f,out_f,kernel_size=(3,3),*args,**kwargs):
    padding=1
    if isinstance(kernel_size,tuple):
      padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
      padding = kernel_size // 2
    return conv_block(in_f,out_f,kernel_size=kernel_size,padding=padding)

def enc_block(in_f, out_f,kernel_size,with_dropout=False,*args, **kwargs):
    return nn.Sequential(
        same_conv_block(in_f,out_f,kernel_size,*args, **kwargs),
        same_conv_block(out_f,out_f,kernel_size,*args, **kwargs),
        nn.MaxPool2d(2,2),
        nn.Dropout2d(0.1) if(with_dropout) else nn.Identity()
    )
class EncoderBNDO(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_blocks = nn.Sequential(
            enc_block(1,64,4,with_dropout=False),
            enc_block(64,64,4,with_dropout=True),
            enc_block(64,128,4,with_dropout=True)
        )
    def forward(self, x):
        return self.enc_blocks(x)
class GoodFellow(nn.Module):
  def __init__(self):
    super(GoodFellow,self).__init__()
    self.encoder = EncoderBNDO()
    self.decoder = BasicDecoderBNDO([128*7*7,1024,1024],7)

  def forward(self,x):
    x = self.encoder(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)