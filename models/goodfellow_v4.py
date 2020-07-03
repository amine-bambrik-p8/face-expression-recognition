import torch.nn as nn
import torch.nn.functional as F
from models.layers.basic_decoder_do import BasicDecoderDO
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
        nn.Dropout2d(0.5) if(with_dropout) else nn.Identity()
    )
class EncoderBNDO(nn.Module):
    def __init__(self):
        super().__init__()
        gate = nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.5),
        )
        self.enc_blocks = nn.Sequential(
            gate,
            enc_block(64,128,3,with_dropout=True),
            enc_block(128,256,3,with_dropout=True),
            enc_block(256,512,3,with_dropout=True)
        )
    def forward(self, x):
        return self.enc_blocks(x)
class GoodFellowV4(nn.Module):
  def __init__(self):
    super(GoodFellowV4,self).__init__()
    self.encoder = EncoderBNDO()
    self.decoder = BasicDecoderDO([512*3*3,256,128],7)

  def forward(self,x):
    x = self.encoder(x)
    x = x.view(x.size(0),-1)
    x = self.decoder(x)
    return F.log_softmax(x,dim=1)