from torch import nn
import torch
import torchvision
import numpy as np
import os

from setup import process_config
from models.goodfellow_v3 import GoodFellowV3

def load_checkpoint(model,config,checkpoint_dir):
    """
    Latest checkpoint loader
    :param file_name: name of the checkpoint file
    :return:
    """
    filename = checkpoint_dir + config.exp_name + "_" +config.label +"checkpoint.pth.tar"
    checkpoint = torch.load(filename,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    pass

config = process_config("configs/test.json")
config.mode="export"
checkpoint_dir = "experiments/{}/checkpoints/".format(config.exp_name)
x = torch.randn(128,1,48,48)
model = GoodFellowV3(config)
model.eval()
output = model(x)
log_probs, preds_tensor = torch.max(output, 1)
preds = preds_tensor.squeeze()
one_hot=torch.nn.functional.one_hot(preds,num_classes=7)

print(one_hot)
print(preds)
