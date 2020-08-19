from torch import nn
import torch
import torchvision
import numpy as np
import os

from setup import process_config
from models.goodfellow_v3 import GoodFellowV3Inference
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
x = torch.randn(48*48*1)
model = GoodFellowV3Inference(config)
load_checkpoint(model,config,checkpoint_dir)
model.eval()
torch.onnx.export(model, x, '{}_onnx_model.onnx'.format(config.exp_name), verbose=True)
