"""
Mnist Data loader, as given in Mnist tutorial
"""
import imageio
import torch
import torchvision
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from dataloaders.transforms import *
from utils.project_data import project_data
class ImageFERDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        test_transform_m = globals()[config.test_transform]
        train_transform_m = globals()[config.train_transform]
        train_transform = getattr(train_transform_m,"transform")()
        test_transform = getattr(test_transform_m,"transform")()
        train_dataset = torchvision.datasets.ImageFolder(
            root=self.config.train_datafolder,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=self.config.test_datafolder,
            transform=test_transform,
            )
        num_train = len(train_dataset)
        self.valid_size = int(num_train * self.config.valid_size)
        self.train_size = num_train - self.valid_size
        self.test_size = len(test_dataset)
        valid_idx,train_idx, = torch.utils.data.random_split(range(num_train),[
            self.valid_size,
            self.train_size
            ]
        )
        train_sampler = SubsetRandomSampler(train_idx.indices)
        valid_sampler = SubsetRandomSampler(valid_idx.indices)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,pin_memory=config.pin_memory,sampler=train_sampler)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,pin_memory=config.pin_memory)
        self.valid_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,pin_memory=config.pin_memory,sampler=valid_sampler)
        self.classes =sorted(("angry","disgust","fear","happy","sad","surprise","neutral"))
    def visualize_images(self,writer):
        train_dataiter = iter(self.train_loader)
        test_dataiter = iter(self.test_loader)
        images, labels = train_dataiter.next()

        # create grid of images
        train_img_grid = torchvision.utils.make_grid(images)
        
        images, labels = test_dataiter.next()
        test_img_grid = torchvision.utils.make_grid(images)
        
        # write to tensorboard
        writer.add_image('%s Train Data: %s transfrom'%(self.config.exp_name,self.config.train_transform), train_img_grid)
        writer.add_image('%s Test Data: %s transfrom'%(self.config.exp_name,self.config.test_transform), test_img_grid)
    def project_data(self,writer):
        project_data(writer,self.train_loader,self.classes)
    def visualize(self,writer):
        self.visualize_images(writer)
        self.project_data(writer)
    def finalize(self):
        pass