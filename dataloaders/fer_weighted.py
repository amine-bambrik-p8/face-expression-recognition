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
from torch.utils.data.sampler import WeightedRandomSampler
from dataloaders.transforms import *
from utils.project_data import project_data
class ImageFERDataLoaderWeighted:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        test_transform_m = globals()[config.test_transform]
        train_transform_m = globals()[config.train_transform]
        valid_transform_m = globals()[config.valid_transform]
        train_transform = getattr(train_transform_m,"transform")()
        valid_transform = getattr(valid_transform_m,"transform")()
        test_transform = getattr(test_transform_m,"transform")()
        train_dataset = torchvision.datasets.ImageFolder(
            root=self.config.train_datafolder,
            transform=train_transform
        )
        valid_dataset = torchvision.datasets.ImageFolder(
            root=self.config.valid_datafolder,
            transform=valid_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=self.config.test_datafolder,
            transform=test_transform,
            )
        self.valid_size = len(valid_dataset)
        self.train_size = len(train_dataset)
        self.test_size = len(test_dataset)
        weights = self.make_weights_for_balanced_classes(train_dataset.imgs,len(train_dataset.classes),range(len(train_dataset.imgs)))
        weights = torch.DoubleTensor(weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,pin_memory=config.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,pin_memory=config.pin_memory)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,pin_memory=config.pin_memory)
        self.classes =sorted(("angry","disgust","fear","happy","sad","surprise","neutral"))
    def make_weights_for_balanced_classes(self,images, nclasses,indices):                        
        count = [0] * nclasses                                                      
        for index in indices:                                                         
            count[images[index][1]] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)                                              
        for index in indices:                                          
            weight[index] = weight_per_class[images[index][1]]                                  
        return weight
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