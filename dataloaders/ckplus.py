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

class CKPlusDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        if self.config.data_mode == "download":
            raise NotImplementedError("This mode is not implemented YET")
        elif self.config.data_mode == "imgs":
            dataset = torchvision.datasets.ImageFolder(
                root=self.config.data_folder,
                transform=transforms.Compose([
                    transforms.Grayscale(1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5),(0.5))
                ])
            )
            num_train = len(dataset)
            self.valid_size = int(num_train * self.config.valid_size)
            self.test_size = int(num_train * self.config.test_size)
            self.train_size = num_train - self.test_size - self.valid_size

            valid_idx,test_idx,train_idx, = torch.utils.data.random_split(range(num_train),[
                self.valid_size,
                self.test_size,
                self.train_size
                ]
            )
            
            train_sampler = SubsetRandomSampler(train_idx.indices)
            test_sampler = SubsetRandomSampler(test_idx.indices)
            valid_sampler = SubsetRandomSampler(valid_idx.indices)        
            self.train_loader = torch.utils.data.DataLoader(dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,sampler=train_sampler)
            self.test_loader = torch.utils.data.DataLoader(dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,sampler=test_sampler)
            self.valid_loader = torch.utils.data.DataLoader(dataset,batch_size=self.config.batch_size,num_workers=self.config.data_loader_workers,sampler=valid_sampler)
            classes = ("anger","contempt","disgust","fear","happy","sadness","surprise")
            
        elif self.config.data_mode == "numpy":
            raise NotImplementedError("This mode is not implemented YET")

        elif config.data_mode == "random":
            raise NotImplementedError("This mode is not implemented YET")

        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return imageio.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(imageio.imread(img_epoch))
            except OSError as e:
                pass

        imageio.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass