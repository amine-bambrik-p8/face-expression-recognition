"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torch.nn.functional as F

from agents.base import BaseAgent

from models import *
from dataloaders import *

from tensorboardX import SummaryWriter
# from utils.metrics import AverageMeter, AverageMeterList
# from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class LeNet5Agent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = globals()[config.model]()

        # define data_loader
        self.data_loader = globals()[config.data_loader](config)

        # define loss
        self.loss = nn.NLLLoss()

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA*****  Using {} model\n".format(config.model))
            # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU***** Using {} model\n".format(config.model))

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment=config.model)

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + self.config.exp_name + "checkpoint.pth.tar"
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")
        pass

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + self.config.exp_name + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch(epoch)
            self.validate()

            self.current_epoch += 1
    def train_one_epoch(self,epoch):
        """
        One epoch of training
        :return:
        """
        running_loss = 0.0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == self.config.log_interval-1:
                running_loss=running_loss / self.config.log_interval
                self.summary_writer.add_scalar('training_loss',
                            running_loss,
                            epoch * len(self.data_loader.train_loader) + batch_idx)
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch,self.config.batch_size * batch_idx,  self.data_loader.train_size,
                           100.0*self.config.batch_size * batch_idx / self.data_loader.train_size, running_loss))
                running_loss=0.0
            self.current_iteration += 1
    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.valid_loader)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, self.data_loader.valid_size,
            100.0*correct / self.data_loader.valid_size))
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        dataiter = iter(self.data_loader.train_loader)
        images, labels = dataiter.next()
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)
        # write to tensorboard
        self.summary_writer.add_image('four_fer_images', img_grid)
        self.writer.close()
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.data_loader.finalize()
        pass