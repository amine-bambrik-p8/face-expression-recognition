"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch.nn import *
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from utils.project_data import project_data
from agents.base import BaseAgent
from models import *
from dataloaders import *
from agents.optimizers import *
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True


class GenericAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)



        # define model
        self.model = globals()[self.config.model]()
        # define data_loader
        self.data_loader = globals()[self.config.data_loader](self.config)
        # define loss
        self.loss = globals()[self.config.loss_function]()
        optim_m = globals()[self.config.optimizer]
        # define optimizer
        self.optimizer = getattr(optim_m,"optimizer")(self.model,config)
        self.checkpoint_dir = "experiments/{}/checkpoints/".format(self.config.exp_name)
        


        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.predictions = np.array([])
        self.labels = np.array([])



        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING!: You have a CUDA device, so you should use it")

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

        self.summary_writer = SummaryWriter(log_dir="./experiments/{}/summaries".format(self.config.exp_name), comment=self.config.model)





        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.checkpoint_dir+self.config.exp_name+"checkpoint.pth.tar"
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.best_metric = checkpoint['accuracy']
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.checkpoint_dir))
            self.logger.info("**First time to train**")
        pass

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'accuracy':self.best_metric,
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.checkpoint_dir + self.config.exp_name + filename)

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            getattr(self,self.config.mode)()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            (train_loss,train_accuracy) = self.train_one_epoch()
            if epoch % self.config.validate_every == self.config.validate_every-1:
                (loss,accuracy) = self.validate()
                self.summary_writer.add_scalars('accuracy', {
                        'training':train_accuracy,
                        'validation':accuracy
                        }, epoch)
                self.summary_writer.add_scalars('loss', {
                        'training':train_loss,
                        'validation':loss
                        }, epoch)
                if self.best_metric < accuracy:
                    self.logger.info('Saving Model with accuracy %f previous best accuracy was %f \n'% (accuracy, self.best_metric))
                    self.best_metric = accuracy
                    self.save_checkpoint()
            self.current_epoch += 1
        
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        running_loss = 0.0
        correct = 0
        total = 0
        total_running_loss = 0.0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                total_running_loss += running_loss
                _,pred = output.max(1)  # get the index of the max log-probability
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                self.predictions = np.append(self.predictions,pred.cpu().numpy())
                self.labels = np.append(self.labels,target.cpu().numpy())
            if batch_idx % self.config.log_interval == self.config.log_interval-1:
                running_loss=running_loss / self.config.log_interval
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch,total,  self.data_loader.train_size,
                           100.0*total / self.data_loader.train_size, running_loss))
                running_loss=0.0
            self.current_iteration += 1
        total_running_loss /= len(self.data_loader.train_loader)
        accuracy = 100* correct / total
        self.logger.info('\Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, total,
            accuracy))
        return (total_running_loss,accuracy)
    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.data_loader.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                valid_loss += self.loss(output, target).item()  # sum up batch loss
                _,pred = output.max(1)  # get the index of the max log-probability
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        valid_loss /= len(self.data_loader.valid_loader)
        accuracy = 100.0*correct / total
        self.logger.info('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, total,
            accuracy))
        return (valid_loss,accuracy)
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.data_loader.test_loader)
        accuracy = 100.0*correct / self.data_loader.test_size
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, self.data_loader.test_size,
            accuracy))
        return accuracy
            
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.summary_writer.close()
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.data_loader.finalize()
        pass
