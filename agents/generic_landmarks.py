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
from agents.generic import GenericAgent
from models import *
from dataloaders import *
from agents.optimizers import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.plot_confusion_matrix import plot_confusion_matrix
cudnn.benchmark = True


class GenericAgentLandmarks(GenericAgent):

   
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
            images,landmarks, target = data[0].to(self.device),data[1].to(self.device), target.to(self.device)
            data= [images,landmarks]
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                total_running_loss += loss.item()
                pred,_ = self.output_to_probs(output)
                correct += (pred == target).sum().item()
                total += target.size(0)
            if batch_idx % self.config.log_interval == self.config.log_interval-1:
                running_loss=running_loss / self.config.log_interval
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch,total,  self.data_loader.train_size,
                           100.0*total / self.data_loader.train_size, running_loss))
                running_loss=0.0
            self.current_iteration += 1
        accuracy = 100*correct/total
        number_of_batches = len(self.data_loader.train_loader)
        total_running_loss /= number_of_batches
        self.logger.info('\Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            total_running_loss, correct, total,
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
                images,landmarks, target = data[0].to(self.device),data[1].to(self.device), target.to(self.device)
                data= [images,landmarks]
                output = self.model(data)
                valid_loss += self.loss(output, target).item()  # sum up batch loss
                
                pred,_=self.output_to_probs(output)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        number_of_batches = len(self.data_loader.valid_loader)
        valid_loss /= number_of_batches
        accuracy = 100.0*correct / total
        self.logger.info('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, correct, total,
            accuracy))
        return (valid_loss,accuracy)
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        predictions = None
        labels = None
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                images,landmarks, target = data[0].to(self.device),data[1].to(self.device), target.to(self.device)
                data= [images,landmarks]
                output = self.model(data)
                test_loss += self.loss(output, target).item()  # sum up batch loss
                pred,_=self.output_to_probs(output)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                if predictions is not None:
                    predictions = np.concatenate((predictions, pred.cpu().numpy()), axis=None)
                    labels = np.concatenate((labels,target.cpu().numpy()),axis=None)
                else:
                    predictions = pred.cpu().numpy()
                    labels =target.cpu().numpy()
        test_loss /= len(self.data_loader.test_loader)
        accuracy = 100.0*correct / total
        print(classification_report(labels, predictions, target_names=self.data_loader.classes))
        cm=confusion_matrix(labels, predictions, labels=range(len(self.data_loader.classes)))
        print(cm)
        fig = plot_confusion_matrix(cm,self.data_loader.classes)
        self.summary_writer.add_figure("Confusion matrix",fig)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total,
            accuracy))
        return accuracy
    def visualize(self):
        self.data_loader.visualize(self.summary_writer)
        dataiter = iter(self.data_loader.train_loader)
        data,_ = next(dataiter)
        images,landmarks = data[0].cpu(),data[1].cpu()
        self.model = self.model.cpu()
        self.summary_writer.add_graph(self.model, [images,landmarks])
    
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.summary_writer.close()
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.data_loader.finalize()
        pass
