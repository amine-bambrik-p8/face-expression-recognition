from agents import *
from setup import process_config
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.plot_confusion_matrix import plot_confusion_matrix
cudnn.benchmark = True
class BootstrapTester(BaseAgent):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.agents = []
        self.load_checkpoint()
        self.data_loader= self.agents[0].data_loader
        self.summary_writer = SummaryWriter(log_dir="./experiments/{}/summaries".format(self.config.exp_name), comment=self.config.model)


    def load_checkpoint(self):
        for exp_name in self.config.exps:
            exp_path = exp_name
            config_exp = process_config(exp_path)
            self.agents.append(globals()[config_exp.agent](config_exp))

    def run(self):
        try:
            getattr(self,self.config.mode)()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")


    def train(self):
        for agent in self.agents:
            agent.config.mode="train"
            agent.run()
    
    def test(self):
        test_loss = 0
        correct = 0
        total = 0
        predictions = None
        labels = None

        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                vote = None
                for agent in self.agents:
                    data, target = data.to(agent.device), target.to(agent.device)
                    output = agent.model(data)
                    _,pred=torch.max(output,1)
                    one_hot=torch.nn.functional.one_hot(pred,num_classes=7)
                    if(vote is None):
                        vote = one_hot;
                    else:
                        vote += one_hot
                _,pred = torch.max(vote,1)
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
    def output_to_probs(self, output):
        '''
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        '''
        # convert output probabilities to predicted class
        log_probs, preds_tensor = torch.max(output, 1)
        preds = preds_tensor.squeeze()
        return preds, torch.exp(log_probs)
        
    def finalize(self):
        pass