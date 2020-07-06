"""
Mnist Data loader, as given in Mnist tutorial
"""
import pandas as pd
import numpy as np
import torch
from dataloaders.transforms import *

class CSVFERDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config
        data = pd.read_csv(self.config.csv_path)
        training_data = data[data["Usage"]=="Training"].sort_values("emotion");
        
        self.train_labels = training_data.emotion.values
        self.train = training_data.pixels.str.split(' ', expand=True).apply(pd.to_numeric, errors='coerce').values.reshape(-1,48*48)
        self.test_data = data[data["Usage"]=="PrivateTest"].sort_values("emotion");
        self.test_labels = self.test_data.emotion.values
        self.test = self.test_data.pixels.str.split(' ', expand=True).apply(pd.to_numeric, errors='coerce').values.reshape(-1,48*48)

        self.train_labels = torch.Tensor(self.train_labels)
        self.train = torch.Tensor(self.train)
        self.test_labels = torch.Tensor(self.test_labels)
        self.test = torch.Tensor(self.test)

        self.target_names = ("angry","disgust","fear","happy","sad","surprise","neutral")
        self.n_classes = len(self.target_names)

    def finalize(self):
        pass