"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging
from agents.base import BaseAgent

class CDCGanAgent(BaseAgent):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        # define model
        self.gmodel = globals()[self.config.gmodel]()
        self.dmodel = globals()[self.config.dmodel]()
        # define data_loader
        self.data_loader = globals()[self.config.data_loader](self.config)
        # define loss
        self.loss = globals()[self.config.loss_function]()
        goptim_m = globals()[self.config.goptimizer]
        # define optimizer
        self.goptimizer = getattr(goptim_m,"optimizer")(self.gmodel,config)
        doptim_m = globals()[self.config.doptimizer]
        # define optimizer
        self.optimizer = getattr(doptim_m,"optimizer")(self.dmodel,config)
        self.checkpoint_dir = "experiments/{}/checkpoints/".format(self.config.exp_name)
        
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0
        self.predictions = torch.tensor([],dtype=torch.int)
        self.labels = torch.tensor([],dtype=torch.int)

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
            self.gmodel = self.gmodel.to(self.device)
            self.dmodel = self.dmodel.to(self.device)
            self.loss = self.loss.to(self.device)
            self.logger.info("Program will run on *****GPU-CUDA*****  Using {} model\n".format(config.model))
            # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU***** Using {} model\n".format(config.model))
        self.summary_writer = SummaryWriter(log_dir="./experiments/{}/summaries".format(self.config.exp_name), comment=self.config.model)

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """


    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self,epoch):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self,epoch):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError
    def test(self):
        """Run test data through model
        :return:
        """
        raise NotImplementedError
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError