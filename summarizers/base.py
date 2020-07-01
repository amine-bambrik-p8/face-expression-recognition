from torch.utils.tensorboard import SummaryWriter

class BaseSummarizer:
    def __init__(self, config):
        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir="./experiments/{}/summaries".format(self.config.exp_name), comment=self.config.model)
    def before_all(self):
        raise NotImplementedError
    def after_all(self):
        raise NotImplementedError
    def visualize_images(self):
        raise NotImplementedError
    def visualize_model(self):
        raise NotImplementedError
    def finalize(self):
        raise NotImplementedError

