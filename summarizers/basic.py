from utils.project_data import project_data
from torch.utils.tensorboard import SummaryWriter

class BasicSummarizer(BaseSummarizer):
    def __init__(self, config):
        super().__init__(config)
    def before_all(self):
        project_data(self.summary_writer,data_loader.train_loader,data_loader.classes)
        
    def visualize_images(self,images):
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)
        # write to tensorboard
        self.summary_writer.add_image('one_batch_of_fer_images(%d)' % self.config.batch_size, img_grid)
    def visualize_model(self,model,data):
        self.summary_writer.add_graph(model, images)
    def finalize():
        self.summary_writer.close()