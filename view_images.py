from dataloaders import *
from torch.utils.tensorboard import SummaryWriter
import argparse
from setup import process_config
import torchvision


def main():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    # parse the config json file
    config = process_config(args.config)
    data_loader = globals()[config.data_loader](config)
    summary_writer = SummaryWriter(log_dir=config.summary_dir, comment=config.model)
    dataiter = iter(data_loader.train_loader)
    images, labels = dataiter.next()
    print(images.shape)
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    summary_writer.add_image('one_batch_of_images(%d)' % config.batch_size, img_grid)
    summary_writer.close()


if __name__ == '__main__':
    main()


