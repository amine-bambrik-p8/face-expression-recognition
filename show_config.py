import argparse
from pprint import pprint
from utils.dirs import create_dirs
from utils.config import get_config_from_json

def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    json_file = args.config
    config, _ = get_config_from_json(json_file)
    print(" The Configuration of your experiment ..")
    pprint(config)
    
if __name__ == '__main__':
    main()