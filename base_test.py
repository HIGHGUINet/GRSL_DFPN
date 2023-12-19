import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from PIL import Image
from torchvision.transforms import ToTensor
import utils.util as util

from model import DFPN

import argparse
import yaml

# Argument
BATCH_SIZE = 1          # Batch_size
DATASET = 'RSHaze.yml'  # dataset name
VISUALIZE = True        # visdom visualization
SEED = 5345312          # Set random seed
DATA_DIR = ""           # dataset dir
CONFIG_PATH = ""        # config path

def parse_args_and_config():

    parser = argparse.ArgumentParser(description='Baseline for Satellite Image ...')
    parser.add_argument('--config', type=str, default=DATASET, help="Path to the config file")
    parser.add_argument('--root_folder', type=str, default='/data/dataset/', help="Root folder of dataset ('D:/dataset/', '/data/dataset')") 
    parser.add_argument('--seed', type=int, default=SEED, metavar='N', help='Seed for initializing training (default: 654546)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--visualize', type=int, default=VISUALIZE, metavar='N', help='visualize using visdom (default: True)')

    args = parser.parse_args()

    print(args)

    with open(os.path.join('configs', args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():

    # Argument and configuration
    args, config = parse_args_and_config()

    # Load model
    model = DFPN().eval().cuda()
    state_dict = torch.load('./weights/RSHaze_train.pth')

    model.load_state_dict(state_dict)

    # Test image path 
    if DATASET == 'RSHaze.yml':
        val_path = './sample_data/'        
        
        val_name = sorted(os.listdir(val_path))        
    

    # metric

    save_path = 'results/'
    with torch.no_grad():
    
        for i in range(len(val_name)):
        # for i in range(3):
            val_image = Image.open(os.path.join(val_path, val_name[i])).convert('RGB')

            val_image = ToTensor()(val_image).unsqueeze(0).cuda()

            val_result, _, _ = model(val_image)

            val_result = util.tensor2im_gt(val_result)

            if not os.path.exists("./results/" + config.test.name):
                os.makedirs("./results/" + config.test.name)

                util.save_image(val_result, save_path + config.test.name + '/' + val_name[i][:-3] + 'png')

            else:
                util.save_image(val_result, save_path + config.test.name + '/' + val_name[i][:-3] + 'png')
           

            print(str(i) + ' done')


if __name__ == '__main__':
    main()