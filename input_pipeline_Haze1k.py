import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from random import randrange, randint
import torch
import torchvision.transforms.functional as TF


class PairDataset(Dataset):

    def __init__(self, first_dir, second_dir, num_samples):
        """
        Arguments:
            first_dir, second_dir: strings, paths to folders with images.
            num_samples: an integer.
            image_size: a tuple of integers (height, width).
        """

        # Dehazing
        self.first_dir = first_dir
        self.second_dir = second_dir

        self.haze_names = sorted(os.listdir(first_dir))
        self.gt_names = sorted(os.listdir(second_dir))
        
        self.num_samples = num_samples

        size = 240
        self.transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomCrop((size, size)),
                # transforms.Resize((size, size), interpolation=Image.BICUBIC),
                # transforms.RandomResizedCrop((size, size), interpolation=Image.BICUBIC)
            ]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.totensor_gt = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        """
        Get a random pair of image crops.
        It returns a tuple of float tensors with shape [3, height, width].
        They represent RGB images with pixel values in [0, 1] range.
        """

        # RESIDE
        i = np.random.randint(0, len(self.haze_names))

        # Haze ---------------------------------------------------------------------------------------------------------
        name1, name2 = self.haze_names[i], self.gt_names[i]

        haze = Image.open(os.path.join(self.first_dir, name1))
        gt = Image.open(os.path.join(self.second_dir, name2))
        # --------------------------------------------------------------------------------------------------------------

        seed = np.random.randint(45545)

        random.seed(seed)
        torch.manual_seed(seed)
        haze = self.transform(haze)

        random.seed(seed)
        torch.manual_seed(seed)
        gt = self.transform(gt)

        haze = self.totensor(haze)
        gt = self.totensor_gt(gt)

        return haze, gt

