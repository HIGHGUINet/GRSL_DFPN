from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
import shutil
import torchvision
import matplotlib.pyplot as plt

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = denormalize(image_tensor)
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    # image_numpy = image_numpy * 0.5 - 0.5
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy.astype(imtype)


def tensor2im_gt(image_tensor, imtype=np.uint8):
    # image_tensor = denormalize(image_tensor)
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    # image_numpy = image_numpy * 0.5 - 0.5
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy.astype(imtype)


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    denorm = tensors.clone()

    for c in range(tensors.shape[1]):
        denorm[:, c] = denorm[:, c].mul_(std[c]).add_(mean[c])

    denorm = torch.clamp(denorm, 0, 255)

    return denorm


def save_image(image_numpy, image_path):
    image_pil = None
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0], image_numpy.shape[1]))
        image_pil = Image.fromarray(image_numpy, 'L')
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def set_seed(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def feat_visualization(features):
    
    # rand_feat = random.choice([0, 16, 32, 48])
    # grid = torchvision.utils.make_grid(features[rand_feat:rand_feat+16, :, :].unsqueeze(1), nrow=4, normalize=False, padding=0)
    grid = torchvision.utils.make_grid(features[:64, :, :].unsqueeze(1), nrow=8, normalize=False, padding=0)
    grid = plt.cm.viridis(grid[0].cpu().numpy())[..., :3]
    grid = torch.Tensor(grid).permute(2, 0, 1)

    return grid

# Copy code at the best PSNR
def copy_code(code_path):

    if not os.path.exists(code_path):
        os.makedirs(code_path)

        shutil.copy2("BaseNet_train.py", code_path + "BaseNet_train.py")              # main
        shutil.copy2("input_pipeline_RICE.py", code_path + "input_pipeline_RICE.py")    # dataloader
        shutil.copy2("BaseNet.py", code_path + "BaseNet.py")                    # train
        shutil.copy2("model.py", code_path + "model.py")                                # model

    else:
        shutil.copy2("BaseNet_train.py", code_path + "BaseNet_train.py")              # main
        shutil.copy2("input_pipeline_RICE.py", code_path + "input_pipeline_RICE.py")    # dataloader
        shutil.copy2("BaseNet.py", code_path + "BaseNet.py")                    # train
        shutil.copy2("model.py", code_path + "model.py")                                # model

