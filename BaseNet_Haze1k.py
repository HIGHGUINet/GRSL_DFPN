import torch
import torch.nn as nn
import torch.optim as optim

from restormer_model import SIDFormer

from loss.loss_functions import ContentLoss, TVLoss, ColorAngleLoss, Get_gradient

import utils.util as util
from collections import OrderedDict
from torch.optim import lr_scheduler
from PIL import Image
from torchvision.transforms import ToTensor, GaussianBlur

import torch.nn.functional as F
import os
from math import log10
from IQA_pytorch import SSIM
from random import randint

import utils.visdom_visualize as viz
from focal_frequency_loss import FocalFrequencyLoss as FFL


class base_net:

    def __init__(self, config):

        # Set loss functions
        self.l1_criterion = nn.L1Loss()
        self.content_criterion = ContentLoss()

        self.ssim_loss = SSIM(3)

        self.ffl = FFL(loss_weight=1.0, alpha=1.0)

        # Betas
        betas = (0.9, 0.999)

        # Load network
        self.net = SIDFormer().cuda()

        # Optimizer
        self.optim = optim.AdamW(self.net.parameters(), lr=1e-4, betas=betas, weight_decay=1e-2)

        # Scheduler
        # Exponential scheduler
        # self.scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=0.90)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optim, T_max=900000, eta_min=1e-5)
        # self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10000, eta_min=1e-5)

        # Visdom
        self.viz = viz.Visualizer(config.visualize.port, config.visualize.server)

        # Number of Network Parameters for training
        net_params = sum(p.numel() for p in self.net.parameters())

        print('Net params:', format(net_params, ','))

        self.gaussian_blur = GaussianBlur(kernel_size=(15, 15), sigma=(3, 3))
       
    # Iteration network for train
    def train_step(self, haze, gt, path=None, num_iter=None, visualize=True):

        # Fed Network =======================================================================
        removed, low, high = self.net(haze)
        # ===================================================================================

        # Loss ==============================================================================
        # Perceptual loss
        # content_loss = self.content_criterion(removed, gt) * 1
        content_loss = torch.Tensor([0])

        # L1 loss
        l1_loss = self.l1_criterion(removed, gt)
        # ssim_loss = self.ssim_loss(removed, gt, as_loss=True)

        # low_freq loss
        gt_low = self.gaussian_blur(gt)
        # low_freq_loss = self.l1_criterion(low, gt_low) * 1
        low_freq_loss = self.ssim_loss(low, gt_low, as_loss=True) * 1

        # high_freq loss
        gt_high = gt - gt_low
        high_freq_loss = self.l1_criterion(high, gt_high) * 1
        
        # Additional losses
        # ff_loss = self.ffl(removed, gt)                                   # frequency loss
        # ssim_loss = self.ssim_loss(removed, gt, as_loss=True)           # ssim loss

        total_loss = l1_loss + low_freq_loss + high_freq_loss #+ content_loss #low_freq_st_loss
        #+ content_loss #+ ssim_loss * 0.5 # + l1_loss low_freq_ff_loss
        # ===================================================================================

        # Zero gradient
        self.optim.zero_grad()

        # Backward
        total_loss.backward()

        # Update parameter
        self.optim.step()

        # Scheduler step ====================================================================
        # for exponential scheduler
        # if num_iter % 50000 == 0:
        #     self.scheduler.step()

        self.scheduler.step()
        # ===================================================================================

        # Print learning rate
        if num_iter % 100 == 0:
            print('-----Current learning rate: {0}-----'.format(self.scheduler.get_last_lr()))

        lr = self.scheduler.get_last_lr()

        # Validation test during the train ==================================================
        if num_iter % 50 == 0:

            val_name = sorted(os.listdir(path[0]))
            gt_name = sorted(os.listdir(path[1]))

            val_result_list = []
            val_gt_list = []

            psnr_list = []
            ssim_list = []

            with torch.no_grad():

                # idx = randint(0, 10-1)
                idx = randint(0, len(val_name)-1)

                for i in range(9):
                # for i in range(idx, idx + 10):
                # for i in range(len(val_name)):

                    val_image = Image.open(os.path.join(path[0], val_name[i])).convert('RGB')
                    val_gt = Image.open(os.path.join(path[1], gt_name[i])).convert('RGB')

                    # val_image = self.resize(val_image)
                    val_image = (ToTensor()(val_image)).unsqueeze(0).cuda()

                    val_gt = ToTensor()(val_gt).unsqueeze(0).cuda()

                    val_result, _, _ = self.net.eval()(val_image)
                    val_result = val_result.clamp(0, 1)

                    val_result_list.append(val_result)
                    val_gt_list.append(val_gt)

                    psnr_list.extend(to_psnr(val_result, val_gt))
                    ssim_list.extend(self.ssim_loss(val_result, val_gt, as_loss=False))

                    # save results
                    # results = utils.tensor2im_gt(val_result)
                    # utils.save_image(results, './results/HSTS/' + 'pWAEDehaze_' + val_name[i][:-3] + 'png')

                # Calculate PSNR and SSIM
                avg_psnr = sum(psnr_list) / len(psnr_list)
                avg_ssim = sum(ssim_list) / len(ssim_list)

                # Print logs
                print('Iteration: {0} Contents: {1:.3f} SSIM: {2:.3f}'.format(num_iter, content_loss.item(), l1_loss.item()))
                print('Current PSNR: {0:.3f} / SSIM: {1:.4f}'.format(avg_psnr, avg_ssim))

                # Visdom visualization ======================================================
                if visualize:

                    # Graph --------------------------------------------------
                    loss_list = OrderedDict([
                        # Losses
                        ('content', content_loss.item()),
                        ('l1', l1_loss.item()),

                        # Learning rate
                        ('lr', lr[0]),
                    ])

                    self.viz.plot_current_errors(num_iter, loss_list)
                    # --------------------------------------------------------

                    # Image --------------------------------------------------
                    image_list = OrderedDict([
                        ('haze', haze[0]), ('dehaze', removed[0]), ('gt', gt[0]),
                        ('val', val_result_list[0][0]),

                        ('gt_low', gt_low[0]), ('gt_high', gt_high[0]), 
                        ('low', low[0]), ('high', high[0]),
                       
                        # ('conv_feat1', util.feat_visualization(conv_feat[0][0])),
                        # ('conv_feat2', util.feat_visualization(conv_feat[1][0])),
                        # ('conv_feat3', util.feat_visualization(conv_feat[2][0])),
                        # ('trans_feat1', util.feat_visualization(trans_feat[0][0])),
                        # ('trans_feat2', util.feat_visualization(trans_feat[1][0])),
                        # ('trans_feat3', util.feat_visualization(trans_feat[2][0])),

                    ])

                    self.viz.visdom_image(image_list)
                    # --------------------------------------------------------

                    # Metric -------------------------------------------------
                    metric_list = OrderedDict([
                        ('PSNR', avg_psnr),             # PSNR
                        ('SSIM', avg_ssim.item()),      # SSIM
                    ])

                    self.viz.plot_metrics(num_iter, metric_list)
                    # --------------------------------------------------------

                # ===========================================================================

        else:

            avg_psnr = 0
        # ===================================================================================

        return avg_psnr

    def save_model(self, model_path):

        if not os.path.exists("./weights"):
            os.makedirs("./weights")

            torch.save(self.net.state_dict(), model_path + '_train_Haze1k_thick.pth')

        else:
            torch.save(self.net.state_dict(), model_path + '_train_Haze1k_thick.pth')


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

