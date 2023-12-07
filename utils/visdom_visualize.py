import numpy as np
from visdom import Visdom
import torch


class Visualizer():
    def __init__(self, port, server):

        self.viz = Visdom(port=port, server=server)

    def tensor2im(self, image_tensor, imtype=np.uint8):

        image_numpy = image_tensor.detach().cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)

        return image_numpy.astype(imtype)

    def visdom_image(self, img_dict):
            idx = 1
            for _, key in enumerate(img_dict):
                tensor_img = self.tensor2im(img_dict[key].data)
                self.viz.image(tensor_img.transpose([2, 0, 1]), opts=dict(title=key, width=240, height=240), win=idx)
                idx += 1

    def plot_current_errors(self, iter, errors):
        if not hasattr(self, 'plot_errors'):
            self.plot_errors = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_errors['X'].append(iter)
        self.plot_errors['Y'].append([errors[k] for k in self.plot_errors['legend']])
        self.viz.line(
            X=np.stack([np.array(self.plot_errors['X'])]*len(self.plot_errors['legend']), 1),
            Y=np.array(self.plot_errors['Y']),
            opts={'title': 'loss',
                  'legend': self.plot_errors['legend'],
                  'xlabel': 'iter',
                  'ylabel': 'loss'}, win=41)

    def plot_metrics(self, iter, metrics):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.plot_data['X'].append(iter)
        self.plot_data['Y'].append([metrics[k] for k in self.plot_data['legend']])
        self.viz.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={'title': 'metric',
                  'legend': self.plot_data['legend'],
                  'xlabel': 'iter',
                  'ylabel': 'loss'}, win=52)
