import numpy as np
import torchvision.transforms.functional as F
from torch.distributions.uniform import Uniform
import random
import torch


def augment(x, y, cascade=False):

    y = torch.nn.functional.softmax(y,dim=1)
    angle = np.random.uniform(-180, 180)
    scale = np.random.uniform(.8, 1.2)
    # shear = np.random.uniform(-30, 30)
    rand_thresh = random.uniform(0, 1)
    uni_dist = Uniform(-1 * rand_thresh, rand_thresh)

    random_selector = np.random.randint(3)
    # print("random selector is ", random_selector)
    if cascade:

        x_transform = F.affine(x,
                               angle=0, translate=(0, 0), shear=0, scale=scale)
        y_transform = F.affine(y,
                               angle=0, translate=(0, 0), shear=0, scale=scale)
        x_transform = F.affine(x_transform,
                               angle=angle, translate=(0, 0), shear=0, scale=1)
        y_transform = F.affine(y_transform,
                               angle=angle, translate=(0, 0), shear=0, scale=1)
        noise = uni_dist.sample(x_transform.shape[1:]).to(x_transform.device)
        x_transform = x_transform.mul(noise) + x_transform
        y_transform = y_transform

        return x_transform, y_transform
    else:
        # perform scaling
        if random_selector == 0:
            x_transform = F.affine(x,
                                   angle=0, translate=(0, 0), shear=0, scale=scale)
            y_transform = F.affine(y,
                                   angle=0, translate=(0, 0), shear=0, scale=scale)
        elif random_selector == 1:
            x_transform = F.affine(x,
                                   angle=angle, translate=(0, 0), shear=0, scale=1)
            y_transform = F.affine(y,
                                   angle=angle, translate=(0, 0), shear=0, scale=1)
        elif random_selector == 2:

            noise = uni_dist.sample(x.shape[1:]).to(x.device)
            x_transform = x.mul(noise) + x
            y_transform = y
        return x_transform, y_transform
