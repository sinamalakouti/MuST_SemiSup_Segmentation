import numpy as np
import random
import torchvision.transforms.functional as F
from torch.distributions.uniform import Uniform
import torch
import torch.nn as nn
from torch.autograd import Variable


class FeatureDropDecoder(nn.Module):
    def __init__(self):
        super(FeatureDropDecoder, self).__init__()

    def f_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, y):
        x = self.f_dropout(x)

        return x, y


class DropOutDecoder(nn.Module):
    def __init__(self, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)

    def forward(self, x, y):
        x = torch.nn.functional.dropout(x, 0.3, training=True)
        x = self.dropout(x)
        return x, y


class RotationDecoder(nn.Module):
    def __init__(self, angle_min=-180, angle_max=180):
        super(RotationDecoder, self).__init__()

        self.angle_min = angle_min
        self.angle_max = angle_max

    def forward(self, x, y):
        angle = np.random.uniform(self.angle_min, self.angle_max)
        x_transform = F.affine(x,
                               angle=angle, translate=(0, 0), shear=0, scale=1)
        y_transform = F.affine(y,
                               angle=angle, translate=(0, 0), shear=0, scale=1)

        return x_transform, y_transform


class ScaleDecoder(nn.Module):
    def __init__(self, scale_min=0.8, scale_max=1.2):
        super(ScaleDecoder, self).__init__()
        self.angle_min = scale_min
        self.angle_max = scale_max

    def forward(self, x, y):
        scale = np.random.uniform(.8, 1.2)

        x_transform = F.affine(x,
                               angle=0, translate=(0, 0), shear=0, scale=scale)
        y_transform = F.affine(y,
                               angle=0, translate=(0, 0), shear=0, scale=scale)
        return x_transform, y_transform


class UniformNoiseDecoder(nn.Module):

    def __init__(self, noise_min=-0.3, noise_max=0.3):
        super(UniformNoiseDecoder, self).__init__()
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.uni_dist = Uniform(self.noise_min, self.noise_max)

    def forward(self, x, y):
        noise = self.uni_dist.sample(x.shape[1:]).to(x.device)
        x_transform = x.mul(noise) + x
        return x_transform, y


class GaussianNoiseDecoder(nn.Module):
    def __init__(self, mean=0, std=1):
        super(GaussianNoiseDecoder, self).__init__()
        self.mu = mean
        self.sigma = std

    def forward(self, x, y):
        x_transform = x + torch.randn(x.size()).to(x.device) * self.sigma + self.mu
        return x_transform, y


class HFlipDecoder(nn.Module):

    def forward(self, x, y):
        super(HFlipDecoder, self).__init__()
        x_transform = F.hflip(x)
        y_transform = F.hflip(y)
        return x_transform, y_transform


class VFlipDecoder(nn.Module):

    def forward(self, x, y):
        super(VFlipDecoder, self).__init__()
        x_transform = F.vflip(x)
        y_transform = F.vflip(y)
        return x_transform, y_transform


class MixupDecoder(nn.Module):
    def __init__(self):
        None

    def forward(self, x_transform, y_transform):
        None


def get_r_adv(x, y_tr, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


class VATDecoder(nn.Module):
    def __init__(self, xi=1e-1, eps=10.0, iterations=1):
        super(VATDecoder, self).__init__()
        self.xi = xi
        self.eps = eps
        self.it = iterations

    def forward(self, x):
        r_adv = get_r_adv(x, self.it, self.xi, self.eps)
        return x + r_adv


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def gaussian(ins, is_training, stddev=0.2):
    if is_training:
        return ins + Variable(torch.randn(ins.size()).cuda() * stddev)
    return ins


def mixup(x_a, x_b, y_a, y_b):
    alpha = 0.4
    lam = np.random.beta(alpha, alpha)
    x_new = lam * x_a + (1 - lam) * x_b
    y_new = lam * y_a + (1 - lam) * y_b
    return x_new, y_new


def mixup_featureSpace(X, Y):
    p = np.random.permutation(X.shape[0])  # permutation
    X_permute = X[p]
    Y_permute = Y[p]
    X_transform, Y_transform = mixup(X, X_permute, Y, Y_permute)

    return X_transform, Y_transform


class Perturbator(nn.Module):

    def __init__(self):
        super(Perturbator, self).__init__()
        self.feature_dropout = FeatureDropDecoder()
        self.spatial_dropout = DropOutDecoder()
        self.rotation_decoder = RotationDecoder()
        self.scale_decoder = ScaleDecoder()
        self.hflip_decoder = HFlipDecoder()
        self.vflip_decoder = VFlipDecoder()
        self.uni_decoder = UniformNoiseDecoder()
        self.gaussian_decoder = GaussianNoiseDecoder()

    def forward(self, x, y, cascade=False):
        y = torch.nn.functional.softmax(y, dim=1)

        random_selector = np.random.randint(5)

        # print("random selector is ", random_selector)
        if cascade:
            None
            # x_transform = F.affine(x,
            #                        angle=0, translate=(0, 0), shear=0, scale=scale)
            # y_transform = F.affine(y,
            #                        angle=0, translate=(0, 0), shear=0, scale=scale)
            # x_transform = F.affine(x_transform,
            #                        angle=angle, translate=(0, 0), shear=0, scale=1)
            # y_transform = F.affine(y_transform,
            #                        angle=angle, translate=(0, 0), shear=0, scale=1)
            # noise = uni_dist.sample(x_transform.shape[1:]).to(x_transform.device)
            # x_transform = x_transform.mul(noise) + x_transform
            # y_transform = y_transform

            return None, None
        else:

            if random_selector == 0:  # feature drop out
                x_transform, y_transform = self.feature_dropout(x,y)

            elif random_selector == 1:  # spatial dropout
                x_transform, y_transform = self.spatial_dropout(x, y)

            elif random_selector == 2:  # uniform noise

                x_transform, y_transform = self.rotation_decoder(x, y)
                # noise = uni_dist.sample(x.shape[1:]).to(x.device)
                # x_transform = x.mul(noise) + x
                # y_transform = y

            elif random_selector == 3:  # rotate
                x_transform, y_transform = self.scale_decoder(x, y)

            elif random_selector == 4:  # scale
                rand = random.randint(0, 2)
                if rand == 0:
                    x_transform, y_transform = self.hflip_decoder(x, y)
                else:
                    x_transform, y_transform = self.vflip_decoder(x, y)

            if random_selector % 2 == 0:
                x_transform, y_transform = self.uni_decoder(x_transform, y_transform)
            else:
                x_transform, y_transform = self.gaussian_decoder(x_transform, y_transform)
            # elif random_selector == 5: #scale + feature dropout
            #
            #     x_transform = F.affine(x,
            #                            angle=0, translate=(0, 0), shear=0, scale=scale)
            #     y_transform = F.affine(y,
            #                            angle=0, translate=(0, 0), shear=0, scale=scale)
            #     module = FeatureDropDecoder()
            #     x_transform = module(x_transform)
            #     y_transform = y_transform
            # perform scaling
            # if random_selector == 0: #feature drop out
            #     module = FeatureDropDecoder()
            #     x_transform = module(x)
            #     y_transform = y
            #
            # elif random_selector == 1: #spatial dropout
            #     x_transform = torch.nn.functional.dropout(x, 0.3, training=True)
            #     y_transform = y
            #     # x_transform = F.affine(x,
            #     #                        angle=angle, translate=(0, 0), shear=0, scale=1)
            #     # y_transform = F.affine(y,
            #     #                        angle=angle, translate=(0, 0), shear=0, scale=1)
            # elif random_selector == 2: #uniform noise
            #
            #     noise = uni_dist.sample(x.shape[1:]).to(x.device)
            #     x_transform = x.mul(noise) + x
            #     y_transform = y
            #
            # elif random_selector == 3: #rotate
            #     x_transform = F.affine(x,
            #                            angle=angle, translate=(0, 0), shear=0, scale=1)
            #     y_transform = F.affine(y,
            #                            angle=angle, translate=(0, 0), shear=0, scale=1)
            #
            # elif random_selector == 4: #scale
            #     x_transform = F.affine(x,
            #                            angle=0, translate=(0, 0), shear=0, scale=scale)
            #     y_transform = F.affine(y,
            #                            angle=0, translate=(0, 0), shear=0, scale=scale)
            # elif random_selector == 5: #scale + feature dropout
            #
            #     x_transform = F.affine(x,
            #                            angle=0, translate=(0, 0), shear=0, scale=scale)
            #     y_transform = F.affine(y,
            #                            angle=0, translate=(0, 0), shear=0, scale=scale)
            #     module = FeatureDropDecoder()
            #     x_transform = module(x_transform)
            #     y_transform = y_transform
            #
            # elif random_selector == 6: #rotate + feature dropout
            #     x_transform = F.affine(x,
            #                            angle=angle, translate=(0, 0), shear=0, scale=1)
            #     y_transform = F.affine(y,
            #                            angle=angle, translate=(0, 0), shear=0, scale=1)
            #     module = FeatureDropDecoder()
            #     x_transform = module(x_transform)
            #     y_transform = y_transform
            # elif random_selector == 7: # vertical flip
            #     x_transform = F.vflip(x)
            #     y_transform = F.vflip(y)
            # elif random_selector == 8: #horizontal flip
            #     x_transform = F.hflip(x)
            #     y_transform = F.hflip(y)
            # elif random_selector == 9: #feature_mixup
            #     x_transform, y_transform = mixup_featureSpace(x,y)
            # elif random_selector == 10:  # feature_mixup
            #     x_transform, y_transform = mixup_featureSpace(x, y)
            return x_transform, y_transform
