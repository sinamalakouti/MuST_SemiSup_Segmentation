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

    def forward(self, x, y=None):
        x = self.f_dropout(x)

        return x, y


class DropOutDecoder(nn.Module):
    def __init__(self, drop_rate=0.1, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)

    def forward(self, x, y=None):
        x = torch.nn.functional.dropout(x, 0.1, training=True)
        x = self.dropout(x)
        return x, y


class RotationDecoder(nn.Module):
    def __init__(self, angle_min=-180, angle_max=180):
        super(RotationDecoder, self).__init__()

        self.angle_min = angle_min
        self.angle_max = angle_max

    def forward(self, x, y):
        if y is None:
            return x, None
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
        if y is None:
            return x, y
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
        x_transform = x + x.mul(torch.randn(x.size()).to(x.device) * self.sigma + self.mu)
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


# class MixFeat(nn.Module):
#     """MixFeat <https://openreview.net/forum?id=HygT9oRqFX>"""
#
#     def __init__(self, sigma=0.2, **kargs):
#         super().__init__(**kargs)
#         self.sigma = sigma
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#     def call(self, inputs, training=None):  # pylint: disable=arguments-differ
#         def _passthru():
#             return inputs
#
#         def _mixfeat():
#             @tf.custom_gradient
#             def (x):
#                 shape = x.shape
#                 indices = torch.arange(start=0, stop=shape[0])
#                 indices = tf.random.shuffle(indices)
#                 rs = K.concatenate([K.constant([1], dtype="int32"), shape[1:]])
#                 r = K.random_normal(rs, 0, self.sigma, dtype="float16")
#                 theta = K.random_uniform(rs, -np.pi, +np.pi, dtype="float16")
#                 a = 1 + r * K.cos(theta)
#                 b = r * K.sin(theta)
#                 y = x * K.cast(a, K.floatx()) + K.gather(x, indices) * K.cast(
#                     b, K.floatx()
#                 )
#
#                 def _backword(dy):
#                     inv = tf.math.invert_permutation(indices)
#                     return dy * K.cast(a, K.floatx()) + K.gather(dy, inv) * K.cast(
#                         b, K.floatx()
#                     )
#
#                 return y, _backword
#
#             return _forward(inputs)
#
#         return K.in_train_phase(_mixfeat, _passthru, training=training)


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

    def __init__(self, cfg):
        super(Perturbator, self).__init__()
        self.feature_dropout = FeatureDropDecoder()
        self.spatial_dropout = DropOutDecoder()
        self.rotation_decoder = RotationDecoder()
        self.scale_decoder = ScaleDecoder()
        self.hflip_decoder = HFlipDecoder()
        self.vflip_decoder = VFlipDecoder()
        self.uni_decoder = UniformNoiseDecoder()
        self.gaussian_decoder = GaussianNoiseDecoder()
        self.cfg = cfg

    def __fw_geometrical_aug(self, x, y):
        random_selector = np.random.randint(4)

        if random_selector == 0:  # scale
            x_transform, y_transform = self.scale_decoder(x, y)
        elif random_selector == 1:  # rotate
            x_transform, y_transform = self.rotation_decoder(x, y)
        elif random_selector == 2:
            x_transform, y_transform = self.uni_decoder(x, y)
        elif random_selector == 3:
            x_transform, y_transform = x, y
        elif random_selector == 4:  # hflip
            x_transform, y_transform = self.hflip_decoder(x, y)
        elif random_selector == 5:  # vflip
            x_transform, y_transform = self.vflip_decoder(x, y)
        elif random_selector == 6:
            x_transform, y_transform = x,y
        return x_transform, y_transform

    def __fw_feature_space_aug(self, x, y):
        random_selector = np.random.randint(5)
        if random_selector == 0:  # feature drop out
            x_transform, y_transform = self.feature_dropout(x, y)
        elif random_selector == 1:  # spatial drop out
            x_transform, y_transform = self.spatial_dropout(x, y)
        elif random_selector == 2:  # uniform dist
            x_transform, y_transform = self.uni_decoder(x, y)
        elif random_selector == 3:
            x_transform, y_transform = self.gaussian_decoder(x, y)
        elif random_selector == 4:
            x_transform, y_transform = x, y
        return x_transform, y_transform

    def __fw_mix(self, x, y):
        None

    def forward(self, x, y, perturbation_mode='F', use_softmax=False,
                cascade=False):  # mode = F (feature_space), G (geometrical), M (mix)
        if use_softmax or perturbation_mode == 'G':
            y = torch.nn.functional.softmax(y, dim=1)
            assert False, "FUCK perturb softmax ( augment)"

        if perturbation_mode == 'G':
            x_transform, y_transform = self.__fw_geometrical_aug(x, y)
        elif perturbation_mode == 'F':
            x_transform, y_transform = self.__fw_feature_space_aug(x, y)
        else:
            x_transform, y_transform = None, None
        return x_transform, y_transform
