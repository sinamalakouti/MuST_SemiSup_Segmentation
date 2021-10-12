import torch
from utils import ramps


def mse_power(x, y, power=3):
    temp = (x ** power - y ** power) ** 2
    return temp.mean()


def regularization(segments, alpha=0.02):
    n_subjects = segments.shape[0]
    err = 0
    for i in range(n_subjects):
        item = segments[i]
        err += __cos_sim_all_channels(item, alpha)
    return err / n_subjects * alpha


def __cos_sim_all_channels(segments, alpha=0.013):
    n_segments = segments.shape[0]

    cos_sim = torch.nn.CosineSimilarity()
    err = 0
    for i in range(n_segments):
        in1 = segments[i, :, :].reshape(1, 212 * 256)
        for j in range(n_segments):
            if i == j:
                continue
            in2 = segments[j, :, :].reshape(1, 212 * 256)
            sim = cos_sim(in1, in2)
            err += torch.abs(sim)
    return err


def dice_coef_loss(y_pred, y_true, smooth=1):
    y_pred = y_pred.view(-1, 1)
    y_true = y_true.view(1, -1)
    numer = 2 * y_true @ y_pred
    denom = y_true.sum() + y_pred.sum()
    dsc_score = (numer + smooth) / (denom + smooth)
    return 1 - dsc_score


def DSC_BCE_loss(y_pred, y_true, smooth=1):
    dsc = dice_coef_loss(y_pred, y_true, smooth)
    bce = torch.nn.BCEWithLogitsLoss()
    return


class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='exp_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


class soft_dice_loss(torch.nn.Module):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x c x * X x Y( x Z...)  One hot encoding of ground truth
        y_  pred: b x c x X x Y( x Z...)  Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    def __init__(self, ):
        super(soft_dice_loss, self).__init__()

    # skip the batch and class axis for calculating Dice score
    def __call__(self, y_pred, y_true, epsilon=1):
        axes = tuple(range(2, len(y_pred.shape)))
        numerator = 2. * torch.sum(y_pred * y_true, axes)
        denominator = torch.sum(y_pred ** 2 + y_true ** 2, axes)

        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch
