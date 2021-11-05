import torch
from utils import ramps
import torch.nn.functional as F
import torch.nn as nn


class Consistency_CE:
    def __init__(self, n_cls):
        super(Consistency_CE, self).__init__()
        self.centers = [None for i in range(0, n_cls)]
        self.center_momentum = 0.9

    @torch.no_grad
    def update_center(self, center_id, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        self.centers[center_id] *= self.center_momentum
        self.centers[center_id] += batch_center * (1 - self.center_momentum)

    def __call__(self, stud_logit, teach_output, center_id, use_softmax=True):

        if self.centers[center_id] is None:
            self.centers[center_id] = torch.zeros(
                (1, teach_output.shape[1], teach_output.shape[2], teach_output.shape[3])).to(teach_output.device)
        if use_softmax:
            center = self.centers[center_id]
            teach_pred = torch.nn.functional.softmax((teach_output - center) / 0.2, dim=1)
        self.update_center(center_id, teach_output)
        loss = - torch.mean(torch.sum(teach_pred.detach()* torch.nn.functional.log_softmax(stud_logit / 0.5, dim=1), dim=1))

        return loss


def softmax_ce_consistency_loss(x, y):
    return - torch.mean(
        torch.sum(y.detach()
                  * F.log_softmax(x, dim=1), dim=1))


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs / 0.5, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='batchmean')


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

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='linear_rampup'):
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


class abCE_loss(nn.Module):
    """
    Annealed-Bootstrapped cross-entropy loss
    """

    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,
                 reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):
        super(abCE_loss, self).__init__()
        self.weight = torch.FloatTensor(weight) if weight is not None else weight
        self.reduction = reduction
        self.thresh = thresh
        self.min_kept = min_kept
        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func = getattr(ramps, ramp_type)
            self.iters_per_epoch = iters_per_epoch
            self.num_classes = num_classes
            self.start = 1 / num_classes
            self.end = 0.9
            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter, epoch):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)
        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter, epoch):
        batch_kept = self.min_kept * target.size(0)
        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == ignore_index] = 0
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1, ) != ignore_index
        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:
            thresh = self.threshold(curr_iter=curr_iter, epoch=epoch)
        else:
            thresh = self.thresh

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0
        threshold = max(min_threshold, thresh)
        loss_matrix = F.cross_entropy(predict, target,
                                      weight=self.weight.to(predict.device) if self.weight is not None else None,
                                      ignore_index=ignore_index, reduction='none')
        loss_matirx = loss_matrix.contiguous().view(-1, )
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:
            return select_loss_matrix.sum()
        elif self.reduction == 'mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')
