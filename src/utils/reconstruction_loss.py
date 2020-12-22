import torch

def mse_power( x, y,power=3):
    temp = (x ** power - y ** power) **2
    return temp.mean()

def regularizaton(segments,alpha=0.02):

    n_subjects = segments.shape[0]
    err = 0
    for i in range(n_subjects):
        item = segments[i]
        err += __coss_sim_all_channels(item,alpha)
    return err /n_subjects * alpha



def __coss_sim_all_channels(segments,alpha=0.013):
    n_segments = segments.shape[0]

    cos_sim = torch.nn.CosineSimilarity()
    err = 0
    for i in range(n_segments):
        in1 = segments[i,:,:].reshape(1,212*256)
        for j in range(n_segments):
            if i == j:
                continue
            in2 = segments[j, :, :].reshape(1,  212 * 256)
            sim = cos_sim(in1,in2)
            err += torch.abs(sim)
    return err


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)

    return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch
