import numpy as np
import torch
from medpy import metric


# dice coefficient for predicted class
def dice_coef(y_true, y_pred, smooth=1):  # in order to get the
    y_pred = y_pred.float().view(-1, 1)
    y_true = y_true.float().view(1, -1)
    numer = 2 * y_true @ y_pred
    denom = y_true.sum() + y_pred.sum()
    dsc_score = (numer + smooth) / (denom + smooth)
    return dsc_score


def do_eval(y_true, y_pred):
    dsc = dice_coef(y_true, y_pred)
    hd = getHausdorff(np.array(y_true.cpu()), np.array(y_pred.cpu()))
    PPV, sensitivity, specificity = get_confusionMatrix_metrics(y_true, y_pred)
    result = {'dsc': dsc, 'hd': hd, 'ppv': PPV, 'sens': sensitivity, 'spec': specificity}

    return result


def getHausdorff(y_true, y_pred):
    assert len(np.unique(y_pred)) > 2, " y_true is not binary!!!   {}".format(np.unique(y_pred))
    return metric.hd(y_pred, y_true)


# def dice_coef(y_true, y_pred, smooth=1):
#   intersection = torch.mul(y_true, y_pred)
#   n_intersection = torch.sum(intersection,(1,2))
#   n_y_true = torch.sum(y_true,(1,2))
#   n_y_pred = torch.sum(y_pred,(1,2))
#   union = n_y_true + n_y_pred
#   dice = ( 2. *  n_intersection + smooth) / (union + smooth)
#   return dice

def getPPV(y_true, y_pred):
    y_pred = y_pred.float().view(-1, 1)
    y_true = y_true.float().view(1, -1)

    TP = y_true @ y_pred
    true_negatives = (y_true == 0).float()
    FP = true_negatives @ y_pred
    if TP == 0:
        return 0
    return TP / (TP + FP)


def get_confusionMatrix_metrics(y_true, y_pred):
    y_pred = y_pred.float().view(-1, 1)
    y_true = y_true.float().view(1, -1)

    TP = y_true @ y_pred
    ytrue_negatives = (y_true == 0).float()
    pred_negatives = (y_pred == 0).float()

    FP = ytrue_negatives @ y_pred
    FN = y_true @ pred_negatives
    TN = ytrue_negatives @ pred_negatives
    PPV = torch.tensor(0) if TP == 0 else TP / (TP + FP)
    sensitivity = torch.tensor(0) if TP == 0 else TP / (TP + FN)
    specificity = torch.tensor(0) if TN == 0 else TN / (TN + FP)
    return PPV, sensitivity, specificity


def get_dice_coef_per_subject(y_true, y_pred, subjects):
    dsc = []
    for subj in subjects.unique():
        y_pred_subj = y_pred[subjects == subj]
        y_true_subj = y_true[subjects == subj]
        print("for subject {} is {}".format(subj, dice_coef(y_true_subj, y_pred_subj)))
        dsc.append(dice_coef(y_true_subj, y_pred_subj))

    return torch.mean(torch.tensor(dsc))
