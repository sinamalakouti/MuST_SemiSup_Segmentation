import numpy as np
import torch

# dice coefficient for predicted class
def dice_coef(y_true, y_pred, smooth =1 ):
  y_pred = y_pred.float().float().view(-1, 1)
  y_true = y_true.float().view(1, -1)
  numer = 2 * y_true @ y_pred
  denom = y_true.sum() + y_pred.sum()
  dsc_score = (numer + smooth / denom + smooth)
  return dsc_score

# def dice_coef(y_true, y_pred, smooth=1):
#   intersection = torch.mul(y_true, y_pred)
#   n_intersection = torch.sum(intersection,(1,2))
#   n_y_true = torch.sum(y_true,(1,2))
#   n_y_pred = torch.sum(y_pred,(1,2))
#   union = n_y_true + n_y_pred
#   dice = ( 2. *  n_intersection + smooth) / (union + smooth)
#   return dice