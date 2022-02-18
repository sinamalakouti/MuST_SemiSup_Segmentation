import os
import sys
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import utils
from losses.loss import consistency_weight, Consistency_CE, softmax_kl_loss

from losses.evaluation_metrics import dice_coef, do_eval
from dataset.Brat20 import Brat20Test, seg2WT, seg2TC, seg2ET
from models import Perturbations
from models import Pgs
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import random
import argparse
import yaml
from easydict import EasyDict as edict
import datetime
import wandb

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')

for p in sys.path:
    print("path  ", p)


def __fw_unsup_loss(y_stud, y_teach, loss_functions, cfg, mask=None):
    if cfg.unsupervised_training.loss_method == 'output-wise':
        return __fw_outputwise_unsup_loss(y_stud, y_teach, loss_functions, cfg, mask)
    else:
        return __fw_downsample_unsup_loss(y_stud, y_teach, loss_functions, cfg, mask)


def __fw_downsample_unsup_loss(y_stud, y_teach, loss_functions, cfg, mask=None):
    # downsample main decoder's output
    (_, unsup_loss) = loss_functions
    # todo handle if mask is not None
    losses = []
    p = torch.nn.functional.softmax(y_teach, dim=1)
    if cfg.unsupervised_training.T is not None:  # sharpening
        pt = p ** (1 / cfg.unsupervised_training.T)
        y_true = pt / pt.sum(dim=1, keepdim=True)

    n_outputs = len(y_stud)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    for i in range(n_outputs - 1, -1, -1):
        output = y_stud[i]
        if i != n_outputs - 1:
            y_true = maxpool(y_true)
            if y_true.shape[-1] != output.shape[-1] or y_true.shape[-2] != output.shape[-2]:
                h_diff = output.size()[-2] - y_true.size()[-2]
                w_diff = output.size()[-1] - y_true.size()[-1]
                #
                y_true = F.pad(y_true, (w_diff // 2, w_diff - w_diff // 2,
                                        h_diff // 2, h_diff - h_diff // 2))
            assert output.shape[-2:] == y_true.shape[-2:], "output and target shape is not similar!!"

        if cfg.unsupervised_training.consistency_loss == 'CE':
            losses.append(- torch.mean(
                torch.sum(y_true.detach()
                          * torch.nn.functional.log_softmax(stud_pred, dim=1), dim=1)))
        elif cfg.unsupervised_training.consistency_loss == 'KL':
            losses.append(
                softmax_kl_loss(stud_pred, y_true.detach(), conf_mask=False, threshold=None, use_softmax=True))
        elif cfg.unsupervised_training.consistency_loss == 'balanced_CE':
            losses.append(
                unsup_loss(stud_pred, y_true.detach(), i, use_softmax=True))
        elif cfg.unsupervised_training.consistency_loss == 'MSE':
            #     teach_pred = torch.nn.functional.softmax(teach_pred / 0.85, dim=1)
            stud_pred = torch.nn.functional.softmax(stud_pred, dim=1)
            mse = torch.nn.MSELoss()
            loss = mse(stud_pred, y_true.detach())
            losses.append(loss)
    total_loss = sum(losses)
    return total_loss


def __fw_outputwise_unsup_loss(y_stud, y_teach, loss_functions, cfg, masks=None):
    (_, unsup_loss) = loss_functions

    assert len(y_teach) == len(y_stud), "Error! unsup_preds and sup_preds have to have same length"
    num_preds = len(y_teach)
    losses = []

    for i in range(num_preds):
        teach_pred = y_teach[i]
        stud_pred = y_stud[i]
        assert teach_pred.shape == stud_pred.shape, "Error! for preds number {}, supervised and unsupervised" \
                                                    " prediction shape is not similar!".format(i)
        if masks is not None:
            teach_pred = teach_pred * masks[i]
            stud_pred = stud_pred * masks[i]

        if cfg.unsupervised_training.consistency_loss == 'CE':
            teach_pred = torch.nn.functional.softmax(teach_pred, dim=1)
            losses.append(- torch.mean(
                torch.sum(teach_pred.detach()
                          * torch.nn.functional.log_softmax(stud_pred, dim=1), dim=1)))
        elif cfg.unsupervised_training.consistency_loss == 'KL':
            losses.append(
                softmax_kl_loss(stud_pred, teach_pred.detach(), conf_mask=False, threshold=None, use_softmax=True))
        elif cfg.unsupervised_training.consistency_loss == 'balanced_CE':
            losses.append(
                unsup_loss(stud_pred, teach_pred, i, use_softmax=True))
        elif cfg.unsupervised_training.consistency_loss == 'MSE':
            if cfg.experiment_mode != 'semi_alternate_mix_F_G':
                teach_pred = torch.nn.functional.softmax(teach_pred, dim=1)

            if cfg.unsupervised_training.T is not None:# sharpening
                pt = teach_pred ** (1 / cfg.unsupervised_training.T)
                teach_pred = pt / pt.sum(dim=1, keepdim=True)
                if teach_pred.isnan().sum() > 0:
                    teach_pred[teach_pred.isnan()] = 0

                # teach_pred = torch.nn.functional.softmax(teach_pred / 0.85, dim=1)
            stud_pred = torch.nn.functional.softmax(stud_pred, dim=1)
            mse = torch.nn.MSELoss()
            loss = mse(stud_pred, teach_pred.detach())
            losses.append(loss)
    total_loss = sum(losses)
    return total_loss


def __fw_sup_loss(y_preds, y_true, sup_loss):
    # iterate over all level's output
    losses = []
    target_fg = y_true
    # 1 - target_fg
    n_outputs = len(y_preds)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    for i in range(n_outputs - 1, -1, -1):
        # ratio = int(np.round(y_true.shape[2] / output.shape[2]))
        output = y_preds[i]
        if i != n_outputs - 1:
            target_fg = maxpool(target_fg)
            if target_fg.shape[-1] != output.shape[-1] or target_fg.shape[-2] != output.shape[-2]:
                h_diff = output.size()[-2] - target_fg.size()[-2]
                w_diff = output.size()[-1] - target_fg.size()[-1]
                #
                target_fg = F.pad(target_fg, (w_diff // 2, w_diff - w_diff // 2,
                                              h_diff // 2, h_diff - h_diff // 2))

            assert output.shape[-2:] == target_fg.shape[-2:], "output and target shape is not similar!!"
        if output.shape[1] != target_fg.shape[1] and type(sup_loss) == torch.nn.CrossEntropyLoss and len(
                target_fg.shape) > 3:
            target_fg = target_fg.reshape((target_fg.shape[0], target_fg.shape[2], target_fg.shape[3]))

        losses.append(sup_loss(output, target_fg.type(torch.LongTensor).to(output.device)))
    total_loss = sum(losses)
    return total_loss


def compute_loss(y_preds, y_true, loss_functions, is_supervised, cfg, masks=None):
    if is_supervised:
        total_loss = __fw_sup_loss(y_preds, y_true, loss_functions[0])

        ''' y_preds is students preds and y_true is teacher_preds!
                    for comparing outputs together!  # consistency of original output and noisy output 
        '''
    else:
        total_loss = __fw_unsup_loss(y_preds, y_true, loss_functions, cfg, masks)
        # total_loss = __fw_outputwise_unsup_loss(y_preds, y_true, loss_functions, cfg, masks)

    return total_loss


def trainPgs_semi(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions, cons_w_unsup, epochid,
                  cfg):
    model.train()

    train_sup_iterator = iter(train_sup_loader)
    sup_step = 0
    for unsup_step, batch_unsup in enumerate(train_unsup_loader):
        optimizer.zero_grad()
        b_unsup = batch_unsup['data']
        b_unsup = b_unsup.to(device)

        try:
            batch_sup = next(train_sup_iterator)

        except StopIteration:
            train_sup_iterator = iter(train_sup_loader)
            batch_sup = next(train_sup_iterator)

        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)

        sup_outputs, _ = model(b_sup, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)

        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)

        print("**************** UNSUP LOSSS  : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))
        weight_unsup = cons_w_unsup(epochid, unsup_step)
        total_loss = sLoss + weight_unsup * uLoss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            sf = torch.nn.Softmax2d()
            target_sup[target_sup >= 1] = 1
            target_sup = target_sup
            y_pred = sf(sup_outputs[-1])
            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": sup_step + epochid * len(train_unsup_loader),
                 "sup loss": sLoss,
                 "unsup_batch_id": unsup_step + epochid * len(train_unsup_loader),
                 "unsup loss": uLoss,
                 "batch_score_WT": dice_score,
                 'weight unsup': weight_unsup
                 })
        sup_step += 1
    return model


def trainPgs_semi_downSample(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions,
                             cons_w_unsup, epochid,
                             cfg):
    model.train()
    # dataloader
    semi_dataLoader = zip(train_sup_loader, train_unsup_loader)

    for batch_idx, (batch_sup, batch_unsup) in enumerate(semi_dataLoader):
        optimizer.zero_grad()
        # read labeled data
        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)
        # forward-pass: supervised model
        sup_outputs, _ = model(b_sup, is_supervised=True)
        # compute supervised loss
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)
        # remove supervised batch
        del b_sup
        # read unlabeled data
        b_unsup = batch_unsup['data']
        b_unsup = b_unsup.to(device)
        # forward pass: consistency (unsupevised) model
        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        # compute unsupervised loss
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)

        print("**************** UNSUP LOSSS  : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))
        # compute consistency weigth ( for unsupervised loss)
        weight_unsup = cons_w_unsup(epochid, batch_idx)
        # compute total loss
        total_loss = sLoss + weight_unsup * uLoss
        # backward pass & optimizer step
        total_loss.backward()
        optimizer.step()
        # compute batch scores and log to Wandb
        with torch.no_grad():
            sf = torch.nn.Softmax2d()
            target_sup[target_sup >= 1] = 1
            target_sup = target_sup
            y_pred = sf(sup_outputs[-1])
            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": batch_idx + epochid * min(len(train_sup_loader), len(train_sup_loader)),
                 "sup loss": sLoss,
                 "unsup_batch_id": batch_idx + epochid * min(len(train_sup_loader), len(train_sup_loader)),
                 "unsup loss": uLoss,
                 "batch_score_WT": dice_score,
                 'weight unsup': weight_unsup
                 })

    return model


def trainPgs_semi_alternate(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions,
                            cons_w_unsup, epochid,
                            cfg):
    model.train()
    # semi_loader -> size of smaller set (supervised loader)
    semi_loader = zip(train_sup_loader, train_unsup_loader)

    for batch_idx, (batch_sup, batch_unsup) in enumerate(semi_loader):
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        b_unsup = batch_unsup['data']
        b_unsup = b_unsup.to(device)

        # unsupervised training - forward
        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)
        # unsupervised training - backward
        weight_unsup = cons_w_unsup(epochid, batch_idx)
        uloss_final = uLoss * weight_unsup
        uloss_final.backward()
        optimizer[1].step()

        # todo:  delete unsupervised data from GPU
        # supervised training
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)

        # supervised training - forward
        sup_outputs, _ = model(b_sup, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)

        # supervised training - backward
        sLoss.backward()
        optimizer[0].step()

        print("**************** UNSUP LOSSS ( weighted) : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))

        with torch.no_grad():
            sf = torch.nn.Softmax2d()
            target_sup[target_sup >= 1] = 1
            target_sup = target_sup
            y_pred = sf(sup_outputs[-1])
            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)  # I can identify the threshold! -> more confidence!
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "sup loss": sLoss,
                 "unsup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "unsup loss": uLoss,
                 "batch_score_WT": dice_score,
                 'weight unsup': weight_unsup
                 })

    return model


def trainPgs_semi_alternate_I_and_F_aug(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions,
                                        cons_w_unsup, epochid,
                                        cfg):
    model.train()
    # semi_loader -> size of smaller set (supervised loader)
    semi_loader = zip(train_sup_loader, train_unsup_loader)

    for batch_idx, (batch_sup, batch_unsup) in enumerate(semi_loader):
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        b_usnup_orig = batch_unsup['data']  # data original
        # b_unsup = b_unsup.to(device)

        # unsupervised training - forward
        with torch.no_grad():
            teacher_outputs, _ = model(b_usnup_orig.to(device), is_supervised=True)
        b_unsup_aug, teacher_outputs, masks = Perturbations.fw_input_geo_aug(b_usnup_orig, teacher_outputs)
        _, student_outputs = model(b_unsup_aug.to(device), is_supervised=False)

        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, masks=masks, is_supervised=False,
                             cfg=cfg)
        # unsupervised training - backward
        weight_unsup = cons_w_unsup(epochid, batch_idx)
        uloss_finall = uLoss * weight_unsup
        uloss_finall.backward()
        optimizer[1].step()

        # todo:  delete unsupervised data from GPU
        # supervised training
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)

        # supervised training - forward
        sup_outputs, _ = model(b_sup, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)

        # supervised training - backward
        sLoss.backward()
        optimizer[0].step()

        print("**************** UNSUP LOSSS (weighted) : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))

        with torch.no_grad():
            sf = torch.nn.Softmax2d()
            target_sup[target_sup >= 1] = 1
            target_sup = target_sup
            y_pred = sf(sup_outputs[-1])
            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)  # I can identify the threshold! -> more confidence!
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "sup loss": sLoss,
                 "unsup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "unsup loss": uLoss,
                 "batch_score_WT": dice_score,
                 'weight unsup': weight_unsup
                 })

    return model


def trainPgs_sup_upSample(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions, cons_w_unsup,
                          epochid, cfg):
    model.train()

    train_sup_iterator = iter(train_sup_loader)
    sup_step = 0
    for unsup_step, batch_unsup in enumerate(train_unsup_loader):
        optimizer.zero_grad()

        try:
            batch_sup = next(train_sup_iterator)

        except StopIteration:
            train_sup_iterator = iter(train_sup_loader)
            batch_sup = next(train_sup_iterator)

        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)

        sup_outputs, _ = model(b_sup, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)

        print("**************** SUP LOSSS  : {} ****************".format(sLoss))

        total_loss = sLoss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            sf = torch.nn.Softmax2d()
            target_sup[target_sup >= 1] = 1
            target_sup = target_sup
            y_pred = sf(sup_outputs[-1])
            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {
                    "sup_batch_id": sup_step + epochid * len(train_unsup_loader),
                    "sup loss": sLoss,
                    "unsup_batch_id": unsup_step + epochid * len(train_unsup_loader),
                    "batch_score_WT": dice_score,
                })
        sup_step += 1
    return model


def trainPgs_sup(train_sup_loader, model, optimizer, device, loss_functions, epochid, cfg):
    model.train()
    sup_loss = loss_functions[0]

    for step, batch_sup in enumerate(train_sup_loader):
        optimizer.zero_grad()
        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)

        print("subject is : ", batch_sup['subject'])
        sup_outputs, _ = model(b_sup, is_supervised=True)
        total_loss = compute_loss(sup_outputs, target_sup, (sup_loss, None), is_supervised=True, cfg=cfg)

        # wandb.log({"batch_id": step + epochid * len(train_sup_loader), "loss": total_loss})
        print("**************** LOSSS  : {} ****************".format(total_loss))

        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            if cfg.oneHot:
                target_sup[target_sup >= 1] = 1
                target_sup = seg2WT(target_sup, 1, oneHot=cfg.oneHot)
                y_pred = sup_outputs
            else:
                sf = torch.nn.Softmax2d()
                target_sup[target_sup >= 1] = 1
                target_sup = target_sup
                y_pred = sf(sup_outputs[-1])

            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": step + epochid * len(train_sup_loader), "sup_loss": total_loss,
                 "batch_score": dice_score})

    return model


@torch.no_grad()
def eval_per_subjectPgs(model, device, threshold, cfg, data_mode):
    print("******************** EVALUATING {}********************".format(data_mode))

    testset = Brat20Test(f'data/brats20', data_mode, 10, 155,
                         augment=False, center_cropping=True, t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce, oneHot=cfg.oneHot)

    model.eval()
    dice_arrWT = []
    dice_arrTC = []
    dice_arrET = []

    PPV_arrWT = []
    PPV_arrTC = []
    PPV_arrET = []

    sensitivity_arrWT = []
    sensitivity_arrTC = []
    sensitivity_arrET = []

    specificity_arrWT = []
    specificity_arrTC = []
    specificity_arrET = []

    hd_arrWT = []
    hd_arrTC = []
    hd_arrET = []

    paths = testset.paths
    sup_loss = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for path in paths:
            batch = testset.get_subject(path)
            b = batch['data']
            target = batch['label'].to(device)
            subjects = batch['subjects']
            assert len(np.unique(subjects)) == 1, print("More than one subject at a time")
            b = b.to(device)
            outputs, _ = model(b, True)

            if cfg.oneHot:
                sf = torch.nn.Softmax2d()
                outputs = sf(outputs[-1])

            loss_val = compute_loss(outputs, target, (sup_loss, None), is_supervised=True, cfg=cfg)
            print("############# LOSS for subject {} is {} ##############".format(subjects[0], loss_val.item()))
            if cfg.oneHot:
                target[target >= 1] = 1
                # target_WT = seg2WT(target, 1, oneHot=cfg.oneHot)
                y_pred = outputs
            else:
                sf = torch.nn.Softmax2d()
                targetWT = target.clone()
                targetET = target.clone()
                targetTC = target.clone()
                targetWT[targetWT >= 1] = 1
                targetET[~ (targetET == 3)] = 0
                targetET[(targetET == 3)] = 1
                targetTC[~ ((targetTC == 3) | (targetTC == 1))] = 0
                targetTC[(targetTC == 3) | (targetTC == 1)] = 1
                y_pred = sf(outputs[-1])

            y_WT = seg2WT(y_pred, threshold, oneHot=cfg.oneHot)
            y_ET = seg2ET(y_pred, threshold)
            y_TC = seg2TC(y_pred, threshold)

            metrics_WT = do_eval(targetWT.reshape(y_WT.shape), y_WT)
            metrics_ET = do_eval(targetET.reshape(y_ET.shape), y_ET)
            metrics_TC = do_eval(targetTC.reshape(y_TC.shape), y_TC)

            print("*** EVALUATION METRICS FOR SUBJECT {} IS: ".format(subjects[0]))
            print(
                "(WT) :  DICE SCORE   {}, PPV  {},  Sensitivity: {}, Specificity: {}, Hausdorff: {}".format(
                    metrics_WT['dsc'], metrics_WT['ppv'],
                    metrics_WT['sens'],
                    metrics_WT['spec'], metrics_WT['hd']))
            print(
                "(ET) :  DICE SCORE   {}, PPV  {},  Sensitivity: {}, Specificity: {},  Hausdorff: {}".format(
                    metrics_ET['dsc'], metrics_ET['ppv'],
                    metrics_ET['sens'],
                    metrics_ET['spec'], metrics_ET['hd']))
            print(
                "(TC) :  DICE SCORE   {}, PPV  {},  Sensitivity: {}, Specificity: {}, Hausdorff: {}".format(
                    metrics_TC['dsc'], metrics_TC['ppv'],
                    metrics_TC['sens'],
                    metrics_TC['spec'], metrics_TC['hd']))

            dice_arrWT.append(metrics_WT['dsc'].item())
            dice_arrET.append(metrics_ET['dsc'].item())
            dice_arrTC.append(metrics_TC['dsc'].item())
            # ET <-> TC
            PPV_arrWT.append(metrics_WT['ppv'].item())
            PPV_arrET.append(metrics_ET['ppv'].item())
            PPV_arrTC.append(metrics_TC['ppv'].item())
            # ET <-> TC
            sensitivity_arrWT.append(metrics_WT['sens'].item())
            sensitivity_arrET.append(metrics_ET['sens'].item())
            sensitivity_arrTC.append(metrics_TC['sens'].item())
            # ET <-> TC
            specificity_arrWT.append(metrics_WT['spec'].item())
            specificity_arrET.append(metrics_ET['spec'].item())
            specificity_arrTC.append(metrics_TC['spec'].item())

            hd_arrWT.append(metrics_WT['hd'])
            hd_arrET.append(metrics_ET['hd'])
            hd_arrTC.append(metrics_TC['hd'])

    final_dice = {'WT': np.mean(dice_arrWT), 'ET': np.mean(dice_arrET), 'TC': np.mean(dice_arrTC)}
    final_PPV = {'WT': np.mean(PPV_arrWT), 'ET': np.mean(PPV_arrET), 'TC': np.mean(PPV_arrTC)}
    final_sensitivity = {'WT': np.mean(sensitivity_arrWT), 'ET': np.mean(sensitivity_arrET),
                         'TC': np.mean(sensitivity_arrTC)}
    final_specificity = {'WT': np.mean(specificity_arrWT), 'ET': np.mean(specificity_arrET),
                         'TC': np.mean(specificity_arrTC)}
    final_hd = {'WT': np.mean(hd_arrWT), 'ET': np.mean(hd_arrET),
                'TC': np.mean(hd_arrTC)}
    return final_dice, final_PPV, final_sensitivity, final_specificity, final_hd
    # return np.mean(np.array(dice_arrWT)), np.mean(np.array(dice_arrET)), np.mean(np.array(dice_arrTC))


def initialize_directories(output_dir, seed, cfg):
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    output_dir = os.path.join(output_dir, cfg.experiment_mode)
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    if cfg.experiment_mode != 'fully_sup':
        output_dir = os.path.join(output_dir, "sup_ratio_{}".format(cfg.train_sup_rate))
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % output_dir)

    elif cfg.experiment_mode == 'fully_sup':
        output_dir = os.path.join(output_dir, "fullySup_ratio_{}".format(cfg.train_sup_rate))
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % output_dir)

    output_dir = os.path.join(output_dir, "seed_{}".format(seed))
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    now = str(datetime.datetime.now())
    output_dir = os.path.join(output_dir, "{}".format(now))
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)

    output_model_dir = os.path.join(output_dir, "best_model")

    print("output_model_dir is   ", output_model_dir)

    if not os.path.isdir(output_model_dir):
        try:
            os.mkdir(output_model_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_model_dir)

    if not os.path.isdir(os.path.join(output_dir, "runs")):
        try:
            os.mkdir(os.path.join(output_dir, "runs"), 0o777)
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(output_dir, "runs"))
    # save config file!
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
        docs = yaml.dump(cfg, file)

    output_image_dir = os.path.join(output_dir, "result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)
    return output_model_dir, output_image_dir


def get_train_dataloaders(dataset, cfg):
    train_unsup_loader = None
    train_sup_loader = utils.get_trainset(dataset, batch_size=cfg.batch_size, intensity_rescale=cfg.intensity_rescale,
                                          mixup_threshold=cfg.mixup_threshold, mode=cfg.train_sup_mode, t1=cfg.t1,
                                          t2=cfg.t2, t1ce=cfg.t1ce, augment=cfg.augment, seed=cfg.seed)

    print('size of labeled training set: number of subjects:    ', len(train_sup_loader.dataset.subjects_name))
    print("labeled subjects  ", train_sup_loader.dataset.subjects_name)

    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'partially_sup_upSample' or \
            cfg.experiment_mode == 'semi_downSample' or cfg.experiment_mode == 'semi_alternate':
        train_unsup_loader = utils.get_trainset(dataset, batch_size=cfg.batch_size,
                                                intensity_rescale=cfg.intensity_rescale,
                                                mixup_threshold=cfg.mixup_threshold,
                                                mode=cfg.train_unsup_mode, t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce,
                                                augment=cfg.augment, seed=cfg.seed)
    elif cfg.experiment_mode == 'semi_alternate_mix_F_G':
        train_unsup_loader = utils.get_trainset(dataset, batch_size=cfg.batch_size,
                                                intensity_rescale=cfg.intensity_rescale,
                                                mixup_threshold=cfg.mixup_threshold,
                                                mode=cfg.train_unsup_mode, t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce,
                                                augment=False, seed=cfg.seed)

        print('size of unlabeled training set: number of subjects:    ', len(train_unsup_loader.dataset.subjects_name))
        print("un labeled subjects  ", train_unsup_loader.dataset.subjects_name)
    return train_sup_loader, train_unsup_loader


def Pgs_train_val(dataset, n_epochs, wmh_threshold, output_dir, args, cfg, seed):
    inputs_dim = [4, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 4]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    wandb.run.name = "{}_PGS_{}_{}_supRate{}_seed{}_".format(cfg.experiment_mode, "trainALL2018", 'valNew2019',
                                                             cfg.train_sup_rate, seed)

    '''  uncomment this when you want to create a new split
    dataroot_dir = f'data/brats20'
    all_train_csv = os.path.join(dataroot_dir, 'trainset/brats2018.csv')
    supdir_path = os.path.join(dataroot_dir, 'trainset')

     semi_sup_split(all_train_csv=all_train_csv, sup_dir_path=supdir_path, unsup_dir_path=supdir_path,
                    ratio=cfg.train_sup_rate / 100, seed=cfg.seed)
    '''

    best_score = 0
    start_epoch = 0
    print('-' * 50)
    print("EXPERIMENT-   PGS on BRATS")
    print('-' * 50)

    output_model_dir, output_image_dir = initialize_directories(output_dir, seed, cfg)

    print("EXPERIMENT DESCRIPTION:   {}".format(cfg.information))

    print("******* TRAINING PGS ***********")
    print("sup learning_rate is    ", cfg.supervised_training.lr)
    print("unsup learning_rate is    ", cfg.unsupervised_training.lr)

    # load model
    pgsnet = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)

    if torch.cuda.is_available():
        if type(pgsnet) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            pgsnet = torch.nn.DataParallel(pgsnet)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    pgsnet.to(device)

    # loading the datasets
    train_sup_loader, train_unsup_loader = get_train_dataloaders(dataset, cfg)

    # load optimizers & schedulers
    if 'alternate' in cfg.experiment_mode:
        optimizer_unsup = torch.optim.SGD(pgsnet.parameters(), cfg.unsupervised_training.lr, momentum=0.9,
                                          weight_decay=1e-4)
        scheduler_unsup = lr_scheduler.StepLR(optimizer_unsup,
                                              step_size=cfg.unsupervised_training.scheduler_step_size,
                                              gamma=cfg.unsupervised_training.lr_gamma)
    else:
        optimizer_unsup = None
        scheduler_unsup = None
    optimizer_sup = torch.optim.SGD(pgsnet.parameters(), cfg.supervised_training.lr, momentum=0.9,
                                    weight_decay=1e-4)

    scheduler_sup = lr_scheduler.StepLR(optimizer_sup, step_size=cfg.supervised_training.scheduler_step_size,
                                        gamma=cfg.supervised_training.lr_gamma)

    # consistency regularization weight

    cons_w_unsup = consistency_weight(final_w=cfg['unsupervised_training']['consist_w_unsup']['final_w'],
                                      iters_per_epoch=len(train_sup_loader),
                                      rampup_ends=cfg['unsupervised_training']['consist_w_unsup'][
                                          'rampup_ends'],
                                      ramp_type=cfg['unsupervised_training']['consist_w_unsup']['rampup'])

    if cfg.unsupervised_training.consistency_loss == 'balanced_CE':
        cons_loss_fn = Consistency_CE(5)
    else:
        cons_loss_fn = None

    for epoch in range(start_epoch, n_epochs):
        print("iteration:  ", epoch)

        if cfg.experiment_mode == 'semi':

            trainPgs_semi(train_sup_loader, train_unsup_loader, pgsnet, optimizer_sup, device,
                          (torch.nn.CrossEntropyLoss(), cons_loss_fn), cons_w_unsup,
                          epoch, cfg)
        elif cfg.experiment_mode == 'semi_downSample':

            trainPgs_semi_downSample(train_sup_loader, train_unsup_loader, pgsnet, optimizer_sup, device,
                                     (torch.nn.CrossEntropyLoss(), cons_loss_fn),
                                     cons_w_unsup,
                                     epoch, cfg)
        elif cfg.experiment_mode == 'semi_alternate':

            trainPgs_semi_alternate(train_sup_loader, train_unsup_loader, pgsnet,
                                    (optimizer_sup, optimizer_unsup), device,
                                    (torch.nn.CrossEntropyLoss(), cons_loss_fn),
                                    cons_w_unsup,
                                    epoch, cfg)
        elif cfg.experiment_mode == 'semi_alternate_mix_F_G':
            trainPgs_semi_alternate_I_and_F_aug(train_sup_loader, train_unsup_loader, pgsnet,
                                                (optimizer_sup, optimizer_unsup), device,
                                                (torch.nn.CrossEntropyLoss(), cons_loss_fn),
                                                cons_w_unsup,
                                                epoch, cfg)

        elif cfg.experiment_mode == 'partially_sup':

            trainPgs_sup(train_sup_loader, pgsnet, optimizer_sup, device,
                         (torch.nn.CrossEntropyLoss(), None),
                         epoch, cfg)
        elif cfg.experiment_mode == 'fully_sup':
            trainPgs_sup(train_sup_loader, pgsnet, optimizer_sup, device,
                         (torch.nn.CrossEntropyLoss(), None),
                         epoch, cfg)
        elif cfg.experiment_mode == 'partially_sup_upSample':
            trainPgs_sup_upSample(train_sup_loader, train_unsup_loader, pgsnet, optimizer_sup, device,
                                  (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()), None,
                                  epoch, cfg)

        if epoch % 2 == 0:
            val_final_dice, val_final_PPV, val_final_sensitivity, val_final_specificity, val_final_hd = \
                eval_per_subjectPgs(pgsnet, device, wmh_threshold, cfg, cfg.val_mode)

            if val_final_dice['WT'] > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch,
                                                                                                     val_final_dice[
                                                                                                         'WT']))
                best_score = val_final_dice['WT']
                if args.save_model:
                    path = os.path.join(output_model_dir, 'pgsnet_best.model')
                    with open(path, 'wb') as f:
                        torch.save(pgsnet, f)
            test_final_dice, test_final_PPV, test_final_sensitivity, test_final_specificity, test_final_hd = \
                eval_per_subjectPgs(pgsnet, device, wmh_threshold, cfg, cfg.test_mode)
            save_score_all(output_image_dir,
                           (test_final_dice, test_final_hd, test_final_PPV, test_final_sensitivity,
                            test_final_specificity),
                           epoch, mode=cfg.test_mode)
            wandb.log({'epoch_id': epoch,
                       'test_WT_subject_wis_DSC': test_final_dice['WT'],
                       'test_WT_subject_wise_HD': test_final_hd['WT'],
                       'test_WT_subject_wise_PPV': test_final_PPV['WT'],
                       'test_WT_subject_wise_SENSITIVITY': test_final_sensitivity['WT'],
                       'test_WT_subject_wise_val_SPECIFCITY': test_final_specificity['WT'],
                       'test_ET_subject_wise_DSC': test_final_dice['ET'],
                       'test_ET_subject_wise_HD': test_final_hd['ET'],
                       'test_ET_subject_wise_PPV': test_final_PPV['ET'],
                       'test_ET_subject_wise_SENSITIVITY': test_final_sensitivity['ET'],
                       'test_ET_subject_wise_val_SPECIFCITY': test_final_specificity['ET'],
                       'test_TC_subject_wise_DSC': test_final_dice['TC'],
                       'test_TC_subject_wise_HD': test_final_hd['TC'],
                       'test_TC_subject_wise_PPV': test_final_PPV['TC'],
                       'test_TC_subject_wise_SENSITIVITY': test_final_sensitivity['TC'],
                       'test_TC_subject_wise_val_SPECIFCITY': test_final_specificity['TC'],
                       })

            save_score_all(output_image_dir,
                           (val_final_dice, val_final_hd, val_final_PPV, val_final_sensitivity, val_final_specificity),
                           epoch, mode=cfg.val_mode)

            wandb.log({'epoch_id': epoch,
                       'val_WT_subject_wise_DSC': val_final_dice['WT'],
                       'val_WT_subject_wise_HD': val_final_hd['WT'],
                       'val_WT_subject_wise_PPV': val_final_PPV['WT'],
                       'val_WT_subject_wise_SENSITIVITY': val_final_sensitivity['WT'],
                       'val_WT_subject_wise_SPECIFCITY': val_final_specificity['WT'],
                       'val_ET_subject_wise_DSC': val_final_dice['ET'],
                       'val_ET_subject_wise_HD': val_final_hd['ET'],
                       'val_ET_subject_wise_PPV': val_final_PPV['ET'],
                       'val_ET_subject_wise_SENSITIVITY': val_final_sensitivity['ET'],
                       'val_ET_subject_wise_SPECIFCITY': val_final_specificity['ET'],
                       'val_TC_subject_wise_DSC': val_final_dice['TC'],
                       'val_TC_subject_wise_HD': val_final_hd['TC'],
                       'val_TC_subject_wise_PPV': val_final_PPV['TC'],
                       'val_TC_subject_wise_SENSITIVITY': val_final_sensitivity['TC'],
                       'val_TC_subject_wise_SPECIFCITY': val_final_specificity['TC'],
                       })

        if scheduler_unsup is not None:
            scheduler_sup.step()
            scheduler_unsup.step()
        else:
            scheduler_sup.step()

    final_dice, final_PPV, final_sensitivity, final_specificity, final_hd = eval_per_subjectPgs(pgsnet, device,
                                                                                                wmh_threshold,
                                                                                                cfg,
                                                                                                cfg.val_mode)
    print("** (WT) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, PPV:{}, Sensitivity: {}, Specificity: {}  **"
          .format(cfg.n_epochs, final_dice['WT'], final_hd['WT'], final_PPV['WT'], final_sensitivity['WT'],
                  final_specificity['WT']))
    print("** (ET) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, :{}, Sensitivity:{}, Specificity: {}  **"
          .format(cfg.n_epochs, final_dice['ET'], final_hd['ET'], final_PPV['ET'], final_sensitivity['ET'],
                  final_specificity['ET']))
    print("** (TC) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, PPV:{}, Sensitivity:{}, Specificity: {}  **".
          format(cfg.n_epochs, final_dice['TC'], final_hd['TC'], final_PPV['TC'], final_sensitivity['TC'],
                 final_specificity['TC']))

    save_score_all(output_image_dir, (final_dice, final_hd, final_PPV, final_sensitivity, final_specificity),
                   cfg.n_epochs, mode=cfg.val_mode)
    wandb.log({'epoch_id': cfg.n_epochs,
               'val_WT_subject_wise_DSC': final_dice['WT'],
               'val_WT_subject_wise_HD': final_hd['WT'],
               'val_WT_subject_wise_PPV': final_PPV['WT'],
               'val_WT_subject_wise_SENSITIVITY': final_sensitivity['WT'],
               'val_WT_subject_wise_SPECIFCITY': final_specificity['WT'],
               'val_ET_subject_wise_DSC': final_dice['ET'],
               'val_ET_subject_wise_HD': final_hd['ET'],
               'val_ET_subject_wise_PPV': final_PPV['ET'],
               'val_ET_subject_wise_SENSITIVITY': final_sensitivity['ET'],
               'val_ET_subject_wise_SPECIFCITY': final_specificity['ET'],
               'val_TC_subject_wise_DSC': final_dice['TC'],
               'val_TC_subject_wise_HD': final_hd['TC'],
               'val_TC_subject_wise_PPV': final_PPV['TC'],
               'val_TC_subject_wise_SENSITIVITY': final_sensitivity['TC'],
               'val_TC_subject_wise_SPECIFCITY': final_specificity['TC'],
               })

    test_final_dice, test_final_PPV, test_final_sensitivity, test_final_specificity, test_final_hd = \
        eval_per_subjectPgs(pgsnet, device, wmh_threshold, cfg, cfg.test_mode)
    save_score_all(output_image_dir,
                   (test_final_dice, test_final_hd, test_final_PPV, test_final_sensitivity,
                    test_final_specificity),
                   epoch, mode=cfg.test_mode)
    wandb.log({'epoch_id': epoch,
               'test_WT_subject_wis_DSC': test_final_dice['WT'],
               'test_WT_subject_wise_HD': test_final_hd['WT'],
               'test_WT_subject_wise_PPV': test_final_PPV['WT'],
               'test_WT_subject_wise_SENSITIVITY': test_final_sensitivity['WT'],
               'test_WT_subject_wise_val_SPECIFCITY': test_final_specificity['WT'],
               'test_ET_subject_wise_DSC': test_final_dice['ET'],
               'test_ET_subject_wise_HD': test_final_hd['ET'],
               'test_ET_subject_wise_PPV': test_final_PPV['ET'],
               'test_ET_subject_wise_SENSITIVITY': test_final_sensitivity['ET'],
               'test_ET_subject_wise_val_SPECIFCITY': test_final_specificity['ET'],
               'test_TC_subject_wise_DSC': test_final_dice['TC'],
               'test_TC_subject_wise_HD': test_final_hd['TC'],
               'test_TC_subject_wise_PPV': test_final_PPV['TC'],
               'test_TC_subject_wise_SENSITIVITY': test_final_sensitivity['TC'],
               'test_TC_subject_wise_val_SPECIFCITY': test_final_specificity['TC'],
               })


@torch.no_grad()
def save_score(dir_path, score, iteration):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iteration))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average WT dice score per subject at iter {}  :   {}\n"
                "average ET dice score per subject   {}\n"
                "average TC dice score per subject    {}\n".format(iteration, score['WT'], score['ET'], score['TC']))


@torch.no_grad()
def save_score_all(dir_path, scores, iteration, mode):  # mode = 2020 test or 2019 test
    final_dice = scores[0]
    final_hd = scores[1]
    final_PPV = scores[2]
    final_sensitivity = scores[3]
    final_specificity = scores[4]

    dir_path = os.path.join(dir_path, "{}_results_iter{}".format(mode, iteration))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("AVG SCORES PER  SUBJECTS AT ITERATION {}:\n"
                " **WT**  DICE: {}, PPV:{}, Sensitivity: {}, hd: {}, specificity: {}\n"
                " **ET**  DICE: {}, PPV:{}, Sensitivity: {}, hd: {}, specificity: {}\n"
                " **TC**  DICE: {}, PPV:{}, Sensitivity: {}, hd: {}, specificity: {}\n".
                format(iteration, final_dice['WT'], final_PPV['WT'], final_sensitivity['WT'], final_hd['WT'],
                       final_specificity['WT'], final_dice['ET'], final_PPV['ET'], final_sensitivity['ET'],
                       final_hd['ET'], final_specificity['ET'], final_dice['TC'], final_PPV['TC'],
                       final_sensitivity['TC'], final_hd['TC'], final_specificity['TC']))


@torch.no_grad()
def save_score_all2(dir_path, scores, iteration):
    final_dice = scores[0]
    final_hd = scores[1]
    final_PPV = scores[2]
    final_sensitivity = scores[3]

    dir_path = os.path.join(dir_path, "results_iter{}".format(iteration))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("AVG SCORES PER  SUBJECTS AT ITERATION {}:\n"
                " **WT**  DICE: {}, PPV:{}, Sensitivity: {}, h95: {}\n"
                " **ET**  DICE: {}, PPV:{}, Sensitivity: {}, h95: {}\n"
                " **TC**  DICE: {}, PPV:{}, Sensitivity: {}, h95: {}\n"
                .format(iteration, final_dice['WT'], final_PPV['WT'], final_sensitivity['WT'], final_hd['WT'],
                        final_dice['ET'], final_PPV['ET'], final_sensitivity['ET'], final_hd['ET'],
                        final_dice['TC'], final_PPV['TC'], final_sensitivity['TC'], final_hd['TC']))


@torch.no_grad()
def save_predictions(y_pred, threshold, dir_path, score, iteration):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iteration))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average dice score per subject (5 image) at iter {}  :   {}".format(iteration, score))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    for image_id, image in enumerate(y_pred):
        image = image >= threshold
        image = image.to('cpu')
        plt.imshow(image)
        image_path = os.path.join(dir_path, "image_id_{}.jpg".format(image_id))
        plt.savefig(image_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="dasfdsfsaf",
        type=str,
        help="output directory for results"
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--config",
        type=str,
        default='PGS_config.yaml'
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default="CVPR2022_BRATS"
    )

    dataset = utils.Constants.Datasets.Brat20
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda

    if 'semi' in cfg.experiment_mode or cfg.experiment_mode == 'partially_sup_upSample':
        cfg.train_sup_mode = 'train2018_semi_sup' + str(cfg.train_sup_rate)
        cfg.train_unsup_mode = 'train2018_semi_unsup' + str(cfg.train_sup_rate)
    elif cfg.experiment_mode == 'partially_sup':
        cfg.train_sup_mode = 'train2018_semi_sup' + str(cfg.train_sup_rate)
        cfg.train_unsup_mode = None
    elif cfg.experiment_mode == 'fully_sup':
        cfg.train_sup_mode = 'all_train2018_sup'
        cfg.train_unsup_mode = None

    config_params = dict(args=args, config=cfg)

    wandb.init(project=args.wandb, config=config_params)
    Pgs_train_val(dataset, cfg.n_epochs, cfg.wmh_threshold, args.output_dir, args, cfg, cfg.seed)


if __name__ == '__main__':
    main()
