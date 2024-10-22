import os
import sys
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import utils
from losses.loss import consistency_weight, softmax_kl_loss, Consistency_CE

from losses.evaluation_metrics import dice_coef, get_dice_coef_per_subject, do_eval
from dataset.Brat20 import Brat20Test, seg2WT, seg2TC, seg2ET

from models import Pgs_MT
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import random
import argparse
import yaml
from easydict import EasyDict as edict

import wandb

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')

for p in sys.path:
    print("path  ", p)
# random_seeds = [41, 42, 43]

utils.Constants.USE_CUDA = True
parser = argparse.ArgumentParser()

global_step = 0


def __fw_outputwise_unsup_loss(y_stud, y_teach, loss_functions, cfg):
    (_, unsup_loss) = loss_functions

    assert len(y_teach) == len(y_stud), "Error! unsup_preds and sup_preds have to have same length"
    num_preds = len(y_teach)
    losses = []
    for i in range(num_preds):
        teach_pred = y_teach[i]

        stud_pred = y_stud[i]
        assert teach_pred.shape == stud_pred.shape, "Error! for preds number {}, supervised and unsupervised" \
                                                    " prediction shape is not similar!".format(i)

        if cfg.consistency_loss == 'CE':
            losses.append(- torch.mean(
                torch.sum(teach_pred.detach()
                          * torch.nn.functional.log_softmax(stud_pred / 0.5, dim=1), dim=1)))
        elif cfg.consistency_loss == 'KL':
            losses.append(
                softmax_kl_loss(stud_pred, teach_pred.detach(), conf_mask=False, threshold=None, use_softmax=True))

        elif cfg.consistency_loss == 'balanced_CE':
            losses.append(unsup_loss(stud_pred, teach_pred, i, use_softmax=True))
    total_loss = sum(losses)
    return total_loss


def __fw_sup_loss(y_preds, y_true, sup_loss):
    total_loss = 0
    # iterate over all level's output
    losses = []
    for output in y_preds:
        ratio = int(np.round(y_true.shape[2] / output.shape[2]))
        maxpool = nn.MaxPool2d(kernel_size=2, stride=ratio, padding=0)
        target = maxpool(y_true)

        if target.shape[-1] != output.shape[-1] or target.shape[-2] != output.shape[-2]:
            h_diff = output.size()[-2] - target.size()[-2]
            w_diff = output.size()[-1] - target.size()[-1]
            #
            target = F.pad(target, (w_diff // 2, w_diff - w_diff // 2,
                                    h_diff // 2, h_diff - h_diff // 2))

        assert output.shape[-2:] == target.shape[-2:], "output and target shape is not similar!!"
        if output.shape[1] != target.shape[1] and type(sup_loss) == torch.nn.CrossEntropyLoss and len(target.shape) > 3:
            target = target.reshape((target.shape[0], target.shape[2], target.shape[3])).type(torch.LongTensor)
        losses.append(sup_loss(output, target.type(torch.LongTensor).to(output.device)))
    total_loss = sum(losses)
    return total_loss


def compute_loss(y_preds, y_true, loss_functions, is_supervised, cfg):
    if is_supervised:
        total_loss = __fw_sup_loss(y_preds, y_true, loss_functions[0])

        ''' y_preds is students preds and y_true is teacher_preds!
                    for comparing outputs together!  # consistency of original output and noisy output 
        '''

    else:
        total_loss = __fw_outputwise_unsup_loss(y_preds, y_true, loss_functions, cfg)

    return total_loss


def trainPgs_semi_downSample(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions,
                             cons_w_unsup, epochid,
                             cfg):
    total_loss = 0
    model.train()

    semi_dataLoader = zip(train_sup_loader, train_unsup_loader)

    for batch_idx, (batch_sup, batch_unsup) in enumerate(semi_dataLoader):
        optimizer.zero_grad()
        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)
        sup_outputs, _ = model(b_sup, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)

        b_unsup = batch_unsup['data']
        b_unsup = b_unsup.to(device)
        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)

        print("**************** UNSUP LOSSS  : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))
        weight_unsup = cons_w_unsup(epochid, batch_idx)
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
                {"sup_batch_id": batch_idx + epochid * min(len(train_unsup_loader), len(train_sup_loader)),
                 "sup loss": sLoss,
                 "unsup_batch_id": batch_idx + epochid * min(len(train_unsup_loader), len(train_sup_loader)),
                 "unsup loss": uLoss,
                 "batch_score_WT": dice_score,
                 'weight unsup': weight_unsup
                 })

    return model, total_loss


def trainPgs_semi(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions, cons_w_unsup, epochid,
                  cfg):
    global global_step
    total_loss = 0
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

        sup_outputs, _ = model(b_sup, is_supervised=True, model_for_sup='stud')
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)

        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)

        print("**************** UNSUP LOSSS  : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))
        weight_unsup = cons_w_unsup(epochid, unsup_step)
        total_loss = sLoss + uLoss
        total_loss.backward()
        optimizer.step()
        global_step += 1
        if global_step % 100 == 0 and global_step < 1000:
            print("glboal step is: ", global_step)

        model.module.update_params(global_step, epochid, max_step=5 * len(train_unsup_loader),
                                   max_epoch=5)  # 6 num of epochs

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
                 "batch_score_WT": dice_score})
        sup_step += 1
    return model, total_loss


def trainPgs_sup(train_sup_loader, model, optimizer, device, loss_functions, epochid, cfg):
    total_loss = 0
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

    return model, total_loss


def eval_per_subjectPgs(model, device, threshold, cfg, data_mode, model_for_sup):
    print("******************** EVALUATING {}********************".format(data_mode))

    testset = Brat20Test(f'../data/brats20', data_mode, 10, 155,
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
            outputs, _ = model(b, True, model_for_sup=model_for_sup)

            if cfg.oneHot:
                sf = torch.nn.Softmax2d()
                outputs = sf(outputs[-1])

            loss_val = compute_loss(outputs, target, (sup_loss, None), is_supervised=True, cfg=cfg)
            print("############# LOSS for subject {} is {} ##############".format(subjects[0], loss_val.item()))
            if cfg.oneHot:
                target[target >= 1] = 1
                target_WT = seg2WT(target, 1, oneHot=cfg.oneHot)
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

            # dice_scoreWT = dice_coef(targetWT.reshape(y_WT.shape), y_WT)
            # dice_scoreET = dice_coef(targetET, y_ET)
            # dice_scoreTC = dice_coef(targetTC.reshape(y_TC.shape), y_TC)
            #
            # PPV_scoreWT, sensitivity_WT, specificity_WT = get_confusionMatrix_metrics(targetWT.reshape(y_WT.shape),
            #                                                                           y_WT)
            # PPV_scoreET, sensitivity_ET, specificity_ET = get_confusionMatrix_metrics(targetET, y_ET)
            # PPV_scoreTC, sensitivity_TC, specificity_TC = get_confusionMatrix_metrics(targetTC.reshape(y_TC.shape),
            #                                                                           y_TC)

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
            ### ET <-> TC
            PPV_arrWT.append(metrics_WT['ppv'].item())
            PPV_arrET.append(metrics_ET['ppv'].item())
            PPV_arrTC.append(metrics_TC['ppv'].item())
            ### ET <-> TC
            sensitivity_arrWT.append(metrics_WT['sens'].item())
            sensitivity_arrET.append(metrics_ET['sens'].item())
            sensitivity_arrTC.append(metrics_TC['sens'].item())
            ### ET <-> TC
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


def evaluatePGS(model, dataset, device, threshold, cfg, training_mode):
    print("******************** EVALUATING  : {} ********************".format(training_mode))

    testset = utils.get_testset(dataset, cfg.batch_size, intensity_rescale=cfg.intensity_rescale,
                                mixup_threshold=cfg.mixup_threshold, mode=training_mode,
                                t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce, augment=False)

    model.eval()

    dice_arr = []
    segmentation_outputs = []
    all_preds = None
    all_targets = None
    all_subjects = None

    with torch.no_grad():
        for batch_id, batch in enumerate(testset):
            b = batch['data']
            b = b.to(device)
            target = batch['label'].to(device)
            predictions, _ = model(b, True)
            # apply softmax
            sf = torch.nn.Softmax2d()
            y_pred = sf(predictions[-1])

            if batch_id == 0:
                all_preds = y_pred
                all_subjects = batch['subject']
                all_targets = target
            else:
                all_preds = torch.cat((all_preds, y_pred), dim=0)
                all_subjects = torch.cat((all_subjects, batch['subject']), dim=0)
                all_targets = torch.cat((all_targets, target), dim=0)

            y_WT = seg2WT(y_pred, threshold)
            target[target >= 1] = 1
            target_WT = target
            dice_score = dice_coef(target_WT.reshape(y_WT.shape), y_WT)
            dice_arr.append(dice_score.detach().item())
            outputs = predictions[-1].reshape(predictions[-1].shape[0], predictions[-1].shape[1],
                                              predictions[-1].shape[2],
                                              predictions[-1].shape[3])
            for output in outputs:
                segmentation_outputs.append(output)

    all_targets[all_targets >= 1] = 1
    all_preds = seg2WT(all_preds, threshold)
    subject_wise_DSC = get_dice_coef_per_subject(all_targets.reshape(all_preds.shape), all_preds, all_subjects)

    return np.mean(np.array(dice_arr)), subject_wise_DSC, segmentation_outputs


def Pgs_train_val(dataset, n_epochs, wmh_threshold, output_dir, learning_rate, args, cfg, seed):
    inputs_dim = [4, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 4]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    wandb.run.name = "{}_PGSMT_{}_{}_supRate{}_seed{}_".format(cfg.experiment_mode, "trainALL2018", "valNew2019",
                                                               cfg.train_sup_rate, seed)

    dataroot_dir = f'../data/brats20'
    all_train_csv = os.path.join(dataroot_dir, 'trainset/brats2018.csv')
    supdir_path = os.path.join(dataroot_dir, 'trainset')

    # semi_sup_split(all_train_csv=all_train_csv, sup_dir_path=supdir_path, unsup_dir_path=supdir_path,
    #                ratio=cfg.train_sup_rate / 100, seed=cfg.seed)

    step_size = cfg.scheduler_step_size

    best_score = 0
    start_epoch = 0
    global global_step

    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'semi_downSample':
        output_dir = os.path.join(output_dir, "sup_ratio_{}".format(cfg.train_sup_rate))
        if not os.path.isdir(output_dir):
            try:
                os.mkdir(output_dir, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % output_dir)
    elif cfg.experiment_mode == 'partially_sup':
        output_dir = os.path.join(output_dir, "partiallySup_ratio_{}".format(cfg.train_sup_rate))
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

    output_image_dir = os.path.join(output_dir, "result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)

    # log_file_path = os.path.join(output_dir, "log_{}.log".format(args.training_mode))
    # sys.stdout = open(log_file_path, "w")
    print("EXPERIMENT DESCRIPTION:   {}".format(cfg.information))

    print("******* TRAINING PGS ***********")
    print("learning_rate is    ", learning_rate)
    # print("scheduler step size is :   ", step_size)
    print("output_dir is    ", output_dir)

    pgsnet = Pgs_MT.PGS_MT(inputs_dim, outputs_dim, kernels, strides, cfg)

    if torch.cuda.is_available():
        if type(pgsnet) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            pgsnet = torch.nn.DataParallel(pgsnet)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)
    pgsnet.to(device)
    optimizer = torch.optim.SGD(pgsnet.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(pgsnet.parameters(), lr=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.lr_gamma)  # don't use it
    # final_dice, final_PPV, final_sensitivity, final_specificity, final_hd = eval_per_subjectPgs(pgsnet, device,
    #                                                                                             wmh_threshold,
    #                                                                                             cfg,
    #                                                                                             cfg.val_mode)
    train_sup_loader = utils.get_trainset(dataset, batch_size=cfg.batch_size, intensity_rescale=cfg.intensity_rescale,
                                          mixup_threshold=cfg.mixup_threshold, mode=cfg.train_sup_mode, t1=cfg.t1,
                                          t2=cfg.t2, t1ce=cfg.t1ce, augment=cfg.augment, seed=cfg.seed)

    print('size of labeled training set: number of subjects:    ', len(train_sup_loader.dataset.subjects_name))
    print("labeled subjects  ", train_sup_loader.dataset.subjects_name)

    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'semi_downSample':
        train_unsup_loader = utils.get_trainset(dataset, batch_size=32, intensity_rescale=cfg.intensity_rescale,
                                                mixup_threshold=cfg.mixup_threshold,
                                                mode=cfg.train_unsup_mode, t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce,
                                                augment=cfg.augment, seed=cfg.seed)
        print('size of unlabeled training set: number of subjects:    ', len(train_unsup_loader.dataset.subjects_name))
        print("un labeled subjects  ", train_unsup_loader.dataset.subjects_name)

    sup_loss_fn = torch.nn.CrossEntropyLoss()
    if cfg.consistency_loss == 'balanced_CE':
        cons_loss_fn = Consistency_CE(5)
    else:
        cons_loss_fn = None

    for epoch in range(start_epoch, n_epochs):
        print("iteration:  ", epoch)

        # pgsnet, loss = trainPGS(train_loader, pgsnet, optimizer, device, epoch)
        if cfg.experiment_mode == 'semi':
            if epoch < 3 and False:  # todo if epoch < a -> train supervised
                print("training supervised because epoch < 3")
                pgsnet, loss = trainPgs_sup(train_sup_loader, pgsnet, optimizer, device,
                                            (torch.nn.CrossEntropyLoss(), None),
                                            epoch, cfg)
            else:
                rampup_ends = 3
                cons_w_unsup = consistency_weight(final_w=5,
                                                  iters_per_epoch=len(train_unsup_loader),
                                                  rampup_ends=rampup_ends)

                pgsnet, loss = trainPgs_semi(train_sup_loader, train_unsup_loader, pgsnet, optimizer, device,
                                             (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()), cons_w_unsup,
                                             epoch, cfg)
        elif cfg.experiment_mode == 'semi_downSample':
            cons_w_unsup = consistency_weight(final_w=cfg['consist_w_unsup']['final_w'],
                                              iters_per_epoch=len(train_sup_loader),
                                              rampup_ends=cfg['consist_w_unsup']['rampup_ends'],
                                              ramp_type=cfg['consist_w_unsup']['rampup'])

            pgsnet, loss = trainPgs_semi_downSample(train_sup_loader, train_unsup_loader, pgsnet, optimizer, device,
                                                    (torch.nn.CrossEntropyLoss(), cons_loss_fn),
                                                    cons_w_unsup,
                                                    epoch, cfg)

        # score, segmentations = evaluatePGS(pgsnet, dataset, device, wmh_threshold, cfg, cfg.val_mode)
        elif cfg.experiment_mode == 'partially_sup':
            pgsnet, loss = trainPgs_sup(train_sup_loader, pgsnet, optimizer, device,
                                        (torch.nn.CrossEntropyLoss(), None),
                                        epoch, cfg)
        elif cfg.experiment_mode == 'fully_sup':
            pgsnet, loss = trainPgs_sup(train_sup_loader, pgsnet, optimizer, device,
                                        (torch.nn.CrossEntropyLoss(), None),
                                        epoch, cfg)

        if epoch % 2 == 0:
            final_dice, final_PPV, final_sensitivity, final_specificity, final_hd = eval_per_subjectPgs(pgsnet, device,
                                                                                                        wmh_threshold,
                                                                                                        cfg,
                                                                                                        cfg.val_mode,
                                                                                                        model_for_sup='teach')
            print(
                "** (WT) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, PPV:{}, Sensitivity: {}, Specificity: {}  **".
                    format(epoch, final_dice['WT'], final_hd['WT'], final_PPV['WT'], final_sensitivity['WT'],
                           final_specificity['WT']))
            print(
                "** (ET) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, :{}, Sensitivity:{}, Specificity: {}  **".
                    format(epoch, final_dice['ET'], final_hd['ET'], final_PPV['ET'], final_sensitivity['ET'],
                           final_specificity['ET']))
            print(
                "** (TC) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, PPV:{}, Sensitivity:{}, Specificity: {}  **".
                    format(epoch, final_dice['TC'], final_hd['TC'], final_PPV['TC'], final_sensitivity['TC'],
                           final_specificity['TC']))

            if final_dice['WT'] > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch,
                                                                                                     final_dice['WT']))
                best_score = final_dice['WT']
                path = os.path.join(output_model_dir, 'pgsnet_best_lr{}.model'.format(learning_rate))
                with open(path, 'wb') as f:
                    torch.save(pgsnet, f)
            save_score_all(output_image_dir, (final_dice, final_hd, final_PPV, final_sensitivity, final_specificity),
                           epoch)

            final_dice_stud, _, _, _, _ = eval_per_subjectPgs(pgsnet, device, wmh_threshold, cfg, cfg.val_mode,
                                                              model_for_sup='stud')
            wandb.log({'epoch_id': epoch,
                       'WT_subject_wise_val_DSC': final_dice['WT'],
                       'WT_subject_wise_val_HD': final_hd['WT'],
                       'WT_subject_wise_val_PPV': final_PPV['WT'],
                       'WT_subject_wise_val_SENSITIVITY': final_sensitivity['WT'],
                       'WT_subject_wise_val_SPECIFCITY': final_specificity['WT'],
                       'ET_subject_wise_val_DSC': final_dice['ET'],
                       'ET_subject_wise_val_HD': final_hd['ET'],
                       'ET_subject_wise_val_PPV': final_PPV['ET'],
                       'ET_subject_wise_val_SENSITIVITY': final_sensitivity['ET'],
                       'ET_subject_wise_val_SPECIFCITY': final_specificity['ET'],
                       'TC_subject_wise_val_DSC': final_dice['TC'],
                       'TC_subject_wise_val_HD': final_hd['TC'],
                       'TC_subject_wise_val_PPV': final_PPV['TC'],
                       'TC_subject_wise_val_SENSITIVITY': final_sensitivity['TC'],
                       'TC_subject_wise_val_SPECIFCITY': final_specificity['TC'],
                       'Stud_WT_subject_wise_val_DSC': final_dice_stud['WT'],
                       'Stud_ET_subject_wise_val_DSC': final_dice_stud['ET'],
                       'Stud_TC_subject_wise_val_DSC': final_dice_stud['TC']
                       })

            # final_dice, final_hd, final_PPV, final_sensitivity = eval_per_subjectPgs2(pgsnet, device, wmh_threshold,
            #                                                                           cfg, cfg.val_mode)
            #
            # print("** (WT) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, H95: {}, PPV:{}, Sensitivity:{} **".
            #       format(epoch, final_dice['WT'], final_hd['WT'], final_PPV['WT'], final_sensitivity['WT']))
            # print("** (ET) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, H95: {}, PPV:{}, Sensitivity:{} **".
            #       format(epoch, final_dice['ET'], final_hd['ET'], final_PPV['ET'], final_sensitivity['ET']))
            # print("** (TC) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, H95: PPV{}, PPV:{}, Sensitivity:{} **".
            #       format(epoch, final_dice['TC'], final_hd['TC'], final_PPV['TC'], final_sensitivity['TC']))
            #
            #
            # if final_dice['WT'] > best_score:
            #     print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch,
            #                                                                                          final_dice['WT']))
            #     best_score = final_dice['WT']
            #     path = os.path.join(output_model_dir, 'pgsnet_best_lr{}.model'.format(learning_rate))
            #     with open(path, 'wb') as f:
            #         torch.save(pgsnet, f)
            #     # batch_wise_test_DSC, subject_wise_test_DSC, _ = evaluatePGS(pgsnet, dataset, device, wmh_threshold,
            #     #                                                             cfg, cfg.test_mode)
            #     # subject_wise_test_DSC = eval_per_subjectPgs(pgsnet, device, wmh_threshold, cfg, cfg.test_mode)
            #     #
            #     # wandb.log({"epoch_id": epoch, "subject_wise_test_DSC": subject_wise_test_DSC})
            # save_score_all(output_image_dir, (final_dice, final_hd, final_PPV, final_sensitivity), epoch)
            # wandb.log({'epoch_id': epoch,
            #            'WT_subject_wise_val_DSC': final_dice['WT'], 'WT_subject_wise_val_H95': final_hd['WT'],
            #            'WT_subject_wise_val_PPV': final_PPV['WT'],
            #            'WT_subject_wise_val_SENSITIVITY': final_sensitivity['WT'],
            #            'ET_subject_wise_val_DSC': final_dice['ET'], 'ET_subject_wise_val_H95': final_hd['ET'],
            #            'ET_subject_wise_val_PPV': final_PPV['ET'],
            #            'ET_subject_wise_val_SENSITIVITY': final_sensitivity['ET'],
            #            'TC_subject_wise_val_DSC': final_dice['TC'], 'TC_subject_wise_val_H95': final_hd['TC'],
            #            'TC_subject_wise_val_PPV': final_PPV['TC'],
            #            'TC_subject_wise_val_SENSITIVITY': final_sensitivity['TC']
            #            })

        scheduler.step()

    final_dice, final_PPV, final_sensitivity, final_specificity, final_hd = eval_per_subjectPgs(pgsnet, device,
                                                                                                wmh_threshold,
                                                                                                cfg,
                                                                                                cfg.val_mode)
    print(
        "** (WT) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, PPV:{}, Sensitivity: {}, Specificity: {}  **".
            format(cfg.n_epochs, final_dice['WT'], final_hd['WT'], final_PPV['WT'], final_sensitivity['WT'],
                   final_specificity['WT']))
    print(
        "** (ET) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, :{}, Sensitivity:{}, Specificity: {}  **".
            format(cfg.n_epochs, final_dice['ET'], final_hd['ET'], final_PPV['ET'], final_sensitivity['ET'],
                   final_specificity['ET']))
    print(
        "** (TC) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, PPV:{}, Sensitivity:{}, Specificity: {}  **".
            format(cfg.n_epochs, final_dice['TC'], final_hd['TC'], final_PPV['TC'], final_sensitivity['TC'],
                   final_specificity['TC']))

    save_score_all(output_image_dir, (final_dice, final_hd, final_PPV, final_sensitivity, final_specificity),
                   cfg.n_epochs)
    wandb.log({'epoch_id': cfg.n_epochs,
               'WT_subject_wise_val_DSC': final_dice['WT'],
               'WT_subject_wise_val_HD': final_hd['WT'],
               'WT_subject_wise_val_PPV': final_PPV['WT'],
               'WT_subject_wise_val_SENSITIVITY': final_sensitivity['WT'],
               'WT_subject_wise_val_SPECIFCITY': final_specificity['WT'],
               'ET_subject_wise_val_DSC': final_dice['ET'],
               'ET_subject_wise_val_HD': final_hd['ET'],
               'ET_subject_wise_val_PPV': final_PPV['ET'],
               'ET_subject_wise_val_SENSITIVITY': final_sensitivity['ET'],
               'ET_subject_wise_val_SPECIFCITY': final_specificity['ET'],
               'TC_subject_wise_val_DSC': final_dice['TC'],
               'TC_subject_wise_val_HD': final_hd['TC'],
               'TC_subject_wise_val_PPV': final_PPV['TC'],
               'TC_subject_wise_val_SENSITIVITY': final_sensitivity['TC'],
               'TC_subject_wise_val_SPECIFCITY': final_specificity['TC'],
               })
    # final_dice, final_hd, final_PPV, final_sensitivity = eval_per_subjectPgs2(pgsnet, device, wmh_threshold,
    #                                                                           cfg, cfg.val_mode)
    #
    # print("** (WT) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, H95: {}, PPV:{}, Sensitivity:{} **".
    #       format(cfg.n_epochs, final_dice['WT'], final_hd['WT'], final_PPV['WT'], final_sensitivity['WT']))
    # print("** (ET) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, H95: {}, PPV:{}, Sensitivity:{} **".
    #       format(cfg.n_epochs, final_dice['ET'], final_hd['ET'], final_PPV['ET'], final_sensitivity['ET']))
    # print("** (TC) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, H95: PPV{}, PPV:{}, Sensitivity:{} **".
    #       format(cfg.n_epochs, final_dice['TC'], final_hd['TC'], final_PPV['TC'], final_sensitivity['TC']))
    #
    #
    #
    # #
    # save_score_all(output_image_dir, (final_dice, final_hd, final_PPV, final_sensitivity), cfg.n_epochs)
    # wandb.log({'epoch_id': cfg.n_epochs,
    #            'WT_subject_wise_val_DSC': final_dice['WT'], 'WT_subject_wise_val_H95': final_hd['WT'],
    #            'WT_subject_wise_val_PPV': final_PPV['WT'],
    #            'WT_subject_wise_val_SENSITIVITY': final_sensitivity['WT'],
    #            'ET_subject_wise_val_DSC': final_dice['ET'], 'ET_subject_wise_val_H95': final_hd['ET'],
    #            'ET_subject_wise_val_PPV': final_PPV['ET'],
    #            'ET_subject_wise_val_SENSITIVITY': final_sensitivity['ET'],
    #            'TC_subject_wise_val_DSC': final_dice['TC'], 'TC_subject_wise_val_H95': final_hd['TC'],
    #            'TC_subject_wise_val_PPV': final_PPV['TC'],
    #            'TC_subject_wise_val_SENSITIVITY': final_sensitivity['TC']
    #            })
    #
    # # save_score_all(output_image_dir, (final_dice, final_specificity, final_PPV, final_sensitivity), 39)
    # # wandb.log({'epoch_id': cfg.n_epochs,
    # #            'WT_subject_wise_val_DSC': final_dice['WT'],
    # #            'WT_subject_wise_val_PPV': final_PPV['WT'],
    # #            'WT_subject_wise_val_SENSITIVITY': final_sensitivity['WT'],
    # #            'WT_subject_wise_specificity': final_specificity['WT'],
    # #            'ET_subject_wise_val_DSC': final_dice['ET'],
    # #            'ET_subject_wise_val_PPV': final_PPV['ET'],
    # #            'ET_subject_wise_val_SENSITIVITY': final_sensitivity['ET'],
    # #            'ET_subject_wise_specificity': final_specificity['ET'],
    # #            'TC_subject_wise_val_DSC': final_dice['TC'],
    # #            'TC_subject_wise_val_PPV': final_PPV['TC'],
    # #            'TC_subject_wise_val_SENSITIVITY': final_sensitivity['TC'],
    # #            'TC_subject_wise_specificity': final_specificity['TC'],
    # #            })


def save_score(dir_path, score, iter):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average WT dice score per subject at iter {}  :   {}\n"
                "average ET dice score per subject   {}\n"
                "average TC dice score per subject    {}\n".format(iter, score['WT'], score['ET'], score['TC']))


def save_score_all(dir_path, scores, iter):
    final_dice = scores[0]
    final_hd = scores[1]
    final_PPV = scores[2]
    final_sensitivity = scores[3]
    final_specificity = scores[4]

    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
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
                " **TC**  DICE: {}, PPV:{}, Sensitivity: {}, hd: {}, specificity: {}\n".format(
            iter, final_dice['WT'], final_PPV['WT'], final_sensitivity['WT'], final_hd['WT'], final_specificity['WT'],
            final_dice['ET'], final_PPV['ET'], final_sensitivity['ET'], final_hd['ET'], final_specificity['ET'],
            final_dice['TC'], final_PPV['TC'], final_sensitivity['TC'], final_hd['TC'], final_specificity['TC']))


def save_score_all2(dir_path, scores, iter):
    final_dice = scores[0]
    final_hd = scores[1]
    final_PPV = scores[2]
    final_sensitivity = scores[3]

    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
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
                " **TC**  DICE: {}, PPV:{}, Sensitivity: {}, h95: {}\n".format(
            iter, final_dice['WT'], final_PPV['WT'], final_sensitivity['WT'], final_hd['WT'],
            final_dice['ET'], final_PPV['ET'], final_sensitivity['ET'], final_hd['ET'],
            final_dice['TC'], final_PPV['TC'], final_sensitivity['TC'], final_hd['TC']))


def save_predictions(y_pred, threshold, dir_path, score, iter):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average dice score per subject (5 image) at iter {}  :   {}".format(iter, score))
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
        "--config",
        type=str,
        default='PGS_config.yaml'
    )

    parser.add_argument(
        "--training_mode",
        default="semi_sup",
        type=str,
        help="training mode supervised (sup), n subject supervised (n_sup), all supervised (all_sup)"
    )

    dataset = utils.Constants.Datasets.Brat20
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda

    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'semi_downSample':
        cfg.train_sup_mode = 'train2018_semi_sup' + str(cfg.train_sup_rate)
        cfg.train_unsup_mode = 'train2018_semi_unsup' + str(cfg.train_sup_rate)
    elif cfg.experiment_mode == 'partially_sup':
        cfg.train_sup_mode = 'train2018_semi_sup' + str(cfg.train_sup_rate)
        cfg.train_unsup_mode = None
    elif cfg.experiment_mode == 'fully_sup':
        cfg.train_sup_mode = 'all_train2018_sup'
        cfg.train_unsup_mode = None

    config_params = dict(args=args, config=cfg)
    cfg.model = "PGSMAT"
    wandb.init(project="CVPR2022_BRATS", config=config_params)
    Pgs_train_val(dataset, cfg.n_epochs, cfg.wmh_threshold, args.output_dir, cfg.lr, args, cfg, cfg.seed)


if __name__ == '__main__':
    main()
