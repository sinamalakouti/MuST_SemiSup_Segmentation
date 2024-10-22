import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from torch.optim import lr_scheduler

import wandb
from dataset.Brat20 import Brat20Test, seg2WT, seg2TC, seg2ET
from losses.evaluation_metrics import dice_coef, do_eval
from losses.loss import consistency_weight, softmax_kl_loss
from models.Baseline_Unet import Unet
from utils import utils
import datetime

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')

for p in sys.path:
    print("path  ", p)

utils.Constants.USE_CUDA = True
parser = argparse.ArgumentParser()


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
        if cfg.unsupervised_training.consistency_loss == 'CE':

            teach_pred = torch.nn.functional.softmax(teach_pred, dim=1)  # todo
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
            if cfg.layerwise != 'layerwiseG':
                teach_pred = torch.nn.functional.softmax(teach_pred / cfg.temp, dim=1)
                teach_pred = torch.nn.functional.softmax(teach_pred / cfg.temp, dim=1)
            stud_pred = torch.nn.functional.softmax(stud_pred, dim=1)
            mse = torch.nn.MSELoss()
            loss = mse(stud_pred, teach_pred.detach())
            losses.append(loss)
    total_loss = sum(losses)
    return total_loss


def compute_loss(y_preds, y_true, loss_functions, is_supervised, cfg):
    if is_supervised:
        if y_true.shape[1] == 1:
            y_true = y_true.reshape((y_true.shape[0], y_true.shape[2], y_true.shape[3]))

        total_loss = loss_functions[0](y_preds, y_true.type(torch.LongTensor).to(y_preds.device))

        ''' y_preds is students preds and y_true is teacher_preds!
                    for comparing outputs together!  # consistency of original output and noisy output 
        '''

    else:
        total_loss = __fw_outputwise_unsup_loss(y_preds, y_true, loss_functions, cfg)

    return total_loss


def trainUnet_semi_alternate(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions,
                             cons_w_unsup, epochid,
                             cfg):
    total_loss = 0
    model.train()

    semi_loader = zip(train_sup_loader, train_unsup_loader)

    for batch_idx, (batch_sup, batch_unsup) in enumerate(semi_loader):
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        b_unsup = batch_unsup['data']
        b_unsup = b_unsup.to(device)

        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)
        weight_unsup = cons_w_unsup(epochid, batch_idx)
        total_loss = uLoss * weight_unsup
        total_loss.backward()
        optimizer[1].step()

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        model.train()

        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)
        sup_outputs, _ = model(b_sup, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)
        sLoss.backward()
        optimizer[0].step()

        print("**************** UNSUP LOSSS  : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))

        with torch.no_grad():
            sf = torch.nn.Softmax2d()
            target_sup[target_sup >= 1] = 1
            target_sup = target_sup
            y_pred = sf(sup_outputs)
            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "sup loss": sLoss,
                 "unsup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "unsup loss": uLoss,
                 "batch_score_WT": dice_score,
                 'weight unsup': weight_unsup
                 })

    return model, total_loss


def trainUnet_semi(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions, epochid, cfg,
                   cons_w_unsup):
    total_loss = 0
    model.train()

    train_unsup_iterator = iter(train_unsup_loader)
    unsup_step = 0
    for sup_step, batch_sup in enumerate(train_sup_loader):
        optimizer.zero_grad()
        b_sup = batch_sup['data']
        b_sup = b_sup.to(device)

        try:
            batch_unsup = next(train_unsup_iterator)

        except StopIteration:
            train_unsup_iterator = iter(train_sup_loader)
            batch_unsup = next(train_unsup_iterator)

        b_unsup = batch_unsup['data'].to(device)
        target_sup = batch_sup['label'].to(device)
        del batch_sup
        del batch_unsup
        torch.cuda.empty_cache()
        sup_outputs, _ = model(b_sup, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)

        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)

        print("**************** UNSUP LOSSS  : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))
        weight_unsup = cons_w_unsup(epochid, sup_step)
        total_loss = weight_unsup * uLoss + sLoss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            sf = torch.nn.Softmax2d()
            target_sup[target_sup >= 1] = 1
            target_sup = target_sup
            y_pred = sf(sup_outputs)
            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": sup_step + epochid * len(train_sup_loader),
                 "sup loss": sLoss,
                 "unsup_batch_id": unsup_step + epochid * len(train_sup_loader),
                 "unsup loss": uLoss,
                 "batch_score_WT": dice_score})
        unsup_step += 1
    return model, total_loss


def trainUnet_sup(train_sup_loader, model, optimizer, device, loss_functions, epochid, cfg):
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
                y_pred = sf(sup_outputs)

            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": step + epochid * len(train_sup_loader), "sup_loss": total_loss,
                 "batch_score": dice_score})

    return model, total_loss


def eval_per_subjectUnet(model, device, threshold, cfg, data_mode):
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
                outputs = sf(outputs)

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
                y_pred = sf(outputs)

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


def initialize_directories(output_dir, seed, cfg):
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'semi_alterate':
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

    output_image_dir = os.path.join(output_dir, "result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)
    return output_model_dir, output_image_dir


def get_train_dataloaders(dataset, cfg):
    train_sup_loader = utils.get_trainset(dataset, batch_size=cfg.batch_size, intensity_rescale=cfg.intensity_rescale,
                                          mixup_threshold=cfg.mixup_threshold, mode=cfg.train_sup_mode, t1=cfg.t1,
                                          t2=cfg.t2, t1ce=cfg.t1ce, augment=cfg.augment, seed=cfg.seed)

    print('size of labeled training set: number of subjects:    ', len(train_sup_loader.dataset.subjects_name))
    print("labeled subjects  ", train_sup_loader.dataset.subjects_name)
    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'partially_sup_upSample' or cfg.experiment_mode == 'semi_downSample' or cfg.experiment_mode == 'semi_alternate':
        train_unsup_loader = utils.get_trainset(dataset, batch_size=cfg.batch_size,
                                                intensity_rescale=cfg.intensity_rescale,
                                                mixup_threshold=cfg.mixup_threshold,
                                                mode=cfg.train_unsup_mode, t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce,
                                                augment=cfg.augment, seed=cfg.seed)

        print('size of unlabeled training set: number of subjects:    ', len(train_unsup_loader.dataset.subjects_name))
        print("un labeled subjects  ", train_unsup_loader.dataset.subjects_name)
    else:
        train_unsup_loader = None

    return train_sup_loader, train_unsup_loader


def updating_modules(cfg, scheduler_sup, scheduler_unsup):
    if cfg.experiment_mode == 'semi_alternate':
        scheduler_sup.step()
        scheduler_unsup.step()
    else:
        scheduler_sup.step()


def Unet_train_val(dataset, n_epochs, wmh_threshold, output_dir, cfg, seed):
    # model's modules dimensions
    inputs_dim = [4, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 4]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # initializing WANDB
    wandb.run.name = "{}_UNET_{}_{}_supRate{}_seed{}_".format(cfg.experiment_mode, "trainALL2018", "valNew2019",
                                                              cfg.train_sup_rate, seed)

    best_score = 0
    start_epoch = 0
    print('-' * 50)
    print("EXPERIMENT-   UNET on BRATS")
    print('-' * 50)
    output_model_dir, output_image_dir = initialize_directories(output_dir, seed, cfg)
    #   loading model
    unet = Unet(inputs_dim, outputs_dim, kernels, strides, cfg)
    #   setting the device
    if torch.cuda.is_available():
        if type(unet) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            unet = torch.nn.DataParallel(unet)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    unet.to(device)

    # loading the datasets
    train_sup_loader, train_unsup_loader = get_train_dataloaders(dataset, cfg)

    # load optimizers
    optimizer_unsup = torch.optim.SGD(unet.parameters(), cfg.unsupervised_training.lr, momentum=0.9,
                                      weight_decay=1e-4)
    optimizer_sup = torch.optim.SGD(unet.parameters(), cfg.supervised_training.lr, momentum=0.9,
                                    weight_decay=1e-4)

    # load schedulers
    scheduler_unsup = lr_scheduler.StepLR(optimizer_unsup,
                                          step_size=cfg.unsupervised_training.scheduler_step_size,
                                          gamma=cfg.unsupervised_training.lr_gamma, verbose=True)
    scheduler_sup = lr_scheduler.StepLR(optimizer_sup, step_size=cfg.supervised_training.scheduler_step_size,
                                        gamma=cfg.supervised_training.lr_gamma, verbose=True)

    # consistency weight scheduler
    cons_w_unsup = consistency_weight(final_w=cfg['unsupervised_training']['consist_w_unsup']['final_w'],
                                      iters_per_epoch=len(train_sup_loader),
                                      rampup_ends=cfg['unsupervised_training']['consist_w_unsup'][
                                          'rampup_ends'],
                                      ramp_type=cfg['unsupervised_training']['consist_w_unsup']['rampup'])

    # Training
    for epoch in range(start_epoch, n_epochs):
        print("iteration:  ", epoch)

        if cfg.experiment_mode == 'semi':
            unet, loss = trainUnet_semi(train_sup_loader, train_unsup_loader, unet, optimizer_sup, device,
                                        (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()), epoch, cfg,
                                        cons_w_unsup)
        elif cfg.experiment_mode == 'semi_alternate':
            unet, loss = trainUnet_semi_alternate(train_sup_loader, train_unsup_loader, unet,
                                                  (optimizer_sup, optimizer_unsup), device,
                                                  (torch.nn.CrossEntropyLoss(), None),
                                                  cons_w_unsup,
                                                  epoch, cfg)
        elif cfg.experiment_mode == 'partially_sup':
            unet, loss = trainUnet_sup(train_sup_loader, unet, optimizer_sup, device,
                                       (torch.nn.CrossEntropyLoss(), None),
                                       epoch, cfg)
        elif cfg.experiment_mode == 'fully_sup':
            unet, loss = trainUnet_sup(train_sup_loader, unet, optimizer_sup, device,
                                       (torch.nn.CrossEntropyLoss(), None),
                                       epoch, cfg)
        # log every 2 epochs
        if epoch % 2 == 0:
            val_score = evaluate(unet, device, wmh_threshold, cfg, epoch, output_image_dir)

            if val_score > best_score:
                print('*' * 50)
                print("BEST VALIDATION SCORE @ ITERATION {} is {}".format(epoch, val_score))
                best_score = val_score
                path = os.path.join(output_model_dir, 'best_model.model')
                with open(path, 'wb') as f:  # todo
                    torch.save(unet, f)

        # updating modules
        updating_modules(cfg, scheduler_sup, scheduler_unsup)

    val_score = evaluate(unet, device, wmh_threshold, cfg, cfg.n_epoch, output_image_dir)
    if val_score > best_score:
        print('*' * 50)
        print("BEST VALIDATION SCORE @ ITERATION {} is {}".format(cfg.n_epoch, val_score))
        best_score = val_score
        path = os.path.join(output_model_dir, 'best_model.model')
        with open(path, 'wb') as f:  # todo
            torch.save(unet, f)


def evaluate(unet, device, wmh_threshold, cfg, epoch, output_image_dir, ):
    # Evaluate on Validation set
    val_final_dice, val_final_PPV, val_final_sensitivity, val_final_specificity, val_final_hd = eval_per_subjectUnet(
        unet, device,
        wmh_threshold,
        cfg,
        cfg.val_mode)

    # Evaluate on Test set
    test_final_dice, test_final_PPV, test_final_sensitivity, test_final_specificity, test_final_hd = \
        eval_per_subjectUnet(
            unet, device,
            wmh_threshold,
            cfg,
            cfg.test_mode)

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
    return val_final_dice['WT']


def save_score_all(dir_path, scores, iter, mode):  # mode = 2020 test or 2019 test
    final_dice = scores[0]
    final_hd = scores[1]
    final_PPV = scores[2]
    final_sensitivity = scores[3]
    final_specificity = scores[4]

    dir_path = os.path.join(dir_path, "{}_results_iter{}".format(mode, iter))
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
        default='config.yaml'
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda Commented : specify when running the code

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'semi_alternate':
        cfg.train_sup_mode = 'train2018_semi_sup' + str(cfg.train_sup_rate)
        cfg.train_unsup_mode = 'train2018_semi_unsup' + str(cfg.train_sup_rate)
    elif cfg.experiment_mode == 'partially_sup':
        cfg.train_sup_mode = 'train2018_semi_sup' + str(cfg.train_sup_rate)
        cfg.train_unsup_mode = None
    elif cfg.experiment_mode == 'fully_sup':
        cfg.train_sup_mode = 'all_train2018_sup'
        cfg.train_unsup_mode = None
    config_params = dict(args=args, config=cfg)
    wandb.init(project="CVPR2022_BRATS", config=config_params)
    Unet_train_val(dataset, cfg.n_epochs, cfg.wmh_threshold, args.output_dir, cfg, cfg.seed)


if __name__ == '__main__':
    main()
