import os
import sys
import datetime
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import utils, eval_utils
from utils import model_utils
from losses.loss import consistency_weight, Consistency_CE, softmax_kl_loss

from evaluation_metrics import dice_coef, get_dice_coef_per_subject, get_confusionMatrix_metrics, do_eval

from dataset.wmh_utils import get_splits
from dataset.Brat20 import seg2WT
from dataset.wmhChallenge import WmhChallenge
from models import Pgs
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
sys.path.append('src/utils')

for p in sys.path:
    print("path  ", p)
# random_seeds = [41, 42, 43]

utils.Constants.USE_CUDA = True
parser = argparse.ArgumentParser()
@torch.no_grad()
def temp_rampDown(x_s, y_s, x_e, y_e, cur_x):

    if cur_x >= x_e :
        return y_e
    r = (y_e-y_s)/(x_e-x_s)
    cur_y = r * cur_x + y_s
    return cur_y

def __fw_outputwise_unsup_loss(y_stud, y_teach, loss_functions, cfg, cur_epoch):
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
            
            print("hereee22222")
            print(teach_pred.shape)
            print(stud_pred.shape)
          #  cur_temp = temp_rampDown(cfg.temp.start, cfg.temp.min, cfg.temp.end,cfg.temp.max, cur_epoch)
            teach_pred = torch.nn.functional.softmax(teach_pred / cfg.temp, dim=1)
            stud_pred = torch.nn.functional.softmax(stud_pred, dim=1)
            mse = torch.nn.MSELoss()
            loss = mse(stud_pred, teach_pred.detach())
            losses.append(loss)
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


def compute_loss(y_preds, y_true, loss_functions, is_supervised, cfg, cur_epoch = 0):
    if is_supervised:
        total_loss = __fw_sup_loss(y_preds, y_true, loss_functions[0])

        ''' y_preds is students preds and y_true is teacher_preds!
                    for comparing outputs together!  # consistency of original output and noisy output 
        '''

    else:
        total_loss = __fw_outputwise_unsup_loss(y_preds, y_true, loss_functions, cfg, cur_epoch)

    return total_loss


def trainPgs_semi_alternate(train_sup_loader, train_unsup_loader, model, optimizer, device, loss_functions,
                            cons_w_unsup, epochid,
                            cfg):
    total_loss = 0
    model.train()
    torch.cuda.empty_cache()
    semi_loader = zip(train_sup_loader, train_unsup_loader)

    for batch_idx, (batch_sup, batch_unsup) in enumerate(semi_loader):
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        b_unsup = batch_unsup['img']
        b_unsup = b_unsup.to(device)

        teacher_outputs, student_outputs = model(b_unsup, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg, cur_epoch=epochid)
        weight_unsup = cons_w_unsup(epochid, batch_idx)
        total_loss = uLoss * weight_unsup
        total_loss.backward()
        optimizer[1].step()

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        model.train()
        b_unsup = b_unsup.to('cpu')
        del batch_unsup
      #  print("dlete   dafdsaffdasfsfsdfdfdff ,", batch_unsup)
        
        
        b_sup = batch_sup['img'].to(device)
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
            y_pred = sf(sup_outputs[-1])
            y_WMH = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WMH.shape), y_WMH)
            wandb.log(
                {"sup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "sup loss": sLoss,
                 "unsup_batch_id": batch_idx + epochid * len(train_sup_loader),
                 "unsup loss": uLoss,
                 "batch_score_WMH": dice_score,
                 'weight unsup': weight_unsup
                 })

        del b_sup
        del target_sup
                
    return model, total_loss


def trainPgs_sup(train_sup_loader, model, optimizer, device, loss_functions, epochid, cfg):
    total_loss = 0
    model.train()
    sup_loss = loss_functions[0]

    for step, batch_sup in enumerate(train_sup_loader):

        optimizer.zero_grad()
        b_sup = batch_sup['img'].to(device)
        target_sup = batch_sup['label'].to(device)

        # print("subject is : ", batch_sup['subject'])
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
                y_pred = sf(sup_outputs[-1])

            y_WT = seg2WT(y_pred, 0.5, cfg.oneHot)
            dice_score = dice_coef(target_sup.reshape(y_WT.shape), y_WT)
            wandb.log(
                {"sup_batch_id": step + epochid * len(train_sup_loader), "sup_loss": total_loss,
                 "batch_score": dice_score})

    return model, total_loss


def eval_per_subjectPgs(model, device, threshold, cfg, data_mode, val_loader):
    print("******************** EVALUATING {}********************".format(data_mode))
    # todo
#    torch.cuda.empty_cache()

    dice_arrWMH = []
    PPV_arrWMH = []
    sensitivity_arrWMH = []
    specificity_arrWMH = []
    hd_arrWMH = []

    # paths = testset.paths
    sup_loss = torch.nn.CrossEntropyLoss()
    slices_per_domain = {
        '0': 48,
        '1': 48,
        '2': 83
    }
    model.to(device)
    with torch.no_grad():
        model.eval()

        x_all_test = None
        y_all_test = None
        b_all_test = None
        domain_all_test = None

        for i, batch in enumerate(val_loader):

            if x_all_test is None:
                x_all_test = batch['img']
                y_all_test = batch['label']
                b_all_test = batch['mask'].bool()
                domain_all_test = batch['domain_s']

                continue
            x_all_test = torch.cat((x_all_test, batch['img']), dim=0)
            y_all_test = torch.cat((y_all_test, batch['label']), dim=0)
            b_all_test = torch.cat((b_all_test, batch['mask'].bool()),
                                   dim=0)
            domain_all_test = torch.cat((domain_all_test, batch['domain_s']), dim=0)

        first = 0
        while first < len(val_loader):
            torch.cuda.empty_cache()
            step = slices_per_domain[str(domain_all_test[first].item())]
            last = first + step
            x_subject = x_all_test[first:last, :, :, :].to(device)
            y_subject = y_all_test[first:last, :, :, :]
            y_subject[y_subject != 1] = 0
            brain_mask = b_all_test[first:last, :, :, :]
            #first = first + step
            yhat_subject, _ = model(x_subject, True)
<<<<<<< HEAD

            x_subject = x_subject.to('cpu')
         #   y_subject = y_all_test[first:last, :, :, :].to('cpu')
         #   y_subject = y_all_test[first:last, :, :, :].to('cpu')
        
            brain_mask = brain_mask.to('cpu')
            y_subject = y_subject.to(device)
=======
            x_subject = x_subject.to('cpu')
            y_subject = y_all_test[first:last, :, :, :].to('cpu')
            y_subject = y_all_test[first:last, :, :, :].to('cpu')
>>>>>>> 608e686a84ebc9a30aa48737d34799ae71b41288
            loss_val = compute_loss(yhat_subject, y_subject, (sup_loss, None), is_supervised=True, cfg=cfg)
            first = first + step
            print("############# LOSS for subject is {} ##############".format( loss_val.item()))

            sf = torch.nn.Softmax2d()
            y_subject = y_subject.clone()
            y_subject[y_subject >= 1] = 1
            y_pred = sf(yhat_subject[-1])
            y_WT = seg2WT(y_pred, threshold, oneHot=cfg.oneHot)
            brain_mask = brain_mask.reshape(y_WT.shape)
            y_subject = y_subject.reshape(y_WT.shape)
            y_WT = y_WT[brain_mask]
            y_subject = y_subject[brain_mask]
            metrics_WMH = do_eval(y_subject, y_WT)
            print(
                "(WMH) :  DICE SCORE   {}, PPV  {},  Sensitivity: {}, Specificity: {}, Hausdorff: {}".format(
                    metrics_WMH['dsc'], metrics_WMH['ppv'],
                    metrics_WMH['sens'],
                    metrics_WMH['spec'], metrics_WMH['hd']))

            dice_arrWMH.append(metrics_WMH['dsc'].item())

            PPV_arrWMH.append(metrics_WMH['ppv'].item())

            sensitivity_arrWMH.append(metrics_WMH['sens'].item())

            specificity_arrWMH.append(metrics_WMH['spec'].item())
            hd_arrWMH.append(metrics_WMH['hd'])

    final_dice = {'WMH': np.mean(dice_arrWMH)}
    final_PPV = {'WMH': np.mean(PPV_arrWMH)}
    final_sensitivity = {'WMH': np.mean(sensitivity_arrWMH)}

    final_specificity = {'WMH': np.mean(specificity_arrWMH)}
    final_hd = {'WMH': np.mean(hd_arrWMH)}
    return final_dice, final_PPV, final_sensitivity, final_specificity, final_hd
    # return np.mean(np.array(dice_arrWT)), np.mean(np.array(dice_arrET)), np.mean(np.array(dice_arrTC))


def Pgs_train_val(dataset, n_epochs, wmh_threshold, output_dir, args, cfg, seed):
    inputs_dim = [2, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 2]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    wandb.run.name = "{}_PGS_{}_{}_supRate{}_seed{}_".format(cfg.experiment_mode, "trainALL2018", cfg.val_mode,
                                                             cfg.train_sup_rate, seed)
    # todo
    '''  uncomment this when you want to create a new split
    dataroot_dir = f'data/brats20'
    all_train_csv = os.path.join(dataroot_dir, 'trainset/brats2018.csv')
    supdir_path = os.path.join(dataroot_dir, 'trainset')

     semi_sup_split(all_train_csv=all_train_csv, sup_dir_path=supdir_path, unsup_dir_path=supdir_path,
                    ratio=cfg.train_sup_rate / 100, seed=cfg.seed)
    '''

    best_score = 0
    start_epoch = 0

    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    if cfg.experiment_mode == 'semi' or cfg.experiment_mode == 'partially_sup_upSample' or cfg.experiment_mode == 'semi_downSample' or cfg.experiment_mode == 'semi_alternate':
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
    
    
    output_dir = os.path.join(output_dir, "temp_{}".format(cfg.temp))
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

    # log_file_path = os.path.join(output_dir, "log_{}.log".format(args.training_mode))
    # sys.stdout = open(log_file_path, "w")
    print("EXPERIMENT DESCRIPTION:   {}".format(cfg.information))

    print("******* TRAINING PGS ***********")
    print("sup learning_rate is    ", cfg.supervised_training.lr)
    print("unsup learning_rate is    ", cfg.unsupervised_training.lr)
    # print("scheduler step size is :   ", step_size)
    print("output_dir is    ", output_dir)

    pgsnet = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)

    if torch.cuda.is_available():
        if type(pgsnet) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            pgsnet = torch.nn.DataParallel(pgsnet)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)
    pgsnet.to(device)
    # todo
<<<<<<< HEAD
  #  cfg.experiment_mode = 'partially_sup'
=======

>>>>>>> 608e686a84ebc9a30aa48737d34799ae71b41288
    splits, num_domains = get_splits(
        'WMH_SEG',  # get data of different domains
        T1=cfg.t1,
        whitestripe=False,
        supRatio=cfg.train_sup_rate,
        seed=cfg.seed,
        experiment_mode=cfg.experiment_mode)

    trainset_sup = splits['train_sup']()

    if splits['train_unsup'] is not None:
        trainset_unsup = splits['train_unsup']()
    else:
        trainset_unsup = None

    valset = splits['val']()

    trainset_sup = WmhChallenge(trainset_sup,
                                base_and_aug=False,
                                do_aug=True
                                )
    if trainset_unsup is not None:
        trainset_unsup = WmhChallenge(trainset_unsup,
                                      base_and_aug=False,
                                      do_aug=True
                                      )
    valset = WmhChallenge(valset,
                          base_and_aug=False,
                          do_aug=False
                          )

    if cfg.experiment_mode in ['semi_alternate']:
        train_sup_loader = torch.utils.data.DataLoader(trainset_sup,
                                                       batch_size=cfg.batch_size,
                                                       drop_last=True,
                                                       num_workers=4,
                                                       shuffle=True)

        train_unsup_loader = torch.utils.data.DataLoader(trainset_unsup,
                                                         batch_size=cfg.batch_size,
                                                         drop_last=True,
                                                         num_workers=4,
                                                         shuffle=True)

    elif cfg.experiment_mode in ['partially_sup', 'fully_sup']:

        train_sup_loader = torch.utils.data.DataLoader(trainset_sup,
                                                       batch_size=cfg.batch_size,
                                                       drop_last=True,
                                                       num_workers=4,
                                                       shuffle=True)

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=1,
                                             drop_last=False,
                                             num_workers=4,
                                             shuffle=False)

<<<<<<< HEAD
   # val_final_dice, val_final_PPV, val_final_sensitivity, val_final_specificity, val_final_hd = eval_per_subjectPgs(
    #    pgsnet, device,
     #   wmh_threshold,
      #  cfg,
       # cfg.val_mode,
       # val_loader)
=======

>>>>>>> 608e686a84ebc9a30aa48737d34799ae71b41288

    optimizer_unsup = torch.optim.SGD(pgsnet.parameters(), cfg.unsupervised_training.lr, momentum=0.9,
                                      weight_decay=1e-4)
    optimizer_sup = torch.optim.SGD(pgsnet.parameters(), cfg.supervised_training.lr, momentum=0.9,
                                     weight_decay=1e-4)

    #optimizer_sup = torch.optim.Adam(pgsnet.parameters(), cfg.supervised_training.lr, weight_decay=5e-4)
    optimizer_unsup = torch.optim.SGD(pgsnet.parameters(), cfg.unsupervised_training.lr, momentum=0.9,
                                      weight_decay=1e-4)
    #optimizer_unsup = torch.optim.Adam(pgsnet.parameters(), cfg.unsupervised_training.lr,weight_decay=1e-4)



    scheduler_unsup = lr_scheduler.StepLR(optimizer_unsup,
                                          step_size=cfg.unsupervised_training.scheduler_step_size,
                                          gamma=cfg.unsupervised_training.lr_gamma)
    scheduler_sup = lr_scheduler.StepLR(optimizer_sup, step_size=cfg.supervised_training.scheduler_step_size,
                                        gamma=cfg.supervised_training.lr_gamma)

    cons_w_unsup = consistency_weight(final_w=cfg['unsupervised_training']['consist_w_unsup']['final_w'],
                                      iters_per_epoch=len(train_sup_loader),
                                      rampup_ends=cfg['unsupervised_training']['consist_w_unsup'][
                                          'rampup_ends'],
                                      ramp_type=cfg['unsupervised_training']['consist_w_unsup']['rampup'])

    for epoch in range(start_epoch, n_epochs):
        print("iteration:  ", epoch)

        # pgsnet, loss = trainPGS(train_loader, pgsnet, optimizer, device, epoch)

        if cfg.experiment_mode == 'semi_alternate':

            pgsnet, loss = trainPgs_semi_alternate(train_sup_loader, train_unsup_loader, pgsnet,
                                                   (optimizer_sup, optimizer_unsup), device,
                                                   (torch.nn.CrossEntropyLoss(), None),
                                                   cons_w_unsup,
                                                   epoch, cfg)
        elif cfg.experiment_mode == 'partially_sup':

            pgsnet, loss = trainPgs_sup(train_sup_loader, pgsnet, optimizer_sup, device,
                                        (torch.nn.CrossEntropyLoss(), None),
                                        epoch, cfg)
        elif cfg.experiment_mode == 'fully_sup':
            pgsnet, loss = trainPgs_sup(train_sup_loader, pgsnet, optimizer_sup, device,
                                        (torch.nn.CrossEntropyLoss(), None),
                                        epoch, cfg)

        if epoch % 2 == 0:  # todo
            val_final_dice, val_final_PPV, val_final_sensitivity, val_final_specificity, val_final_hd = eval_per_subjectPgs(
                pgsnet, device,
                wmh_threshold,
                cfg,
                cfg.val_mode,
                val_loader)
            #
            if val_final_dice['WMH'] > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch,
                                                                                                     val_final_dice[
                                                                                                         'WMH']))
                best_score = val_final_dice['WMH']
                path = os.path.join(output_model_dir, 'pgsnet_best.model')
                with open(path, 'wb') as f:
                    torch.save(pgsnet, f)

            save_score_all(output_image_dir,
                           (val_final_dice, val_final_hd, val_final_PPV, val_final_sensitivity, val_final_specificity),
                           epoch, mode=cfg.val_mode)

            wandb.log({'epoch_id': epoch,
                       'val_WMH_subject_wise_DSC': val_final_dice['WMH'],
                       'val_WMH_subject_wise_HD': val_final_hd['WMH'],
                       'val_WMH_subject_wise_PPV': val_final_PPV['WMH'],
                       'val_WMH_subject_wise_SENSITIVITY': val_final_sensitivity['WMH'],
                       'val_WMH_subject_wise_SPECIFCITY': val_final_specificity['WMH'],
                       })

        if cfg.experiment_mode == 'semi_alternate':
            scheduler_sup.step()
            scheduler_unsup.step()
        else:
            scheduler_sup.step()

    final_dice, final_PPV, final_sensitivity, final_specificity, final_hd = eval_per_subjectPgs(pgsnet, device,
                                                                                                wmh_threshold,
                                                                                                cfg,
                                                                                                cfg.val_mode, val_loader)
    print(
        "** (WMH) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, PPV:{}, Sensitivity: {}, Specificity: {}  **".
            format(cfg.n_epochs, final_dice['WMH'], final_hd['WMH'], final_PPV['WMH'], final_sensitivity['WMH'],
                   final_specificity['WMH']))

    wandb.log({'epoch_id': cfg.n_epochs,
               'val_WMH_subject_wise_DSC': final_dice['WMH'],
               'val_WMH_subject_wise_HD': final_hd['WMH'],
               'val_WMH_subject_wise_PPV': final_PPV['WMH'],
               'val_WMH_subject_wise_SENSITIVITY': final_sensitivity['WMH'],
               'val_WMH_subject_wise_SPECIFCITY': final_specificity['WMH'],
               })
    save_score_all(output_image_dir, (final_dice, final_hd, final_PPV, final_sensitivity, final_specificity),
                   cfg.n_epochs, mode=cfg.val_mode)


def save_score(dir_path, score, iter):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average WMH dice score per subject at iter {}  :   {}\n".format(iter, score['WMH']))


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
                " **WMH**  DICE: {}, PPV:{}, Sensitivity: {}, hd: {}, specificity: {}\n"
            .format(
            iter, final_dice['WMH'], final_PPV['WMH'], final_sensitivity['WMH'], final_hd['WMH'],
            final_specificity['WMH']))


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
    # todo
    parser.add_argument(
        "--config",
        type=str,
        default='PGS_config_WMH.yaml'
    )

    parser.add_argument(
        "--training_mode",
        default="semi_sup",
        type=str,
        help="training mode supervised (sup), n subject supervised (n_sup), all supervised (all_sup)"
    )

    dataset = utils.Constants.Datasets.Wmh_challenge
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    # ANTHONY COMMENTED THIS OUT 11-13-2021 at 10:10AM
    # GOING TO SET IT MANUALLY OUTSIDE OF SCRIPT
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.cuda


    config_params = dict(args=args, config=cfg)
    wandb.init(project="CVPR2022_WMHChallenge", config=config_params)
    Pgs_train_val(dataset, cfg.n_epochs, cfg.wmh_threshold, args.output_dir, args, cfg, cfg.seed)


if __name__ == '__main__':
    main()
