import os
import sys
import datetime
import yaml
import matplotlib.pyplot as plt
import random
import argparse
from easydict import EasyDict as edict
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

from dataset.wmh_utils import get_splits
from dataset.wmhChallenge import WmhChallenge

from losses.loss import consistency_weight
from losses.evaluation_metrics import dice_coef, do_eval
from models import Pgs_WMH as Pgs
from challenge_evaluation.evalutation import do_eval_train
from utils import utils

import wandb

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('src/utils')

for p in sys.path:
    print("path  ", p)

utils.Constants.USE_CUDA = True
parser = argparse.ArgumentParser()


def seg2WT(preds, threshold, oneHot=False):
    if oneHot:
        preds = preds >= threshold
        WT_pred = preds[:, 1:4, :, :].sum(1) >= 1
    else:
        max_val, max_indx = torch.max(preds, dim=1)
        max_val = (max_val >= threshold).float()
        max_indx[max_indx >= 1] = 1
        WT_pred = torch.multiply(max_indx, max_val)

    # WT_pred = preds[:, 1:4, :, :].sum(1) >= 1
    return WT_pred


def __fw_unsup_loss(y_stud, y_teach, loss_functions, cfg, mask=None):
    return __fw_outputwise_unsup_loss(y_stud, y_teach, loss_functions, cfg, mask)


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

        if cfg.unsupervised_training.T is not None:  # sharpening
            with torch.no_grad():
                pt = teach_pred ** (1 / cfg.unsupervised_training.T)
                teach_pred = pt / pt.sum(dim=1, keepdim=True)
                if teach_pred.isnan().sum() > 0:
                    teach_pred[teach_pred.isnan()] = 0
        mse = torch.nn.MSELoss()
        stud_pred = torch.nn.functional.softmax(stud_pred, dim=1)
        loss = mse(stud_pred, teach_pred)
        losses.append(loss)
    total_loss = sum(losses)
    return total_loss


def __fw_sup_loss(y_preds, y_true, sup_loss):
    # iterate over all level's output
    losses = []
    target_fg = y_true
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
    else:
        total_loss = __fw_unsup_loss(y_preds, y_true, loss_functions, cfg, masks)
    return total_loss


def trainPgs_challenge(train_sup_loader, model, optimizer, device, loss_functions,
                       cons_w_unsup, epochid,
                       cfg):
    model.train()

    for batch_idx,batch in enumerate(train_sup_loader):
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        data = batch['img']
        target = batch['label'].to(device)
        data = data.to(device)

        # supervised training - forward
        sup_outputs, _ = model(data, is_supervised=True)
        sLoss = compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True, cfg=cfg)
        # supervised training - backward
        sLoss.backward()
        optimizer[0].step()

        # UNSUPERVISED TRAINING

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        # unsupervised training - forward
        teacher_outputs, student_outputs = model(data, is_supervised=False)
        uLoss = compute_loss(student_outputs, teacher_outputs, loss_functions, is_supervised=False, cfg=cfg)
        # unsupervised training - backward
        weight_unsup = cons_w_unsup(epochid, batch_idx)
        uloss_final = uLoss * weight_unsup
        uloss_final.backward()
        optimizer[1].step()

        print("**************** UNSUP LOSSS  : {} ****************".format(uLoss))
        print("**************** SUP LOSSS  : {} ****************".format(sLoss))

        with torch.no_grad():
            sf = torch.nn.Softmax2d()

            target_sup[target_sup == 1] = 1
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

@torch.no_grad()
def eval_per_subjectPgs(model, device, threshold, cfg, data_mode, val_loader):
    print("******************** EVALUATING {}********************".format(data_mode))
    dice_arrWMH = []
    f1_arrWMH = []
    sensitivity_arrWMH = []
    avd_arrWMH = []
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

            step = slices_per_domain[str(domain_all_test[first].item())]
            last = first + step
            x_subject = x_all_test[first:last, :, :, :].to(device)
            brain_mask = b_all_test[first:last, :, :, :]
            # first = first + step
            yhat_subject, _ = model(x_subject, True)
            brain_mask = brain_mask.to('cpu')
            y_subject = y_all_test[first:last, :, :, :].to('cpu')

            loss_val = compute_loss(yhat_subject, y_subject, (sup_loss, None), is_supervised=True, cfg=cfg)
            first = first + step
            print("############# LOSS for subject is {} ##############".format(loss_val.item()))

            sf = torch.nn.Softmax2d()
            y_subject = y_subject.clone()

            y_pred = sf(yhat_subject[-1])
            y_WT = y_pred
            # y_WT = seg2WT(y_pred, threshold, oneHot=cfg.oneHot)

            brain_mask = brain_mask.reshape(y_WT.shape)
            y_subject = y_subject.reshape(y_WT.shape)
            y_WT = y_WT[brain_mask]
            y_subject = y_subject[brain_mask].bool()
            dsc, h95, avd, recall, f1 = do_eval_train(y_subject.to('cpu'), y_WT.to('cpu'))
            # metric['dsc']=  dsc, h95, avd, recall, f1
            metrics_WMH = {}
            metrics_WMH['dsc'] = dsc
            metrics_WMH['hd'] = h95
            metrics_WMH['sens'] = recall
            metrics_WMH['f1'] = f1
            metrics_WMH['avd'] = avd

            print(
                "(WMH) :  DICE SCORE   {}, f1  {},  Sensitivity: {}, avd: {}, Hausdorff: {}".format(
                    metrics_WMH['dsc'], metrics_WMH['f1'],
                    metrics_WMH['sens'],
                    metrics_WMH['avd'], metrics_WMH['hd']))

            dice_arrWMH.append(metrics_WMH['dsc'].item())

            f1_arrWMH.append(metrics_WMH['f1'].item())

            sensitivity_arrWMH.append(metrics_WMH['sens'].item())

            avd_arrWMH.append(metrics_WMH['avd'].item())
            hd_arrWMH.append(metrics_WMH['hd'])

    final_dice = {'WMH': np.mean(dice_arrWMH)}
    final_f1 = {'WMH': np.mean(f1_arrWMH)}
    final_sensitivity = {'WMH': np.mean(sensitivity_arrWMH)}

    final_avd = {'WMH': np.mean(avd_arrWMH)}
    final_hd = {'WMH': np.mean(hd_arrWMH)}
    return final_dice, final_f1, final_sensitivity, final_avd, final_hd
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
    if 'semi' in cfg.experiment_mode:
        output_dir = os.path.join(output_dir, str(cfg.unsupervised_training.consistency_training_method))
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
        output_dir = os.path.join(output_dir, "fullySup")
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


def get_train_loaders(dataset, cfg):
    None


def Pgs_train_val(dataset, n_epochs, wmh_threshold, output_dir, args, cfg, seed):
    inputs_dim = [2, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 2]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    wandb.run.name = "{}_{}_{}_{}_supRate{}_seed{}_".format(cfg.experiment_mode, cfg.model, "trainALL2018",
                                                            cfg.val_mode,
                                                            cfg.train_sup_rate, seed)

    best_score = 0
    start_epoch = 0
    print('-' * 50)
    print("EXPERIMENT-   PGS on WMH challenge")
    print('-' * 50)

    output_model_dir, output_image_dir = initialize_directories(output_dir, seed, cfg)

    print("EXPERIMENT DESCRIPTION:   {}".format(cfg.information))

    print("******* TRAINING PGS ***********")
    print("sup learning_rate is    ", cfg.supervised_training.lr)
    print("unsup learning_rate is    ", cfg.unsupervised_training.lr)
    # print("scheduler step size is :   ", step_size)
    print("output_dir is    ", output_dir)

    # load model

    # if cfg.model == 'PGS':
    pgsnet = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)

    if torch.cuda.is_available():
        if type(pgsnet) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            pgsnet = torch.nn.DataParallel(pgsnet)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    pgsnet.to(device)

    # loading datasets
    splits, num_domains = get_splits(
        'WMH_SEG',  # get data of different domains
        T1=cfg.t1,
        whitestripe=False,
        supRatio=cfg.train_sup_rate,
        seed=cfg.seed,
        experiment_mode=cfg.experiment_mode)

    trainset_sup = splits['train_sup']()
    valset = splits['val']()

    trainset_sup = WmhChallenge(trainset_sup,
                                base_and_aug=False,
                                do_aug=True
                                )
    valset = WmhChallenge(valset,
                          base_and_aug=False,
                          do_aug=False
                          )
    # loading data loaders

    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=1,
                                             drop_last=False,
                                             num_workers=4,
                                             shuffle=False)

    train_sup_loader = torch.utils.data.DataLoader(trainset_sup,
                                                   batch_size=cfg.batch_size,
                                                   drop_last=True,
                                                   num_workers=4,
                                                   shuffle=True)

    # load optimizers & schedulers
    optimizer_unsup = torch.optim.SGD(pgsnet.parameters(), cfg.unsupervised_training.lr, momentum=0.9,
                                      weight_decay=1e-4)
    scheduler_unsup = lr_scheduler.StepLR(optimizer_unsup,
                                          step_size=cfg.unsupervised_training.scheduler_step_size,
                                          gamma=cfg.unsupervised_training.lr_gamma)

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

    for epoch in range(start_epoch, n_epochs):
        print("iteration:  ", epoch)

        trainPgs_challenge(train_sup_loader, pgsnet,
                           (optimizer_sup, optimizer_unsup), device,
                           (torch.nn.CrossEntropyLoss(), None),
                           cons_w_unsup,
                           epoch, cfg)

        if epoch % 2 == 0:
            # TODO -> COMPATIBLE WIth CHALLENGE
            val_final_dice, val_final_f1, val_final_sensitivity, val_final_avd, val_final_hd = \
                eval_per_subjectPgs(
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
                           (val_final_dice, val_final_hd, val_final_f1, val_final_sensitivity, val_final_avd),
                           epoch, mode=cfg.val_mode)

            wandb.log({'epoch_id': epoch,
                       'val_WMH_subject_wise_DSC': val_final_dice['WMH'],
                       'val_WMH_subject_wise_HD': val_final_hd['WMH'],
                       'val_WMH_subject_wise_f1': val_final_f1['WMH'],
                       'val_WMH_subject_wise_SENSITIVITY': val_final_sensitivity['WMH'],
                       'val_WMH_subject_wise_avd': val_final_avd['WMH'],
                       })

        if scheduler_unsup is not None:
            scheduler_sup.step()
            scheduler_unsup.step()
        else:
            scheduler_sup.step()

    final_dice, final_f1, final_sensitivity, final_avd, final_hd = eval_per_subjectPgs(pgsnet, device,
                                                                                                wmh_threshold,
                                                                                                cfg,
                                                                                                cfg.val_mode,
                                                                                                val_loader)
    print(
        "** (WMH) SUBJECT WISE SCORE @ Iteration {} is DICE: {}, HD: {}, f1:{}, Sensitivity: {}, avd: {}  **".
            format(cfg.n_epochs, final_dice['WMH'], final_hd['WMH'], final_f1['WMH'], final_sensitivity['WMH'],
                   final_avd['WMH']))

    wandb.log({'epoch_id': cfg.n_epochs,
               'val_WMH_subject_wise_DSC': final_dice['WMH'],
               'val_WMH_subject_wise_HD': final_hd['WMH'],
               'val_WMH_subject_wise_f1': final_f1['WMH'],
               'val_WMH_subject_wise_SENSITIVITY': final_sensitivity['WMH'],
               'val_WMH_subject_wise_avd': final_avd['WMH'],
               })
    save_score_all(output_image_dir, (final_dice, final_hd, final_f1, final_sensitivity, final_avd),
                   cfg.n_epochs, mode=cfg.val_mode)


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
        f.write("average WMH dice score per subject at iter {}  :   {}\n".format(iteration, score['WMH']))


@torch.no_grad()
def save_score_all(dir_path, scores, iteration, mode):  # mode = 2020 test or 2019 test
    final_dice = scores[0]
    final_hd = scores[1]
    final_f1 = scores[2]
    final_sensitivity = scores[3]
    final_avd = scores[4]

    dir_path = os.path.join(dir_path, "{}_results_iter{}".format(mode, iteration))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("AVG SCORES PER  SUBJECTS AT ITERATION {}:\n"
                " **WMH**  DICE: {},f1:{}, Sensitivity: {},"
                " hd: {}, avd: {}\n".format(iteration, final_dice['WMH'], final_f1['WMH'],
                                                    final_sensitivity['WMH'], final_hd['WMH'],
                                                    final_avd['WMH']))


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
        "--config",
        type=str,
        default='PGS_config_WMH.yaml'
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default="MICCAI2022_challengeSubmission"
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=True
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

    config_params = dict(args=args, config=cfg)
    wandb.init(project=args.wandb, config=config_params)
    Pgs_train_val(dataset, cfg.n_epochs, cfg.wmh_threshold, args.output_dir, args, cfg, cfg.seed)


if __name__ == '__main__':
    main()
