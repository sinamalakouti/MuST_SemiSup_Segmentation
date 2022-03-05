import os
import sys
import datetime
import yaml
import matplotlib.pyplot as plt
import random
import argparse
from easydict import EasyDict as edict
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

from dataset.wmh_utils import get_splits
from dataset.Brat20 import seg2WT
from dataset.wmhChallenge import WmhChallenge
from utils import utils

from losses.evaluation_metrics import dice_coef, do_eval
from models import Pgs

from torch.distributions.normal import Normal
from utils.model_utils import update_adaptiveRate, ema_update, copy_params

for p in sys.path:
    print("path  ", p)
# random_seeds = [41, 42, 43]

utils.Constants.USE_CUDA = True
parser = argparse.ArgumentParser()


# TODO : CHECK IF THE IMPLEMNTATION IS CORRECT
@torch.no_grad()
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
            # first = first + step
            yhat_subject, _ = model(x_subject, True)
            x_subject = x_subject.to('cpu')
            #   y_subject = y_all_test[first:last, :, :, :].to('cpu')
            #   y_subject = y_all_test[first:last, :, :, :].to('cpu')

            brain_mask = brain_mask.to('cpu')
            y_subject = y_subject.to(device)

            x_subject = x_subject.to('cpu')
            y_subject = y_all_test[first:last, :, :, :].to('cpu')
            y_subject = y_all_test[first:last, :, :, :].to('cpu')

            loss_val = compute_loss(yhat_subject, y_subject, (sup_loss, None), is_supervised=True, cfg=cfg)
            first = first + step
            print("############# LOSS for subject is {} ##############".format(loss_val.item()))

            sf = torch.nn.Softmax2d()
            y_subject = y_subject.clone()
            y_subject[y_subject >= 1] = 1
            y_pred = sf(yhat_subject[-1])
            y_WT = seg2WT(y_pred, threshold, oneHot=cfg.oneHot)
            brain_mask = brain_mask.reshape(y_WT.shape)
            y_subject = y_subject.reshape(y_WT.shape)
            y_WT = y_WT[brain_mask]
            y_subject = y_subject[brain_mask].bool()
            metrics_WMH = do_eval(y_subject.to('cpu'), y_WT.to('cpu'))
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


def brats(cfg, model_path, result_path):
    inputs_dim = [2, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 4]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    pgsnet = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)
    # load_model


def wmh_dataset(cfg, model_path, result_path):
    inputs_dim = [2, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 2]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    model = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)

    # load model

    if torch.cuda.is_available():
        if type(model) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            model = torch.nn.DataParallel(model)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    model.to(device)

    # loading datasets
    splits, num_domains = get_splits(
        'WMH_SEG',  # get data of different domains
        T1=cfg.t1,
        whitestripe=False,
        supRatio=cfg.train_sup_rate,
        seed=cfg.seed,
        experiment_mode=cfg.experiment_mode)
    valset = splits['val']()
    valset = WmhChallenge(valset,
                          base_and_aug=False,
                          do_aug=False
                          )
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=1,
                                             drop_last=False,
                                             num_workers=4,
                                             shuffle=False)    
    slices_per_domain = {
        '0': 48,
        '1': 48,
        '2': 83
    }
    model.to(device)
    subject = 0
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
            # first = first + step
            yhat_subject, _ = model(x_subject, True)
            x_subject = x_subject.to('cpu')
            brain_mask = brain_mask.to('cpu')

            x_subject = x_subject.to('cpu')
            y_subject = y_all_test[first:last, :, :, :].to('cpu')

            first = first + step

            sf = torch.nn.Softmax2d()
            y_subject = y_subject.clone()
            y_subject[y_subject >= 1] = 1
            y_pred = sf(yhat_subject[-1])
            y_WT = seg2WT(y_pred, 0.5, oneHot=cfg.oneHot)
            brain_mask = brain_mask.reshape(y_WT.shape)
            y_subject = y_subject.reshape(y_WT.shape)
            y_WT = y_WT[brain_mask]
            y_subject = y_subject[brain_mask].bool()

            metrics_WMH = do_eval(y_subject.to('cpu'), y_WT.to('cpu'))

            dir_path = os.path.join(result_path, 'subject_{}'.format(subject))

            if not os.path.isdir(dir_path):
                try:
                    os.mkdir(dir_path, 0o777)
                except OSError:
                    print("Creation of the directory %s failed" % dir_path)

            true_path = os.path.join(dir_path, 'ture_images')
            if not os.path.isdir(true_path):
                try:
                    os.mkdir(true_path, 0o777)
                except OSError:
                    print("Creation of the directory %s failed" % dir_path)

            pred_path = os.path.join(dir_path, 'pred_images')
            if not os.path.isdir(pred_path):
                try:
                    os.mkdir(pred_path, 0o777)
                except OSError:
                    print("Creation of the directory %s failed" % dir_path)

            img_path = os.path.join(dir_path, 'flair')
            for i in range(0, len(y_subject)):
                true = y_subject[i]
                pred = y_WT[i]
                plt.axis('off')
                plt.imshow(true)
                plt.savefig(os.path.join(true_path, "true_{}.png".format(i)))
                plt.axis('off')
                plt.imshow(pred)
                plt.savefig(os.path.join(pred_path, "pred{}.png".format(i)))

                plt.axis('off')
                plt.imshow(x_subject[i][0])
                plt.savefig(os.path.join(img_path, "inptu{}.png".format(i)))

            print(
                "(WMH) :  DICE SCORE   {}, PPV  {},  Sensitivity: {}, Specificity: {}, Hausdorff: {}".format(
                    metrics_WMH['dsc'], metrics_WMH['ppv'],
                    metrics_WMH['sens'],
                    metrics_WMH['spec'], metrics_WMH['hd']))

            with open(dir_path, "w") as f:
                f.write("SCORE for subejct {}:\n"
                        " **WMH**  DICE: {}".
                        format(subject, metrics_WMH['dsc']))

            subject += 1


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
    # todo
    parser.add_argument(
        "--config",
        type=str,
        default='None'
    )

    dataset = utils.Constants.Datasets.Wmh_challenge
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    model_path = None
    result_path = None
    wmh_dataset(cfg, model_path, result_path)


if __name__ == '__main__':
    main()
