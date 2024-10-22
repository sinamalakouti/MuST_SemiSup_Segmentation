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
from dataset.Brat20 import Brat20Test, seg2WT, seg2TC, seg2ET

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


def plot_slice(cfg, model_path_sup, model_path_semi, result_path, sujbect_id, slice_num):
    inputs_dim = [2, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 4]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    sup_model = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)
    semi_model = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)
    # load_model

    sup_model = torch.load(model_path_sup)
    semi_model = torch.load(model_path_semi)

    if torch.cuda.is_available():
        if type(sup_model) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            sup_model = torch.nn.DataParallel(sup_model)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)
    sup_model.to(device)

    if torch.cuda.is_available():
        if type(semi_model) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            semi_model = torch.nn.DataParallel(semi_model)
            device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)
    semi_model.to(device)

    testset = Brat20Test(f'data/brats20', 'test2020_new', 10, 155,
                         augment=False, center_cropping=True, t1=True, t2=True, t1ce=True, oneHot=cfg.oneHot)
    paths = testset.paths
    threshold = 0.5

    most_gap_score = -1
    most_gap_index = -1
    sup_final_preds = None
    semi_final_preds = None
    final_inputs = None
    final_target = None
    with torch.no_grad():
        for path in paths:
            batch = testset.get_subject(path)
            b = batch['data']
            target = batch['label'].to(device)
            subjects = batch['subjects']
            if subjects[0] != sujbect_id:
                continue
            assert len(np.unique(subjects)) == 1, print("More than one subject at a time")
            b = b.to(device)
            outputs_sup, _ = sup_model(b, True)
            outputs_semi, _ = semi_model(b, True)

            sf = torch.nn.Softmax2d()
            pred_sup = sf(outputs_sup[-1])
            pred_semi = sf(outputs_semi[-1])

            print("*** EVALUATION METRICS FOR SUBJECT {} IS: ".format(subjects[0]))

        dir_path = os.path.join(result_path, 'subject_{}'.format(most_gap_index))

        if not os.path.isdir(dir_path):
            try:
                os.mkdir(dir_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        true_path = os.path.join(dir_path, 'true_images')
        if not os.path.isdir(true_path):
            try:
                os.mkdir(true_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        sup_pred_path = os.path.join(dir_path, 'sup_pred_images')
        semi_pred_path = os.path.join(dir_path, 'semi_pred_images')
        if not os.path.isdir(sup_pred_path):
            os.mkdir(sup_pred_path, 0o777)
        if not os.path.isdir(semi_pred_path):
            os.mkdir(semi_pred_path, 0o777)

        img_path = os.path.join(dir_path, 'input_images')
        if not os.path.isdir(img_path):
            os.mkdir(img_path, 0o777)
        print("CREATING PLOTS FOR subject : ", most_gap_index)

        log = os.path.join(dir_path, 'metrics.txt')
        with open(log, "w") as f:
            f.write("AVG GAP for subject {}:\n"
                    " DICE: {}".
                    format(most_gap_index, most_gap_score))

        from matplotlib.colors import ListedColormap
        from skimage.color import label2rgb

        ls = ListedColormap(['black', 'g', 'b', 'yellow'])
        input = b[slice_num].cpu()  #
        y_true = target[slice_num].cpu()  # 0: bg, 1: NET  2: ED  3: ET
        seg_sup = pred_sup[slice_num].cpu()
        seg_semi = pred_semi[slice_num].cpu()

        # target

        gd_net = y_true[1, :, :]
        gd_ed = y_true[2, :, :]
        gd_et = y_true[3, :, :]
        gd = np.zeros(gd_net.shape)
        gd[gd_net == 1] = 1
        gd[gd_ed == 1] = 2
        gd[gd_et == 1] = 3

        gd_img = label2rgb(gd, input[0], colors=['r', 'b', 'g'], bg_label=0)

        # supervised-baseline

        sup_net = seg_sup[1, :, :] >= 0.5
        sup_ed = seg_sup[2, :, :] >= 0.5
        sup_et = seg_sup[3, :, :] >= 0.5

        sup_mix = np.zeros(sup_net.shape)
        sup_mix[sup_net == 1] = 1
        sup_mix[sup_ed == 1] = 2
        sup_mix[sup_et == 1] = 3

        sup_mix_img = label2rgb(sup_mix, input[0], colors=['r', 'b', 'g'], bg_label=0)

        # semi-supervised

        semi_net = seg_semi[1, :, :] >= 0.5
        semi_ed = seg_semi[2, :, :] >= 0.5
        semi_et = seg_semi[3, :, :] >= 0.5

        semi_mix = np.zeros(semi_net.shape)
        semi_mix[semi_net == 1] = 1
        semi_mix[semi_ed == 1] = 2
        semi_mix[semi_et == 1] = 3

        semi_mix_img = label2rgb(semi_mix, input[0], colors=['r', 'b', 'g'], bg_label=0)

        # plot mixes!
        plt.axis('off')
        plt.imshow(gd_img)
        plt.savefig(os.path.join(true_path, 'true_slice_{}.png'.format(slice_num)))

        plt.axis('off')
        plt.imshow(sup_mix_img)
        plt.savefig(os.path.join(sup_pred_path, 'sup_pred_slice_{}.png'.format(slice_num)))

        plt.axis('off')
        plt.imshow(semi_mix_img)
        plt.savefig(os.path.join(semi_pred_path, 'semi_pred_slice_{}.png'.format(slice_num)))


def plot_all_brats(cfg, model_path_sup, model_path_semi, result_path):
    inputs_dim = [2, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 4]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    sup_model = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)
    semi_model = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)
    # load_model

    sup_model = torch.load(model_path_sup)
    semi_model = torch.load(model_path_semi)

    if torch.cuda.is_available():
        if type(sup_model) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            sup_model = torch.nn.DataParallel(sup_model)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)
    sup_model.to(device)

    if torch.cuda.is_available():
        if type(semi_model) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
            semi_model = torch.nn.DataParallel(semi_model)
            device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)
    semi_model.to(device)

    testset = Brat20Test(f'data/brats20', 'test2020_new', 10, 155,
                         augment=False, center_cropping=True, t1=True, t2=True, t1ce=True, oneHot=cfg.oneHot)
    paths = testset.paths
    threshold = 0.5

    most_gap_score = -1
    most_gap_index = -1
    sup_final_preds = None
    semi_final_preds = None
    final_inputs = None
    final_target = None
    with torch.no_grad():
        for path in paths:
            batch = testset.get_subject(path)
            b = batch['data']
            target = batch['label'].to(device)
            subjects = batch['subjects']
            assert len(np.unique(subjects)) == 1, print("More than one subject at a time")
            b = b.to(device)
            outputs_sup, _ = sup_model(b, True)
            outputs_semi, _ = semi_model(b, True)

            sf = torch.nn.Softmax2d()
            targetWT = target.clone()
            targetET = target.clone()
            targetTC = target.clone()
            targetWT[targetWT >= 1] = 1
            targetET[~ (targetET == 3)] = 0
            targetET[(targetET == 3)] = 1
            targetTC[~ ((targetTC == 3) | (targetTC == 1))] = 0
            targetTC[(targetTC == 3) | (targetTC == 1)] = 1
            y_pred_sup = sf(outputs_sup[-1])
            y_pred_semi = sf(outputs_semi[-1])

            y_WT_sup = seg2WT(y_pred_sup, threshold, oneHot=cfg.oneHot)
            y_ET_sup = seg2ET(y_pred_sup, threshold)
            y_TC_sup = seg2TC(y_pred_sup, threshold)

            y_WT_semi = seg2WT(y_pred_semi, threshold, oneHot=cfg.oneHot)
            y_ET_semi = seg2ET(y_pred_semi, threshold)
            y_TC_semi = seg2TC(y_pred_semi, threshold)

            metrics_WT_sup = do_eval(targetWT.reshape(y_WT_sup.shape), y_WT_sup)
            metrics_ET_sup = do_eval(targetET.reshape(y_ET_sup.shape), y_ET_sup)
            metrics_TC_sup = do_eval(targetTC.reshape(y_TC_sup.shape), y_TC_sup)

            metrics_WT_semi = do_eval(targetWT.reshape(y_WT_semi.shape), y_WT_semi)
            metrics_ET_semi = do_eval(targetET.reshape(y_ET_semi.shape), y_ET_semi)
            metrics_TC_semi = do_eval(targetTC.reshape(y_TC_semi.shape), y_TC_semi)

            avg_score_semi = (metrics_WT_semi['dsc'] + metrics_ET_semi['dsc'] + metrics_TC_semi['dsc']) / 3
            avg_score_sup = (metrics_WT_sup['dsc'] + metrics_ET_sup['dsc'] + metrics_TC_sup['dsc']) / 3

            print("*** EVALUATION METRICS FOR SUBJECT {} IS: ".format(subjects[0]))
            gap_score = avg_score_semi - avg_score_sup
            if gap_score > most_gap_score:
                print("max is subject {}   gap {}".format(subjects[0], gap_score))
                most_gap_score = gap_score
                most_gap_index = subjects[0]
                sup_final_preds = (y_WT_sup, y_ET_sup, y_TC_sup)
                semi_final_preds = (y_WT_semi, y_ET_semi, y_TC_semi)

                final_inputs = b
                final_target = (targetWT, targetET, targetTC)

        dir_path = os.path.join(result_path, 'subject_{}'.format(most_gap_index))

        if not os.path.isdir(dir_path):
            try:
                os.mkdir(dir_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        true_path = os.path.join(dir_path, 'true_images')
        if not os.path.isdir(true_path):
            try:
                os.mkdir(true_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        sup_pred_path = os.path.join(dir_path, 'sup_pred_images')
        semi_pred_path = os.path.join(dir_path, 'semi_pred_images')
        if not os.path.isdir(sup_pred_path):
            os.mkdir(sup_pred_path, 0o777)
        if not os.path.isdir(semi_pred_path):
            os.mkdir(semi_pred_path, 0o777)

        img_path = os.path.join(dir_path, 'input_images')
        if not os.path.isdir(img_path):
            os.mkdir(img_path, 0o777)

        print("CREATING PLOTS FOR subject : ", most_gap_index)

        log = os.path.join(dir_path, 'metrics.txt')
        with open(log, "w") as f:
            f.write("AVG GAP for subject {}:\n"
                    " DICE: {}".
                    format(most_gap_index, most_gap_score))

        from matplotlib.colors import ListedColormap
        ls = ListedColormap(['black', 'g', 'b', 'yellow'])
        for i in range(0, len(final_inputs)):
            true_WT, true_ET, true_TC = (final_target[0][i], final_target[1][i], final_target[2][i])

            sup_pred_WT, sup_pred_ET, sup_pred_TC = (
                sup_final_preds[0][i], sup_final_preds[1][i], sup_final_preds[2][i])
            semi_pred_WT, semi_pred_ET, semi_pred_TC = (
                semi_final_preds[0][i], semi_final_preds[1][i], semi_final_preds[2][i])

            # plot true labels
            plt.axis('off')
            plt.imshow(true_WT.cpu(), cmap='gray')
            plt.savefig(os.path.join(true_path, "true_WT_{}.png".format(i)))
            plt.axis('off')
            plt.imshow(true_ET.cpu(), cmap='gray')
            plt.savefig(os.path.join(true_path, "true_ET_{}.png".format(i)))
            plt.axis('off')
            plt.imshow(true_TC.cpu(), cmap='gray')
            plt.savefig(os.path.join(true_path, "true_TC_{}.png".format(i)))

            true_mix = true_WT
            true_mix[true_TC == 1] = 2
            true_mix[true_ET == 1] = 3

            plt.axis('off')
            plt.imshow(true_mix.cpu(), cmap=ls)
            plt.savefig(os.path.join(true_path, "true_mix_{}.png".format(i)))
            # plot supervised predictions

            plt.axis('off')
            plt.imshow(sup_pred_WT.cpu(), cmap='gray')
            plt.savefig(os.path.join(sup_pred_path, "pred_WT_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(sup_pred_ET.cpu(), cmap='gray')
            plt.savefig(os.path.join(sup_pred_path, "pred_ET_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(sup_pred_TC.cpu(), cmap='gray')
            plt.savefig(os.path.join(sup_pred_path, "pred_TC_{}.png".format(i)))

            sup_pred_mix = sup_pred_WT
            sup_pred_mix[sup_pred_TC == 1] = 2
            sup_pred_mix[sup_pred_ET == 1] = 3

            plt.axis('off')
            plt.imshow(sup_pred_mix.cpu(), cmap=ls)
            plt.savefig(os.path.join(sup_pred_path, "sup_pred_mix_{}.png".format(i)))

            # plot semi predictions

            plt.axis('off')
            plt.imshow(semi_pred_WT.cpu(), cmap='gray')
            plt.savefig(os.path.join(semi_pred_path, "pred_WT_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(semi_pred_ET.cpu(), cmap='gray')
            plt.savefig(os.path.join(semi_pred_path, "pred_ET_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(semi_pred_TC.cpu(), cmap='gray')
            plt.savefig(os.path.join(semi_pred_path, "pred_TC_{}.png".format(i)))

            semi_pred_mix = semi_pred_WT
            semi_pred_mix[semi_pred_TC == 1] = 2
            semi_pred_mix[semi_pred_ET == 1] = 3

            plt.axis('off')
            plt.imshow(semi_pred_mix.cpu(), cmap=ls)
            plt.savefig(os.path.join(semi_pred_path, "semi_pred_mix_{}.png".format(i)))

            # plot inputs:  flair, T1, T2, t1ce
            plt.axis('off')
            plt.imshow(final_inputs[i][0].cpu(), cmap='gray')
            plt.savefig(os.path.join(img_path, "input_flair_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(final_inputs[i][1].cpu(), cmap='gray')
            plt.savefig(os.path.join(img_path, "input_t1_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(final_inputs[i][2].cpu(), cmap='gray')
            plt.savefig(os.path.join(img_path, "input_t2_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(final_inputs[i][3].cpu(), cmap='gray')
            plt.savefig(os.path.join(img_path, "input_t1ce_{}.png".format(i)))


def wmh_dataset(cfg, model_path_sup, model_path_semi, result_path):
    inputs_dim = [2, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 2]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    model_sup = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)
    model_semi = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides, cfg)

    # load model

    # if torch.cuda.is_available():
    #    if type(model) is not torch.nn.DataParallel and cfg.parallel and cfg.parallel:
    #       model = torch.nn.DataParallel(model)
    #   device = 'cuda'
    # elif not torch.cuda.is_available():
    #   device = 'cpu'
    device = 'cuda'

    device = torch.device(device)
    #   model.load_state_dict(torch.load(model_path))

    model_sup = torch.load(model_path_sup)
    model_semi = torch.load(model_path_semi)
    model_sup.to(device)
    model_semi.to(device)

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

    subject = 0

    most_gap_score = -1
    most_gap_index = -1
    final_x = -1
    final_true = -1
    final_sup_pred = -1
    final_semi_pred = -1
    with torch.no_grad():
        model_sup.eval()
        model_semi.eval()

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
            sup_yhat_subject, _ = model_sup(x_subject, True)
            semi_yhat_subject, _ = model_semi(x_subject, True)
            x_subject = x_subject.to('cpu')
            brain_mask = brain_mask.to('cpu')

            x_subject = x_subject.to('cpu')
            y_subject = y_all_test[first:last, :, :, :].to('cpu')

            first = first + step

            sf = torch.nn.Softmax2d()
            y_subject = y_subject.clone()
            y_subject[y_subject >= 1] = 1
            sup_y_pred = sf(sup_yhat_subject[-1])
            semi_y_pred = sf(semi_yhat_subject[-1])
            sup_y_WT = seg2WT(sup_y_pred, 0.5, oneHot=cfg.oneHot)
            semi_y_WT = seg2WT(semi_y_pred, 0.5, oneHot=cfg.oneHot)
            # brain_mask = brain_mask.reshape(y_WT.shape)
            y_subject = y_subject.reshape(sup_y_WT.shape)
            # y_WT = y_WT[brain_mask]
            # y_subject = y_subject[brain_mask].bool()

            sup_metrics_WMH = do_eval(y_subject.to('cpu'), sup_y_WT.to('cpu'))
            semi_metrics_WMH = do_eval(y_subject.to('cpu'), semi_y_WT.to('cpu'))
            diff = semi_metrics_WMH['dsc'] - sup_metrics_WMH['dsc']
            print(" now subject   ", subject)
            print('difff ', diff)
            if subject != 4 and diff > most_gap_score:
                print("bestttt")
                print("subject   ", subject)
                print(semi_metrics_WMH['dsc'])
                print(sup_metrics_WMH['dsc'])
                print(diff)
                most_gap_index = subject
                most_gap_score = diff
                final_x = x_subject
                final_true = y_subject
                final_sup_pred = sup_y_WT
                final_semi_pred = semi_y_WT

            # dir_path = os.path.join(result_path, 'subject_{}'.format(subject))
            # if not os.path.isdir(dir_path):
            #    try:
            #       os.mkdir(dir_path, 0o777)
            #  except OSError:
            #     print("Creation of the directory %s failed" % dir_path)

            # log = os.path.join(dir_path, 'metrics.txt')
            # with open(log, "w") as f:
            #    f.write("SCORE for subejct {}:\n"
            #          " **WMH**  DICE: {}".
            #           format(subject, sup_metrics_WMH['dsc']))

            subject += 1

        dir_path = os.path.join(result_path, 'subject_{}'.format(most_gap_index))

        if not os.path.isdir(dir_path):
            try:
                os.mkdir(dir_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        true_path = os.path.join(dir_path, 'true_images')
        if not os.path.isdir(true_path):
            try:
                os.mkdir(true_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        sup_pred_path = os.path.join(dir_path, 'sup_pred_images')
        if not os.path.isdir(sup_pred_path):
            try:
                os.mkdir(sup_pred_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        semi_pred_path = os.path.join(dir_path, 'semi_pred_images')
        if not os.path.isdir(semi_pred_path):
            try:
                os.mkdir(semi_pred_path, 0o777)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        img_path = os.path.join(dir_path, 'flair')
        if not os.path.isdir(img_path):
            os.mkdir(img_path, 0o777)

        #  if subject == 0:
        # subject += 1
        # continue
        print("CREATING PLOTS FOR subject : ", most_gap_index)

        log = os.path.join(dir_path, 'metrics.txt')
        with open(log, "w") as f:
            f.write("diff for subejct {}:\n"
                    " **WMH**  DICE: {}".
                    format(most_gap_index, most_gap_score))

        for i in range(0, len(y_subject)):
            true = final_true[i]
            sup_pred = final_sup_pred[i]
            semi_pred = final_semi_pred[i]
            # print("shape")
            # print(y_subject.shape)
            #  print(true.shape)
            # print(pred.shape)
            # print(x_subject[i][0].shape)

            plt.axis('off')
            plt.imshow(true, cmap='gray')
            plt.savefig(os.path.join(true_path, "true_{}.png".format(i)))

            plt.axis('off')
            plt.imshow(sup_pred.cpu(), cmap='gray')
            plt.savefig(os.path.join(sup_pred_path, "pred{}.png".format(i)))

            plt.axis('off')
            plt.imshow(semi_pred.cpu(), cmap='gray')
            plt.savefig(os.path.join(semi_pred_path, "pred{}.png".format(i)))

            plt.axis('off')
            plt.imshow(final_x[i][0].cpu(), cmap='gray')
            plt.savefig(os.path.join(img_path, "flair{}.png".format(i)))

            plt.axis('off')
            plt.imshow(final_x[i][1].cpu(), cmap='gray')
            plt.savefig(os.path.join(img_path, "t1_{}.png".format(i)))

    # print(
    #     "(WMH) :  DICE SCORE   {}, PPV  {},  Sensitivity: {}, Specificity: {}, Hausdorff: {}".format(
    #         metrics_WMH['dsc'], metrics_WMH['ppv'],
    #         metrics_WMH['sens'],
    #         metrics_WMH['spec'], metrics_WMH['hd']))

    #  log = os.path.join(dir_path, 'metrics.txt')
    #  with open(log, "w") as f:
    #     f.write("SCORE for subejct {}:\n"
    #            " **WMH**  DICE: {}".
    #           format(subject, metrics_WMH['dsc']))


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
        default='PGS_config_WMH.yaml'
    )

    dataset = utils.Constants.Datasets.Wmh_challenge
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))
    # result_path = '/home/sina/WMH_semisup_segmentation/WMH_Unsupervised_Segmentation/src/plots/partially_sup/'
    # model_path = '/home/sina/WMH_semisup_segmentation/WMH_Unsupervised_Segmentation/miccai2022/anthony/semi_alternate/layerwise_normal/sup_ratio_5/seed_40/2022-02-23 21:57:08.434793/best_model/pgsnet_best.model'

    # semi - brats
    #  model_sup_path = '/projects/sina/W-Net/cvpr2022/partiallySup_ratio_3/seed_41/2022-02-15 13:25:46.112535/best_model/pgsnet_best.model'
    # model_semi_path = '/projects/sina/W-Net/miccai2022_final/braTS/semi_alternate/sup_ratio_3/seed_41/1/best_model/pgsnet_best.model'
    # result_path = '/projects/sina/W-Net/src/plots/'

    model_semi_path = '/projects/sina/W-Net/miccai_supplement/semi_alternate/layerwise_normal/sup_ratio_5/seed_42/2022-03-06 16:10:57.314890/best_model/pgsnet_best.model'
    model_sup_path = '/projects/sina/W-Net/miccai_supplement/partially_sup/sup_ratio_5/seed_42/2022-03-06 16:13:33.088266/best_model/pgsnet_best.model'
    result_path = '/projects/sina/W-Net/src/plots/brats-5-percent/'
    # model_sup_path =  '/projects/sina/W-Net/miccai_wmh/partially_sup/sup_ratio_5/seed_42/2022-02-22 23:44:45.731844/best_model/pgsnet_best.model'
    # model_semi_path = '/projects/sina/W-Net/miccai_wmh/semi_alternate/layerwise_normal/sup_ratio_5/seed_42/2022-02-23 17:48:46.447302/best_model/pgsnet_best.model'
    # result_path =  '/projects/sina/W-Net/src/plots/wmh'
    # wmh_dataset(cfg, model_sup_path, model_semi_path, result_path)
    # brats(plot_slice)
    plot_slice(cfg, model_sup_path, model_semi_path, result_path, 356, 48)

if __name__ == '__main__':
    main()
