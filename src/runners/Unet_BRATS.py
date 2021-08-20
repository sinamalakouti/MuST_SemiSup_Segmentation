import os
import sys
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import utils
from utils import model_utils
from evaluation_metrics import dice_coef, get_dice_coef_per_subject
from dataset.Brats20_preprocessed import Brat20Test
from models import Unet
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

import argparse
import yaml
from easydict import EasyDict as edict
from losses.loss import dice_coef_loss, soft_dice_loss

import wandb

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')
# sys.path.append('srs/models')
# sys.path.append('srs/models')

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

for p in sys.path:
    print("path  ", p)
torch.manual_seed(42)
np.random.seed(42)
utils.Constants.USE_CUDA = True
parser = argparse.ArgumentParser()


def __fw_sup_loss(y_preds, y_true, sup_loss):
    total_loss = 0
    # iterate over all level's output

    for output in y_preds:
        ratio = int(np.round(y_true.shape[2] / output.shape[2]))
        maxpool = nn.MaxPool2d(kernel_size=2, stride=ratio, padding=0)
        target = maxpool(y_true)

        if target.shape != output.shape:
            h_diff = output.size()[2] - target.size()[2]
            w_diff = output.size()[3] - target.size()[3]
            #
            target = F.pad(target, (w_diff // 2, w_diff - w_diff // 2,
                                    h_diff // 2, h_diff - h_diff // 2))

        assert output.shape[2:] == target.shape[2:], "output and target shape is not similar!!"
        if output.shape[1] != target.shape[1] and type(sup_loss) == torch.nn.CrossEntropyLoss:
            target = target.reshape((target.shape[0], target.shape[2], target.shape[3])).type(torch.LongTensor)
        total_loss += sup_loss(output, target.to(output.device))
    return total_loss


def unet_compute_loss(y_preds, y_true, loss_fuctions, is_supervised):
    total_loss = 0
    sup_loss = loss_fuctions[0]
    if is_supervised:

        if y_preds.shape[1] != y_true.shape[1]:
            y_true = y_true.reshape((y_true.shape[0], y_true.shape[2], y_true.shape[3])).type(torch.LongTensor)

        total_loss = sup_loss(y_preds, y_true.to(y_preds.device))
    else:
        None
        # for comparing outputs together!
        # total_loss = self.__fw_self_unsup_loss(y_preds, loss_functions)

        # consistency of original output and noisy output
        # total_loss = __fw_outputwise_unsup_loss(y_preds, y_true, loss_functions)

    return total_loss


def trainUnet_sup(train_sup_loader, model, optimizer, device, loss_functions, epochid, cfg):
    model.train()

    for step, batch_sup in enumerate(train_sup_loader):
        optimizer.zero_grad()
        b_sup = batch_sup['data'].to(device)
        target_sup = batch_sup['label'].to(device)

        print("subject is : ", batch_sup['subject'])
        sup_outputs = model(b_sup)
        if cfg.oneHot:
            sf = torch.nn.Softmax2d()
            sup_outputs = sf(sup_outputs)
        total_loss = unet_compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True)
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
                {"batch_id": step + epochid * len(train_sup_loader), "loss": total_loss, "batch_score": dice_score})

    return model, total_loss


def evaluateUnet(model, dataset, device, threshold, cfg, data_mode):
    print("******************** EVALUATING {}********************".format(data_mode))

    testset = utils.get_testset(dataset, cfg.batch_size, intensity_rescale=cfg.intensity_rescale,
                                mixup_threshold=cfg.mixup_threshold, mode=data_mode,
                                t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce, augment=False, oneHot=cfg.oneHot)

    model.eval()

    dice_arr = []
    all_preds = None
    all_targets = None
    all_subjects = None
    with torch.no_grad():
        for batch_id, batch in enumerate(testset):
            b = batch['data']
            b = b.to(device)
            target = batch['label'].to(device)
            outputs = model(b)
            # apply softmax
            if cfg.oneHot:
                sf = torch.nn.Softmax2d()
            else:
                sf = torch.nn.Softmax2d()
            y_pred = sf(outputs)
            if batch_id == 0:
                all_preds = y_pred
                all_subjects = batch['subject']
                all_targets = target
            else:
                all_preds = torch.cat((all_preds, y_pred), dim=0)
                all_subjects = torch.cat((all_subjects, batch['subject']), dim=0)
                all_targets = torch.cat((all_targets, target), dim=0)

            y_WT = seg2WT(y_pred, threshold, cfg.oneHot)
            target[target >= 1] = 1
            target_WT = target
            dice_score = dice_coef(target_WT.reshape(y_WT.shape), y_WT)
            print("now at this iteration   ", batch['subject'])
            print("***** result is  *******")
            print(dice_score)
            dice_arr.append(dice_score.item())

        all_targets[all_targets >= 1] = 1
        all_preds = seg2WT(all_preds, threshold, oneHot=cfg.oneHot)
        subject_wise_DSC = get_dice_coef_per_subject(all_targets.reshape(all_preds.shape), all_preds, all_subjects)

    return np.mean(np.array(dice_arr)), subject_wise_DSC


def eval_per_subjectUnet(model, device, threshold, cfg, data_mode):
    print("******************** EVALUATING {}********************".format(data_mode))
    testset = Brat20Test(f'data/brats20', data_mode, 10, 155,
                         augment=False, center_cropping=True, t1=cfg.t1, t2=cfg.t2, t1ce=cfg.t1ce, oneHot=cfg.oneHot)
    model.eval()
    dice_arr = []
    paths = testset.paths
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for path in paths:
            batch = testset.get_subject(path)
            b = batch['data']
            target = batch['label'].to(device)
            subjects = batch['subjects']
            assert len(np.unique(subjects)) == 1, print("More than one subject at a time")
            b = b.to(device)
            outputs = model(b)
            if cfg.oneHot:
                sf = torch.nn.Softmax2d()
                outputs = sf(outputs)
            print("############# LOSS for subject {} is {} ##############".format(subjects[0], loss_val.item()))

            if cfg.oneHot:
                target[target >= 1] = 1
                target_WT = seg2WT(target, 1, oneHot=cfg.oneHot)
                y_pred = outputs
            else:
                sf = torch.nn.Softmax2d()
                target[target >= 1] = 1
                target_WT = target
                y_pred = sf(outputs)

            y_WT = seg2WT(y_pred, threshold, oneHot=cfg.oneHot)

            dice_score = dice_coef(target_WT.reshape(y_WT.shape), y_WT)
            print("score for subject {} is {}".format(subjects[0], dice_score))
            dice_arr.append(dice_score.item())
    return np.mean(np.array(dice_arr))


def seg2WT(preds, threshold, oneHot=False):
    if oneHot:
        preds = preds >= threshold
        WT_pred = preds[:, 1:4, :, :].sum(1) >= 1
    else:
        max_val, max_indx = torch.max(preds, dim=1)
        max_val = (max_val >= threshold).float()
        max_indx[max_indx >= 1] = 1
        WT_pred = torch.multiply(max_indx, max_val)

    return WT_pred


def train_val(dataset, n_epochs, wmh_threshold, output_dir, learning_rate, args, cfg):
    print("******* TRAINING UNET***********")
    inputs_dim = [4, 64, 128, 256, 512, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, False, False, False, False, False, False, False, False]
    best_score = 0
    start_epoch = 0

    config_dictionary = dict(
        yaml=cfg,
        args=args,
    )
    wandb.init(project="fully_sup_brats", config=config_dictionary)
    wandb.run.name = "UNET_FULLY_SUP" + str(wandb.run.id)
    print("learning_rate is    ", learning_rate)
    step_size = cfg.scheduler_step_size
    print("scheduler step size is :   ", step_size)

    print("output_dir is    ", output_dir)
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    else:
        None

    output_model_dir = os.path.join(output_dir, "best_model")
    print("output_model_dir is   ", output_model_dir)

    if not os.path.isdir(output_model_dir):
        try:
            os.mkdir(output_model_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_model_dir)
    else:
        None

    if not os.path.isdir(os.path.join(output_dir, "runs")):
        try:
            os.mkdir(os.path.join(output_dir, "runs"), 0o777)
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(output_dir, "runs"))
    else:
        None
    output_image_dir = os.path.join(output_dir, "result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)
    else:
        None

    unet = Unet.Wnet(18, 4, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                     separables=separables)
    if torch.cuda.is_available() and type(unet) is not torch.nn.DataParallel:
        unet = torch.nn.DataParallel(unet)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)
    unet.to(device)

    optimizer = torch.optim.SGD(unet.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=cfg.lr_gamma)  # don't use it

    train_sup_loader = utils.get_trainset(dataset, batch_size=cfg.batch_size, intensity_rescale=cfg.intensity_rescale,
                                          mixup_threshold=cfg.mixup_threshold, mode=cfg.train_mode, t1=cfg.t1,
                                          t2=cfg.t2, t1ce=cfg.t1ce, augment=cfg.augment, oneHot=cfg.oneHot)

    sup_loss = torch.nn.CrossEntropyLoss()
    loss_functions = (sup_loss, None)

    for epoch in range(start_epoch, n_epochs):
        print("iteration:  ", epoch)

        unet, loss = trainUnet_sup(train_sup_loader, unet, optimizer, device, loss_functions, epoch, cfg)
        if epoch % 2 == 0:

            subject_wise_score = eval_per_subjectUnet(unet, device, wmh_threshold, cfg, cfg.val_mode)
            print("** SUBJECT WISE SCORE @ Iteration {} is {} **".format(epoch, subject_wise_score))
            # print("** REGULAR SCORE @ Iteration {} is {} **".format(epoch, regular_score))
            if subject_wise_score > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch,
                                                                                                     subject_wise_score))
                best_score = subject_wise_score
                path = os.path.join(output_model_dir, 'psgnet_best_lr{}.model'.format(learning_rate))
                model_utils.save_state_dict(unet, path)

                subject_wise_test_score = eval_per_subjectUnet(unet, device, wmh_threshold, cfg, cfg.test_mode)
                wandb.log({"epoch_id": epoch, "subject_wise_test_DSC": subject_wise_test_score})
                save_score(output_image_dir, subject_wise_score, epoch)
            wandb.log(
                {"epoch_id": epoch, "subject_wise_val_DSC": subject_wise_score})
        scheduler.step()


def save_score(dir_path, score, iter):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)
    else:
        None
    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average dice score per subject (5 image) at iter {}  :   {}".format(iter, score))


def save_predictions(y_pred, threshold, dir_path, score, iter):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)
    else:
        None

    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average dice score per subject (5 image) at iter {}  :   {}".format(iter, score))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)
    else:
        None

    for image_id, image in enumerate(y_pred):
        image = image >= threshold
        image = image.to('cpu')
        plt.imshow(image)
        image_path = os.path.join(dir_path, "image_id_{}.jpg".format(image_id))
        plt.savefig(image_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        default="1",
        type=str,
        help="cuda indices 0,1,2,3"
    )
    parser.add_argument(
        "--output_dir",
        default="dasfdsfsaf",
        type=str,
        help="output directory for results"
    )

    parser.add_argument(
        "--num_supervised",
        default=2,
        type=int,
        help="number of supervised samples"
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

    parser.add_argument(
        "--supervised_subjects",
        type=str,
        help="<subject1>_<subject2> ... <subjectn> or all for all subjects"
    )

    dataset = utils.Constants.Datasets.Brat20
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    train_val(dataset, cfg.n_epochs, cfg.wmh_threshold, args.output_dir, cfg.lr, args, cfg)


if __name__ == '__main__':
    main()
