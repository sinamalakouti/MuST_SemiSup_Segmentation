import torch
import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import utils
from evaluation_metrics import dice_coef
import Pgs
import matplotlib.pyplot as plt
from utils import reconstruction_loss

import numpy as np

import argparse

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')

for p in sys.path:
    print("path  ", p)

utils.Constants.USE_CUDA = True
utils.Constants.N_ITERATION = 20000
parser = argparse.ArgumentParser()


def trainPGS(dataset, model, optimizer, device, epochid):
    model.train()
    train_loader = utils.get_trainset(dataset, 5, True, None, None)


    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        b = batch['data']
        target = batch['label'].to(device)
        b = b.to(device)
        model.to(device)

        unsup_loss = nn.MSELoss()
        sup_loss = torch.nn.BCELoss()
        # sup_loss = reconstruction_loss.dice_coef_loss
        # sup_loss = torch.nn.w
        loss_functions = (sup_loss, unsup_loss)

        if "0286MR72" in batch['subject'][0] or '0120LB' in batch['subject'][0]:
            is_supervised = True

        else:
            is_supervised = False
            # continue
        if epochid < 5 and not is_supervised:
            continue

        print("subject is : ", batch['subject'])

        # if epochid % 4 == 0:
        #     is_supervised =    True
        # else:
        #     is_supervised = False
        #     if epochid < 5:
        #         continue

        outputs = model(b, is_supervised)
        if is_supervised:
            total_loss = model.compute_loss(outputs, target, loss_functions, is_supervised)
        else:
            # raise Exception("unsupervised is false")
            total_loss = model.compute_loss(outputs, outputs, loss_functions, is_supervised)

        print("****** LOSSS  : Is_supervised: {} *********   :".format(is_supervised), total_loss)

        total_loss.backward()
        optimizer.step()
    return model, total_loss


def evaluatePGS(model, dataset, device, threshold):
    testset = utils.get_testset(dataset, 5, True, None, None)

    model.eval()
    model = model.to(device)
    dice_arr = []
    res = []

    with torch.no_grad():
        for batch in testset:
            b = batch['data']
            b = b.to(device)
            target = batch['label'].to(device)
            outputs = model(b,True)

            y_pred = outputs[-1] >= threshold
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2], y_pred.shape[3])

            dice_score = dice_coef(target.reshape(y_pred.shape), y_pred)
            dice_arr.append(dice_score.item())
            if len(res) == 0:
                res.append(y_pred[-1][0])

    return np.mean(np.array(dice_arr)), res


def train_val(dataset, n_epochs, device, wmh_threshold, output_dir, learning_rate):
    inputs_dim = [1, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    print("output_dir is    ", output_dir)
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError:
            print("Creation of the directory %s failed" % output_dir)
    else:
        None

    output_model_dir = os.path.join(output_dir, "best_model")


    print("output_model_dir is   ", output_model_dir)

    if not os.path.isdir(output_model_dir):
        try:
            os.mkdir(output_model_dir)
        except OSError:
            print("Creation of the directory %s failed" % output_model_dir)
    else:
        None

    if not os.path.isdir(os.path.join(output_dir, "runs")):
        try:
            os.mkdir(os.path.join(output_dir, "runs"))
        except OSError:
            print("Creation of the directory %s failed" % os.path.join(output_dir, "runs"))
    else:
        None

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))

    output_image_dir = os.path.join(output_dir, "result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)
    else:
        None

    pgsnet = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides)
    print("learning_rate is    " , learning_rate)
    optimizer = torch.optim.SGD(pgsnet.parameters(), learning_rate,momentum=0.9, weight_decay=1e-4)
    print(pgsnet.parameters())
    step_size = 50
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)# don't use it
    print("scheduler step size is :   ",step_size)
    best_score = 0


    for epoch in range(n_epochs):
        print("iteration:  ", epoch)
        score, _ = evaluatePGS(pgsnet, dataset, device, wmh_threshold)
        pgsnet, loss = trainPGS(dataset, pgsnet, optimizer, device, epoch)
        writer.add_scalar("Loss/train", loss, epoch)

        if epoch % 1 == 0:
            score, _ = evaluatePGS(pgsnet, dataset, device, wmh_threshold)
            writer.add_scalar("dice_score/test", score, epoch)
            print("** SCORE @ Iteration {} is {} **".format(epoch, score))
            if score > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch, score))
                best_score = score
                path = os.path.join(output_model_dir, 'psgnet_best_lr{}.model'.format(learning_rate))
                with open(path, 'wb') as f:
                    torch.save(pgsnet, f)

                save_score(output_image_dir, score, epoch)
                # save_predictions(wmh_threshold, wmh_threshold, output_image_dir, score, epoch)
        scheduler.step()
    writer.flush()
    writer.close()

def save_score(dir_path, score, iter):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)
    else:
        None
    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average dice score per subject (5 image) at iter {}  :   {}".format(iter, score))

def save_predictions(y_pred, threshold, dir_path, score, iter):
    dir_path = os.path.join(dir_path, "results_iter{}".format(iter))
    output_score_path = os.path.join(dir_path, "result.txt")
    with open(output_score_path, "w") as f:
        f.write("average dice score per subject (5 image) at iter {}  :   {}".format(iter, score))
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Creation of the directory %s failed" % dir_path)
    else:
        None
    y_pred = y_pred >= threshold
    for image_id, image in enumerate(y_pred):
        segment = image[0]
        plt.imshow(segment)
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
        "--intensity_normalization",
        default=True,
        type=bool,
        help="intensity_normalization = T/F"
    )
    parser.add_argument(
        "--addT1",
        default=None,
        type=int,
        help="add T1?  1/None"
    )

    parser.add_argument(
        "--mixup-threshold",
        default=None,
        type=float,
        help="mixup threshold = None or float value"
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--optimizer",
        default="SGD",
        type=str,
        help="SGD or ADAM"
    )
    parser.add_argument(
        "--output_dir",
        default="dasfdsfsaf",
        type=str,
        help="output directory for results"
    )

    parser.add_argument(
        "--n_epochs",
        default = 1000,
        type=int,
        help="number of epochs"
    )

    parser.add_argument(
        "--wmh_threshold",
        default=0.5,
        type=float,
        help=" wmh mask threshold between 0 and 1"
    )

    parser.add_argument(
        "--num_supervised",
        default= 2,
        type = int,
        help = "number of supervised samples"
    )

    dataset = utils.Constants.Datasets.PittLocalFull
    args = parser.parse_args()

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:{}".format(args.cuda)
    else:
        dev = "cpu"
    print("device is     ", dev)

    device = torch.device(dev)
    output_dir = '/Users/sinamalakouti/Desktop/alaki'
    train_val(dataset, args.n_epochs, device, args.wmh_threshold, args.output_dir, args.lr)


if __name__ == '__main__':

    main()
