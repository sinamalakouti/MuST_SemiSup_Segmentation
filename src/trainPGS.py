import torch
import sys
import os
import torch
from utils import utils
from evaluation_metrics import dice_coef
import Pgs
import matplotlib.pyplot as plt
from utils import reconstruction_loss
import torch.nn as nn
from torch.optim import lr_scheduler
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


def trainPGS(dataset, model, optimizer, device):
    model.train()
    train_loader = utils.get_trainset(dataset, 5, True, None, None)

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        b = batch['data']
        target = batch['label'].to(device)
        b = b.to(device)
        model.to(device)

        outputs = model(b)
        lossf = nn.MSELoss()
        sup_loss = torch.nn.BCELoss()
        loss_functions = (sup_loss, lossf)
        is_supervised = True
        if step % 4 == 0:
            is_supervised = True
        else:
            is_supervised = False
        if is_supervised:
            total_loss = model.compute_loss(outputs, target, loss_functions, is_supervised)
        else:
            total_loss = model.compute_loss(outputs, outputs, loss_functions, is_supervised)

        print("****** LOSSS  *********   ", total_loss)

        total_loss.backward()
        optimizer.step()
    return model


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
            outputs = model(b)

            y_pred = outputs[-1] >= threshold
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2], y_pred.shape[3])

            dice_score = dice_coef(target.reshape(y_pred.shape), y_pred).mean()
            dice_arr.append(dice_score.item())
            if len(res) == 0:
                res.append(y_pred[-1][0])

    return np.mean(np.array(dice_arr)), res


def train_val(dataset, n_epochs, device, wmh_threshold, output_dir, learning_rate):
    inputs_dim = [1, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    output_model_dir = os.path.join(output_dir, "/best_model")
    if not os.path.isdir(output_model_dir):
        try:
            os.mkdir(output_model_dir)
        except OSError:
            print("Creation of the directory %s failed" % output_model_dir)
    else:
        None

    output_image_dir = os.path.join(output_dir, "/result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)
    else:
        None

    pgsnet = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides)
    print(learning_rate)
    optimizer = torch.optim.SGD(pgsnet.parameters(), learning_rate,momentum=0.9, weight_decay=1e-4)
    # print(pgsnet)
    print(pgsnet.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_score = 0
    for epoch in range(n_epochs):
        print("iteration:  ", epoch)
        pgsnet = trainPGS(dataset, pgsnet, optimizer, device)
        scheduler.step()
        if epoch % 1 == 0:
            score, _ = evaluatePGS(pgsnet, dataset, device, 0.5)
            print("** SCORE @ Iteration {} is {} **".format(epoch, score))
            if score > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch, score))
                best_score = score
                path = os.path.join(output_model_dir, 'psgnet_best_lr{}.model'.format(learning_rate))
                with open(path, 'wb') as f:
                    torch.save(pgsnet, f)

                save_score(output_image_dir, score, epoch)
                # save_predictions(wmh_threshold, wmh_threshold, output_image_dir, score, epoch)


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
        default=0.8,
        type=float,
        help=" wmh mask threshold between 0 and 1"
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