import torch
import sys
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import utils
from evaluation_metrics import dice_coef
import Pgs
import matplotlib.pyplot as plt

import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as augmentor

import argparse
from torch.utils.tensorboard import SummaryWriter
import wandb

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')
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
        total_loss += sup_loss(output, target)
    return total_loss


def compute_loss(y_preds, y_true, loss_functions, is_supervised):
    total_loss = 0
    if is_supervised:
        total_loss = __fw_sup_loss(y_preds, y_true, loss_functions[0])
        # supervised binary loss
        # total_loss = __fw_sup_loss(y_preds, y_true, loss_functions)
    else:
        None
        # for comparing outputs together!
        # total_loss = self.__fw_self_unsup_loss(y_preds, loss_functions)

        # consistency of original output and noisy output
        # total_loss = __fw_outputwise_unsup_loss(y_preds, y_true, loss_functions)

    return total_loss


def trainPgs_semi(train_sup_loader, train_unsup_loader, model, optimizer, device, epochid):
    model.train()

    for step, (batch_sup, batch_unsup) in enumerate(zip(train_sup_loader, train_unsup_loader)):
        optimizer.zero_grad()
        b_sup = batch_sup['data']
        b_unsup = batch_unsup['data']
        b_sup = b_sup.to(device)
        b_unsup = b_unsup.to(device)
        target_sup = batch_sup['label'].to(device)
        continue
        # unsup_loss = nn.BCELoss()
        # sup_loss = torch.nn.BCELoss()
        unsup_loss = nn.BCELoss()
        loss_functions = (sup_loss, unsup_loss)

        print("subject is : ", batch_sup['subject'])
        print("subject is : ", batch_unsup['subject'])
        sup_outputs, _ = model(b_sup, is_supervised=True)
        total_loss = Pgs.compute_loss(sup_outputs, target_sup, loss_functions, is_supervised=True)
        sup_outputs, unsup_outputs = model(b_unsup, is_supervised=False)
        total_loss += Pgs.compute_loss(unsup_outputs, sup_outputs, loss_functions, is_supervised=False)
        print("**************** LOSSS  : {} ****************".format(total_loss))

        total_loss.backward()
        optimizer.step()
    return model, total_loss


def trainPgs_sup(train_sup_loader, model, optimizer, device, epochid):
    model.train()

    for step, batch_sup in enumerate(train_sup_loader):
        optimizer.zero_grad()
        b_sup = batch_sup['data'].to(device)


        target_sup = batch_sup['label'].to(device)
        sup_loss = torch.nn.CrossEntropyLoss()

        print("subject is : ", batch_sup['subject'])
        sup_outputs, _ = model(b_sup, is_supervised=True)
        total_loss = compute_loss(sup_outputs, target_sup, (sup_loss, None), is_supervised=True)

        print("**************** LOSSS  : {} ****************".format(total_loss))

        total_loss.backward()
        optimizer.step()

    return model, total_loss


def trainPGS(train_loader, model, optimizer, device, epochid):
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        b = batch['data']
        b = b.to(device)
        target = batch['label'].to(device)

        # unsup_loss = nn.MSELoss()
        unsup_loss = nn.BCELoss()
        sup_loss = torch.nn.BCELoss()
        # sup_loss = reconstruction_loss.dice_coef_loss
        # sup_loss = torch.nn.w
        loss_functions = (sup_loss, unsup_loss)
        is_supervised = True
        print("subject is : ", batch['subject'])
        sup_outputs, unsup_outputs = model(b, is_supervised)

        if is_supervised:
            total_loss = Pgs.compute_loss(sup_outputs, target, loss_functions, is_supervised)
        else:

            # raise Exception("unsupervised is false")
            total_loss = Pgs.compute_loss(unsup_outputs, sup_outputs, loss_functions, is_supervised)

        print("****** LOSSS  : Is_supervised: {} *********   :".format(is_supervised), total_loss)

        total_loss.backward()
        optimizer.step()
    return model, total_loss


def evaluatePGS(model, dataset, device, threshold):
    print("******************** EVALUATING ********************")
    testset = utils.get_testset(dataset, 32, True, None, None, mode="test2019_new")

    model.eval()

    dice_arr = []
    segmentation_outputs = []

    with torch.no_grad():
        for batch in testset:
            b = batch['data']
            b = b.to(device)
            target = batch['label'].to(device)
            outputs, _ = model(b, True)
            # apply softmax
            sf = torch.nn.Softmax2d()
            y_pred = sf(outputs[-1]) >= threshold

            y_WT = create_WT_output(y_pred)
            target[target >= 1] = 1
            target_WT = target
            dice_score = dice_coef(target_WT.reshape(y_WT.shape), y_WT)
            dice_arr.append(dice_score.item())
            outputs = outputs[-1].reshape(outputs[-1].shape[0], outputs[-1].shape[1], outputs[-1].shape[2],
                                          outputs[-1].shape[3])
            for output in outputs:
                segmentation_outputs.append(output)

    return np.mean(np.array(dice_arr)), segmentation_outputs


def create_WT_output(preds):
    WT_pred = preds[:, 1:4, :, :].sum(1) >= 1
    return WT_pred


def train_val(dataset, n_epochs, device, wmh_threshold, output_dir, learning_rate, args):
    inputs_dim = [4, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64, 4]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

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

    # writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
    wandb.init(project="fully_sup_brats", config=args)
    wandb.run.name = wandb.run.id
    output_image_dir = os.path.join(output_dir, "result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)
    else:
        None

    pgsnet = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides)


    print("learning_rate is    ", learning_rate)
    optimizer = torch.optim.SGD(pgsnet.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
    step_size = 30
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)  # don't use it
    print("scheduler step size is :   ", step_size)
    best_score = 0
    start_epoch = 0

    if torch.cuda.is_available() and type(pgsnet) is not torch.nn.DataParallel:
        pgsnet = torch.nn.DataParallel(pgsnet)
        device = 'cuda'
    elif not torch.cuda.is_available():
        device = 'cpu'

    device = torch.device(device)
    # pgsnet = utils.load_model(path=os.path.join(output_model_dir, 'psgnet_best_lr{}.model'.format(learning_rate)),
    #                           device=device)

    pgsnet.to(device)

    for epoch in range(start_epoch, n_epochs):
        print("iteration:  ", epoch)
        train_sup_loader = utils.get_trainset(dataset, 32, True, None, None, mode='train2018_sup')
        # train_unsup_loader = utils.get_trainset(dataset, 32, True, None, None, mode='train_semi_unsup')
        # pgsnet, loss = trainPGS(train_loader, pgsnet, optimizer, device, epoch)
        # pgsnet, loss = trainPgs_semi(train_sup_loader, train_unsup_loader, pgsnet, optimizer, device, epoch)
        # score, segmentations = evaluatePGS(pgsnet, dataset, device, wmh_threshold)
        pgsnet, loss = trainPgs_sup(train_sup_loader, pgsnet, optimizer, device, epoch)
        # writer.add_scalar("Loss/train", loss, epoch)

        if epoch % 1 == 0:
            score, segmentations = evaluatePGS(pgsnet, dataset, device, wmh_threshold)
            # writer.add_scalar("dice_score/test", score, epoch)
            print("** SCORE @ Iteration {} is {} **".format(epoch, score))
            if score > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch, score))
                best_score = score
                path = os.path.join(output_model_dir, 'psgnet_best_lr{}.model'.format(learning_rate))
                with open(path, 'wb') as f:
                    torch.save(pgsnet, f)

                save_score(output_image_dir, score, epoch)
            wandb.log({"train_loss": loss, "dev_dsc": score})
            # example = segmentations[0]
            # example = example >= wmh_threshold
        #         wandb.log(
        #             {"train_loss": loss, "dev_dsc": score, "image": [wandb.Image(augmentor.to_pil_image(example.int())
        #                                                                          , caption="output example")]})
        scheduler.step()

    # writer.flush()
    # writer.close()


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
        default=400,
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
        default=2,
        type=int,
        help="number of supervised samples"
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

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:{}".format(args.cuda)
    else:
        dev = "cpu"
    print("device is     ", dev)

    device = torch.device(dev)
    train_val(dataset, args.n_epochs, device, args.wmh_threshold, args.output_dir, args.lr, args)


def get_cluster_assumption_representation(h):
    l_rep = h.shape[0]
    n_rows = h.shape[1]
    n_cols = h.shape[2]
    diff = torch.zeros((n_rows, n_cols))

    for r in range(1, n_rows - 1):
        for c in range(1, n_cols - 1):
            main_patch = h[:, r, c]
            main_patch = main_patch.reshape(main_patch.shape[0], 1, 1)
            patch = h[:, r - 1:r + 2, c - 1:c + 2]
            d = torch.sqrt(torch.sum((main_patch - patch) ** 2, axis=0))

            diff[r, c] = d.mean()
    return diff


def get_cluster_assumption(image):
    n_rows = image.shape[0]
    n_cols = image.shape[1]
    size = 1
    diff = torch.zeros(image.shape)
    for r in range(2 + size, n_rows - 2 - size):
        for c in range(2 + size, n_cols - 2 - size):
            main_patch = image[r - 1 - size:r + 1 + size, c - 1 - size:c + 1 + size]
            for rd in range(-1, 2):
                for cd in range(-1, 2):
                    patch = image[(r + rd) - 1 - size:(r + rd) + 1 + size, (c + cd) - 1 - size:(c + cd) + 1 + size]
                    diff[r, c] += torch.sqrt(torch.sum((main_patch - patch) ** 2))
            diff[r, c] = diff[r, c] / 8

    return diff


if __name__ == '__main__':
    main()
