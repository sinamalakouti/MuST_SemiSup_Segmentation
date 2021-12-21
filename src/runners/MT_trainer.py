import sys
import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import utils
from utils import model_utils
from losses.evaluation_metrics import dice_coef
from models import Pgs
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal

import numpy as np
import argparse
import wandb

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')

for p in sys.path:
    print("path  ", p)
torch.manual_seed(42)
np.random.seed(42)
utils.Constants.USE_CUDA = True
parser = argparse.ArgumentParser()


def train_supervised(dataset, model, optimizer, device):
    model.train()
    train_loader = utils.get_trainset(dataset, 5, True, None, None)

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        b = batch['data']
        target = batch['label'].to(device)
        b = b.to(device)
        model.to(device)
        sup_loss = torch.nn.BCELoss()
        loss_functions = (sup_loss, None)

        if "0177CR" in batch['subject'][0] or '0064KW' in batch['subject'][0]:
            is_supervised = True
        else:
            continue

        print("subject is : ", batch['subject'])

        sup_outputs, unsup_outputs = model(b, is_supervised)
        if is_supervised:
            total_loss = model.compute_loss(sup_outputs, target, loss_functions, is_supervised)
        else:
            print("***** ERROR : unsupervised training in only supervised model!")
            assert False, "***** ERROR : unsupervised training in only supervised model!"

        print("****** LOSSS  : Is_supervised: {} *********   :".format(is_supervised), total_loss)

        total_loss.backward()
        optimizer.step()
    return model, total_loss


def train_MT(dataset, student_model, teacher_model, optimizer, device, cur_epoch):
    student_model.train()
    teacher_model.train()

    train_loader = utils.get_trainset(dataset, 5, True, None, None)
    total_num_batches = len(train_loader)
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        b = batch['data']
        target = batch['label'].to(device)
        b = b.to(device)
        student_model.to(device)
        teacher_model.to(device)

        unsup_loss = nn.BCELoss()
        sup_loss = torch.nn.BCELoss()
        loss_functions = (sup_loss, unsup_loss)

        if "0177CR" in batch['subject'][0] or '0064KW' in batch['subject'][0]:
            is_supervised = True

        else:
            is_supervised = False

        print("subject is : ", batch['subject'])
        unsup_err = 0
        sup_err = 0
        if is_supervised:
            student_output, _ = student_model(b, is_supervised)
            sup_err = student_model.compute_loss(student_output, target, loss_functions, is_supervised)
        else:
            with torch.no_grad():
                # todo apply noise twice to the input
                additive_dist = Normal(torch.tensor([0.0]), torch.tensor([0.05]))
                multiplicative_dist = Normal(torch.tensor([1.0]), torch.tensor([0.01]))
                ns = additive_dist.sample(b.shape).reshape(b.shape).to(b.device)
                nm = multiplicative_dist.sample(b.shape).reshape(b.shape).to(b.device)

                b_teacher = (b + ns) * nm

                ns = additive_dist.sample(b.shape).reshape(b.shape).to(b.device)
                nm = multiplicative_dist.sample(b.shape).reshape(b.shape).to(b.device)

                b_student = (b + ns) * nm
                teacher_output, _ = teacher_model(b_teacher, True)

            student_output, _ = student_model(b_student, True)

            unsup_err = student_model.compute_loss(student_output, teacher_output, loss_functions, is_supervised)
        total_loss = sup_err + unsup_err * model_utils.update_adaptiveRate(total_num_batches * cur_epoch + step, 400)
        print("****** LOSSS  : Is_supervised: {} *********   :".format(is_supervised), total_loss)

        total_loss.backward()
        optimizer.step()
        student_model, teacher_model = model_utils.ema_update(student_model, teacher_model,
                                                              cur_epoch * total_num_batches + step, 400)
    return student_model, teacher_model, total_loss


def evaluatePGS(model, dataset, device, threshold):
    testset = utils.get_testset(dataset, 5, True, None, None)

    model.eval()
    model = model.to(device)
    dice_arr = []
    segmentation_outputs = []

    with torch.no_grad():
        for batch in testset:
            b = batch['data']
            b = b.to(device)
            target = batch['label'].to(device)
            outputs, _ = model(b, True)

            y_pred = outputs[-1] >= threshold
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2], y_pred.shape[3])

            dice_score = dice_coef(target.reshape(y_pred.shape), y_pred)
            dice_arr.append(dice_score.item())
            outputs = outputs[-1].reshape(outputs[-1].shape[0], outputs[-1].shape[2], outputs[-1].shape[3])
            for output in outputs:
                segmentation_outputs.append(output)

    return np.mean(np.array(dice_arr)), segmentation_outputs


def train_val(dataset, n_epochs, device, wmh_threshold, output_dir, learning_rate, args):
    inputs_dim = [1, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64]
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
    wandb.init(project="semi_supervised_wmh", config=args)
    wandb.run.name = wandb.run.id
    output_image_dir = os.path.join(output_dir, "result_images/")

    if not os.path.isdir(output_image_dir):
        try:
            os.mkdir(output_image_dir, 0o777)
        except OSError:
            print("Creation of the directory %s failed" % output_image_dir)
    else:
        None

    teacher_pgs = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides)
    student_pgs = Pgs.PGS(inputs_dim, outputs_dim, kernels, strides)

    print("learning_rate is    ", learning_rate)
    teacher_optimizer = torch.optim.RMSprop(teacher_pgs.parameters(), 0.0001, momentum=0.6, weight_decay=1e-5)
    student_optimizer = torch.optim.RMSprop(student_pgs.parameters(), 0.0001, momentum=0.6, weight_decay=1e-5)
    step_size = 80
    scheduler = lr_scheduler.StepLR(student_optimizer, step_size=step_size, gamma=0.9)
    best_score = 0

    # first train teacher model with only supervised sets

    n_sup_epochs = 20
    for epoch in range(n_sup_epochs):
        teacher_pgs.train()
        student_pgs.train()
        student_pgs, loss = train_supervised(dataset, student_pgs, student_optimizer, device)

        score, segmentations = evaluatePGS(teacher_pgs, dataset, device, wmh_threshold)
        # writer.add_scalar("dice_score/test", score, epoch)
        print("** SCORE @ Iteration {} is {} **".format(epoch, score))
        if score > best_score:
            print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch, score))
            best_score = score
            save_predictions(segmentations, wmh_threshold, output_image_dir, score, epoch)

        wandb.log({"train_loss": loss, "dev_dsc": score})
        teacher_pgs.to(device)
        for teach_p, stud_p in zip(teacher_pgs.parameters(), student_pgs.parameters()):
            teach_p.data = stud_p.data
        # student_pgs, teacher_pgs = model_utils.ema_update(student_pgs, teacher_pgs,
        #                                                  epoch * 16 , 400)

    best_score = 0
    for epoch in range(n_sup_epochs, n_epochs + n_sup_epochs):
        teacher_pgs.train()
        student_pgs.train()
        print("iteration:  ", epoch)
        student_pgs, teacher_pgs, loss = train_MT(dataset, student_pgs, teacher_pgs, student_optimizer, device, epoch)
        # writer.add_scalar("Loss/train", loss, epoch)

        if epoch % 1 == 0:
            score, segmentations = evaluatePGS(teacher_pgs, dataset, device, wmh_threshold)
            # writer.add_scalar("dice_score/test", score, epoch)
            print("** SCORE @ Iteration {} is {} **".format(epoch, score))
            if score > best_score:
                print("****************** BEST SCORE @ ITERATION {} is {} ******************".format(epoch, score))
                best_score = score
                path = os.path.join(output_model_dir, 'psgnet_best_lr{}.model'.format(learning_rate))
                with open(path, 'wb') as f:
                    torch.save(teacher_pgs, f)

                save_predictions(segmentations, wmh_threshold, output_image_dir, score, epoch)
        scheduler.step()

        wandb.log({"train_loss": loss, "dev_dsc": score})
    #
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
    dataset = utils.Constants.Datasets.PittLocalFull
    args = parser.parse_args()

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:{}".format(args.cuda)
    else:
        dev = "cpu"
    print("device is     ", dev)

    device = torch.device(dev)
    # output_dir = '/Users/sinamalakouti/Desktop/alaki'
    train_val(dataset, args.n_epochs, device, args.wmh_threshold, args.output_dir, args.lr, args)


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
