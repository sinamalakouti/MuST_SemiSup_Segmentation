from utils.Constants import *
from utils.dataloader import *
import Wnet
import matplotlib.pyplot as plt
import os
import torch
from evaluation_metrics import dice_coef

import random
import numpy as np


# class Utils:
#     def __init__(self, dataset):


# def get_subjects(dataset : utils.dataloader.PittLocalFull):
#     subjects = set()
#     for item in  dataset.order:
#         id = item.strip('_')[0]
#         subjects.add(id)
#     subjects = list(subjects)
#
#     return subjects
#
# def get_random_subjects(subjects, n):
#     if n > len(subjects):
#         raise ("n is bigger than lenght of list")
#     if n == len(subjects):
#         raise ("n = len(array)")
#
#     return random.sample(subjects, n)
def __ema(p1, p2, factor):
    return factor * p1 + (1 - factor) * p2


def ema_update(student, teacher, cur_step, L = 400):
    if cur_step < L:
        alpha = 0.99
    else:
        alpha = 0.999

        for stud_p, teach_p in zip(student.parameters(), teacher.parameters()):
            teacher.data = __ema(teach_p.data, stud_p.data, alpha)
    return student, teacher


def update_adaptiveRate(cur_step, L):
    if cur_step > L:
        return 1.0
    return np.exp(-5 * (1 - cur_step / L) ** 2)


def get_trainset(dataset, batch_size, intensity_rescale, has_t1, mixup_threshold) -> torch.utils.data.DataLoader:
    mem_pin = False
    if Constants.USE_CUDA:
        mem_pin = True
    if dataset is Datasets.PittLocalFull:
        batch_sz = batch_size
        train = torch.utils.data.DataLoader(
            PittLocalFull(
                None,
                has_t1,
                mixup_threshold,
                None,
                intensity_rescale,
                [f'paths/fold-0/data_paths.txt',
                 f'paths/fold-2/data_paths.txt', f'paths/fold-3/data_paths.txt',
                 f'paths/fold-4/data_paths.txt'],
                [f'paths/fold-0/label_paths.txt', f'paths/fold-2/label_paths.txt',
                 f'paths/fold-3/label_paths.txt', f'paths/fold-4/label_paths.txt'],
                [f'paths/fold-0/mask_paths.txt', f'paths/fold-2/mask_paths.txt',
                 f'paths/fold-3/mask_paths.txt', f'paths/fold-4/mask_paths.txt'],
                augment=True,
                is_FCM=Constants.FCM,
                data_paths_t1=[f'paths/fold-0/data_paths_t1.txt',
                               f'paths/fold-2/data_paths_t1.txt', f'paths/fold-3/data_paths_t1.txt',
                               f'paths/fold-4/data_paths_t1.txt']
            ),
            batch_size=batch_sz,
            drop_last=True,
            num_workers=0,
            shuffle=False,
            pin_memory=mem_pin
        )
    return train


def get_testset(dataset, batch_size, intensity_rescale, has_t1, mixup_threshold) -> torch.utils.data.DataLoader:
    mem_pin = False
    if Constants.USE_CUDA:
        mem_pin = True
    if dataset is Datasets.PittLocalFull:
        batch_sz = batch_size
        test = torch.utils.data.DataLoader(
            PittLocalFull(
                None,
                has_t1,
                mixup_threshold,
                None,
                intensity_rescale,
                [f'paths/fold-1/data_paths.txt'],
                [f'paths/fold-1/label_paths.txt'],
                [f'paths/fold-1/mask_paths.txt'],
                augment=False,
                is_FCM=Constants.FCM,
                data_paths_t1=[f'paths/fold-1/data_paths_t1.txt']
            ),
            batch_size=batch_sz,
            drop_last=False,
            num_workers=0,
            shuffle=False,
            pin_memory=mem_pin
        )

        return test


def load_model(path) -> Wnet.Wnet:
    wnet = torch.load(path, map_location=torch.device('cpu'))
    # wnet = torch.load(path)

    return wnet


def save_segment_images(segments, path):
    n_segments = segments.shape[1]
    n_samples = segments.shape[0]
    max_images = segments.argmax(1)
    for i in range(n_samples):
        sample_dir = path + '/sample_{}'.format(i)
        if not os.path.isdir(sample_dir):
            try:
                os.mkdir(sample_dir)
            except OSError:
                print("Creation of the directory %s failed" % path)
            else:
                print("Successfully created the directory %s " % path)

        for j in range(n_segments):
            # maximum = torch.max(segments[:, j, :, :])
            # minimum = torch.min(segments[:, j, :, :])
            # segments[:, j, :, :] = (segments[:, j, :, :] - minimum) / (maximum - minimum)

            plt.imshow(segments[i, j])
            image_path = sample_dir + "/segment_{}.png".format(j)
            plt.savefig(image_path)
            sgm = max_images == j
            plt.imshow(sgm[i].reshape(segments.shape[2], segments.shape[3]), 'gray')
            image_path = sample_dir + "/segment_argmax_{}.png".format(j)
            plt.savefig(image_path)

            threshold = 0.85
            plt.imshow(segments[i, j] > threshold, 'gray')
            image_path = sample_dir + "/segment_threshold_{}_{}.png".format(threshold, j)
            plt.savefig(image_path)


def save_images(input_images, reconstructed_images, path):
    n_image = reconstructed_images.shape[0]
    for i in range(n_image):
        recon_image = reconstructed_images[i, 0]
        input_image = input_images[i, 0]
        recon_image_path = path + "/recon_image{}.png".format(i)
        input_image_path = path + "/input_image{}.png".format(i)
        plt.imshow(recon_image)
        plt.savefig(recon_image_path)
        plt.imshow(input_image)
        plt.savefig(input_image_path)


def save_label(y, path):
    n_imags = y.shape[0]
    for i in range(n_imags):
        label_image = y[i, :]
        label_image_path = path + "/label_image{}.png".format(i)
        plt.imshow(label_image)
        plt.savefig(label_image_path)


def _evaluate(y_true, predictions, segment_index):
    segments = predictions.argmax(1)
    wmh_segment = segments == segment_index
    dice_score = dice_coef(y_true.reshape(wmh_segment.shape), wmh_segment)
    dice_arr = dice_score

    return dice_arr.mean()


def evaluate(dataset, model, output_path):
    testset = utils.get_testset(dataset, 5, True)
    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    model.to(device)

    dice_scores = []
    with torch.no_grad():
        for batch in testset:
            x_test = batch['data']
            x_mask = batch['mask']
            y_true = batch['label']
            x_test = x_test.to(device)
            y_true = y_true.to(device)
            x_mask = x_mask.to(device)
            segmentation = model.U_enc_fw(x_test)
            segmentation = torch.mul(x_mask, segmentation)
            score = _evaluate(y_true, segmentation, 1)
            dice_scores.append(score)
    dice_scores = torch.tensor(dice_scores)
    text = "mean dice score subject wise:  {}\n ".format(torch.mean(dice_scores))
    text_file = open(output_path + "/result.txt", "w")
    text_file.write(text)
    text_file.close()
