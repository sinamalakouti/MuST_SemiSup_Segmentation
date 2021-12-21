from utils.Constants import *
from dataset.dataloader import *
from dataset.Brat20 import *
import matplotlib.pyplot as plt
import os
import torch
from losses.evaluation_metrics import dice_coef


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


def get_trainset(dataset, batch_size, intensity_rescale, mixup_threshold=None,
                 mode='train', t1=False, t2=False, t1ce=False, augment=False, oneHot=False,seed =None) -> torch.utils.data.DataLoader:
    mem_pin = False
    if Constants.USE_CUDA:
        mem_pin = True
    if dataset is Datasets.PittLocalFull:
        batch_sz = batch_size
        train = torch.utils.data.DataLoader(
            PittLocalFull(
                False,
                t1,
                mixup_threshold,
                False,
                intensity_rescale,
                [f'paths/fold-0/data_paths.txt',
                 f'paths/fold-2/data_paths.txt', f'paths/fold-3/data_paths.txt',
                 f'paths/fold-4/data_paths.txt'],
                [f'paths/fold-0/label_paths.txt', f'paths/fold-2/label_paths.txt',
                 f'paths/fold-3/label_paths.txt', f'paths/fold-4/label_paths.txt'],
                [f'paths/fold-0/mask_paths.txt', f'paths/fold-2/mask_paths.txt',
                 f'paths/fold-3/mask_paths.txt', f'paths/fold-4/mask_paths.txt'],
                augment=augment,
                is_FCM=Constants.FCM,
                data_paths_t1=[f'paths/fold-0/data_paths_t1.txt',
                               f'paths/fold-2/data_paths_t1.txt', f'paths/fold-3/data_paths_t1.txt',
                               f'paths/fold-4/data_paths_t1.txt']
            ),
            batch_size=batch_sz,
            drop_last=True,
            num_workers=0,
            shuffle=True,
            pin_memory=mem_pin
        )
    elif dataset is Datasets.Brat20:
        batch_sz = batch_size
        train = torch.utils.data.DataLoader(
            Brat20(
                dataroot_dir=f'data/brats20',
                mode=mode,
                min_slice_index=10,
                max_slice_index=155,
                center_cropping=True,
                t1=t1,
                t2=t2,
                t1ce=t1ce,
                augment=augment,
                oneHot=oneHot,
                seed=seed
            ),
            batch_size=batch_sz,
            drop_last=True,
            num_workers=0,
            shuffle=True,
            pin_memory=mem_pin
        )
    elif dataset is Datasets.WmhChallenge:
        batch_sz = batch_size

        splits, num_domains = get_splits(
            'WMH_SEG',  # get data of different domains
            T1=True,
            whitestripe=False,
            experiment_mode='semi',
            isSupervised=False,
            test_on_local=False)

        train = torch.utils.data.DataLoader(
            Brat20(
                dataroot_dir=f'data/brats20',
                mode=mode,
                min_slice_index=10,
                max_slice_index=155,
                center_cropping=True,
                t1=t1,
                t2=t2,
                t1ce=t1ce,
                augment=augment,
                oneHot=oneHot,
                seed=seed
            ),
            batch_size=batch_sz,
            drop_last=True,
            num_workers=0,
            shuffle=True,
            pin_memory=mem_pin
        )


    return train


def get_testset(dataset, batch_size, intensity_rescale, mixup_threshold=None,
                mode='test2019_new', t1=False, t2=False, t1ce=False, augment=False) -> torch.utils.data.DataLoader:
    mem_pin = False
    if Constants.USE_CUDA:
        mem_pin = True
    if dataset is Datasets.PittLocalFull:
        batch_sz = batch_size
        test = torch.utils.data.DataLoader(
            PittLocalFull(
                False,
                t1,
                mixup_threshold,
                False,
                intensity_rescale,
                [f'paths/fold-1/data_paths.txt'],
                [f'paths/fold-1/label_paths.txt'],
                [f'paths/fold-1/mask_paths.txt'],
                augment=augment,
                is_FCM=Constants.FCM,
                data_paths_t1=[f'paths/fold-1/data_paths_t1.txt']
            ),
            batch_size=batch_sz,
            drop_last=False,
            num_workers=0,
            shuffle=False,
            pin_memory=mem_pin
        )
    elif dataset is Datasets.Brat20:
        batch_sz = batch_size
        test = torch.utils.data.DataLoader(
            Brat20(
                dataroot_dir=f'data/brats20',
                mode=mode,
                min_slice_index=10,
                max_slice_index=155,
                center_cropping=True,
                t1=t1,
                t2=t2,
                t1ce=t1ce,
                augment=augment
            ),
            batch_size=batch_sz,
            drop_last=True,
            num_workers=0,
            shuffle=False,
            pin_memory=mem_pin
        )

    return test


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
    testset = get_testset(dataset, 5, True)
    if torch.cuda.is_available() and Constants.USE_CUDA:
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



#
#
# def get_cluster_assumption_representation(h):
#     l_rep = h.shape[0]
#     n_rows = h.shape[1]
#     n_cols = h.shape[2]
#     diff = torch.zeros((n_rows, n_cols))
#
#     for r in range(1, n_rows - 1):
#         for c in range(1, n_cols - 1):
#             main_patch = h[:, r, c]
#             main_patch = main_patch.reshape(main_patch.shape[0], 1, 1)
#             patch = h[:, r - 1:r + 2, c - 1:c + 2]
#             d = torch.sqrt(torch.sum((main_patch - patch) ** 2, axis=0))
#
#             diff[r, c] = d.mean()
#     return diff
#
#
# def get_cluster_assumption(image):
#     n_rows = image.shape[0]
#     n_cols = image.shape[1]
#     size = 1
#     diff = torch.zeros(image.shape)
#     for r in range(2 + size, n_rows - 2 - size):
#         for c in range(2 + size, n_cols - 2 - size):
#             main_patch = image[r - 1 - size:r + 1 + size, c - 1 - size:c + 1 + size]
#             for rd in range(-1, 2):
#                 for cd in range(-1, 2):
#                     patch = image[(r + rd) - 1 - size:(r + rd) + 1 + size, (c + cd) - 1 - size:(c + cd) + 1 + size]
#                     diff[r, c] += torch.sqrt(torch.sum((main_patch - patch) ** 2))
#             diff[r, c] = diff[r, c] / 8
#
#     return diff
