from utils.Constants import *
from utils.dataloader import *
import Wnet
import matplotlib.pyplot as plt
import os
import torch

# class Utils:
#     def __init__(self, dataset):

def get_trainset(dataset,intensity_rescale) -> torch.utils.data.DataLoader:
    mem_pin = False
    if Constants.USE_CUDA:
        mem_pin = True
    if dataset is Datasets.PittLocalFull:
        batch_sz = 5
        train = torch.utils.data.DataLoader(
            PittLocalFull(
                1,
                None,
                None,
                None,
                intensity_rescale,
                [f'paths/fold-1/data_paths_ws.txt',
                 f'paths/fold-2/data_paths_ws.txt', f'paths/fold-3/data_paths_ws.txt',
                 f'paths/fold-4/data_paths_ws.txt'],
                [f'paths/fold-1/label_paths.txt', f'paths/fold-2/label_paths.txt',
                 f'paths/fold-3/label_paths.txt', f'paths/fold-4/label_paths.txt'],
                [f'paths/fold-1/mask_paths.txt', f'paths/fold-2/mask_paths.txt',
                 f'paths/fold-3/mask_paths.txt', f'paths/fold-4/mask_paths.txt'],
                augment=False),
            batch_size=batch_sz,
            drop_last=True,
            num_workers=0,
            shuffle=True,
            pin_memory=mem_pin
        )
    return train


def get_testset(dataset,intensity_rescale) -> torch.utils.data.DataLoader:
    mem_pin = False
    if Constants.USE_CUDA:
        mem_pin = True
    if dataset is Datasets.PittLocalFull:
        batch_sz = 20
        test = torch.utils.data.DataLoader(
            PittLocalFull(
                1,
                None,
                None,
                None,
                intensity_rescale,
                [f'paths/fold-0/data_paths_ws.txt'],
                [f'paths/fold-0/label_paths.txt'],
                [f'paths/fold-0/mask_paths.txt'],
                augment=False),
            batch_size=batch_sz,
            drop_last=False,
            num_workers=0,
            shuffle=True,
            pin_memory=mem_pin
        )

        return test


def load_model(path) -> Wnet.Wnet:
    wnet = torch.load(path)

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
            maximum = torch.max(segments[:, j, :, :])
            minimum = torch.min(segments[:, j, :, :])
            segments[:, j, :, :] = (segments[:, j, :, :] - minimum) / (maximum - minimum)

            plt.imshow(segments[i, j])
            image_path = sample_dir + "/segment_{}.png".format(j)
            plt.savefig(image_path)
            sgm = max_images == j
            plt.imshow(sgm[i].reshape(segments.shape[2], segments.shape[3]), 'gray')
            image_path = sample_dir + "/segment_argmax_{}.png".format(j)
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
