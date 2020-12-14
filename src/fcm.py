import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from utils import utils
import torch

import sys

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')


def fcm_WMH_segmentation(data, nclusters, csf_background_threshold, binwidth):
    if type(data) == torch.Tensor:
        data = data.numpy()

    shape = data.shape
    data = data.reshape(-1)
    data = data.reshape(1, len(data))

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, nclusters, 2, error=0.005, maxiter=1000, init=None)

    csf_background_indx = np.argmax(u[:, 0])
    csf_background = u[csf_background_indx, :] >= csf_background_threshold
    csf_background = csf_background.reshape(data.shape)
    csf_background_mask = data[csf_background]

    freq1, intensity1 = np.histogram(csf_background_mask,
                                    bins=np.arange(csf_background_mask.min(), csf_background_mask.max() + binwidth,
                                                   binwidth))
    data = data.reshape(-1)
    freq2, intensity2 = np.histogram(data,
                                    bins=np.arange(data.min(), data.max() + binwidth, binwidth))

    n = len(freq1) - 1

    while (freq1[n] == freq2[n]):
        n -= 1
    intensity_threshold = intensity1[n]

    img = data.reshape(shape)
    wmh_segments = img >= intensity_threshold

    return wmh_segments


def fpm():
    None


def removing_hyper_intense():
    None


if __name__ == '__main__':
    utils.Constants.FCM = True
    dataset = utils.Constants.Datasets.PittLocalFull
    trainset = utils.get_trainset(dataset, False)
    for i in range(0,10):
        j = 0
        for batch in trainset:
            b = batch['data']
            j +=1
            # for e in b:
            #     print("here")
            #     example = e[0].numpy()
            #     # plt.imshow(example)
            #     # plt.show()
            #     wmh_seg = fcm_WMH_segmentation(example, 2, 0.05, 1)
            #     wmh_seg = wmh_seg.astype('uint8')
            #     plt.imshow(wmh_seg)
            #     plt.show()
            #     print("nooo")
