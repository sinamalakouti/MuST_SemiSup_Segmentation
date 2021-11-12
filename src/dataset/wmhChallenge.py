from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import copy
import SimpleITK as sitk
import scipy
from pathlib import Path
import os
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as augmentor
import scipy.ndimage

from torchvision.utils import save_image
'''
Code adapted from https://github.com/FourierX9/wmh_ibbmTum/blob/master/test_leave_one_out.py
'''

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70  # to mask the brain
smooth = 1.
thresh_T1 = 30
#
WMH_SEG = ['Utrecht', 'GE3T', 'Singapore']
DOMAINS2IDS = {'Utrecht': 1,
               'GE3T': 2,
               'Singapore': 0}
IDS2DOMAINS = {1: 'Utrecht',
               2: 'GE3T',
               0: 'Singapore'}
DATASETS = {'WMH_SEG': WMH_SEG}
DOMAINS = {'WMH_SEG': 3}


def tensorize(*args):
    return tuple(torch.Tensor(arg).float().unsqueeze(0) for arg in args)


def Utrecht_preprocessing(img, label, T1=None, train=True, whitestripe=False):
    num_selected_slice = np.shape(img)[0]
    img_rows_dataset = np.shape(img)[1]
    img_cols_dataset = np.shape(img)[2]
    if T1 is not None:
        T1_img = np.float32(T1)
        brain_mask_T1 = np.ndarray(
            (num_selected_slice, img_rows_dataset, img_cols_dataset),
            dtype=np.float32)

    brain_mask = np.ndarray(
        (num_selected_slice, img_rows_dataset, img_cols_dataset),
        dtype=np.float32)

    # FLAIR -------------------------------------------
    brain_mask[img >= thresh_FLAIR] = 1
    brain_mask[img < thresh_FLAIR] = 0
    for i in range(np.shape(img)[0]):
        brain_mask[i, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            brain_mask[i, :, :])

    img = img[:, (img_rows_dataset // 2 -
                  rows_standard // 2):(img_rows_dataset // 2 +
                                       rows_standard // 2),
              (img_cols_dataset // 2 -
               cols_standard // 2):(img_cols_dataset // 2 +
                                    cols_standard // 2)]
    brain_mask = brain_mask[:, (img_rows_dataset // 2 -
                                rows_standard // 2):(img_rows_dataset // 2 +
                                                     rows_standard // 2),
                            (img_cols_dataset // 2 -
                             cols_standard // 2):(img_cols_dataset // 2 +
                                                  cols_standard // 2)]
    label = label[:, (img_rows_dataset // 2 -
                      rows_standard // 2):(img_rows_dataset // 2 +
                                           rows_standard // 2),
                  (img_cols_dataset // 2 -
                   cols_standard // 2):(img_cols_dataset // 2 +
                                        cols_standard // 2)]
    # Gaussian Normalization -------------------------
    if not whitestripe:
        img -= np.mean(img[brain_mask == 1])
        img /= np.std(img[brain_mask == 1])

    # T1
    if T1 is not None:
        brain_mask_T1[T1_img >= thresh_T1] = 1
        brain_mask_T1[T1_img < thresh_T1] = 0
        for i in range(np.shape(T1_img)[0]):
            brain_mask_T1[
                i, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                    brain_mask_T1[i, :, :])  # fill the holes inside brain
        T1_img = T1_img[:, (img_rows_dataset // 2 -
                            rows_standard // 2):(img_rows_dataset // 2 +
                                                 rows_standard // 2),
                        (img_cols_dataset // 2 -
                         cols_standard // 2):(img_cols_dataset // 2 +
                                              cols_standard // 2)]
        brain_mask_T1 = brain_mask_T1[:, (
            img_rows_dataset // 2 -
            rows_standard // 2):(img_rows_dataset // 2 + rows_standard // 2), (
                img_cols_dataset // 2 -
                cols_standard // 2):(img_cols_dataset // 2 +
                                     cols_standard // 2)]
        # Gaussian Normalization ----
        if not whitestripe:
            T1_img -= np.mean(T1_img[brain_mask_T1 == 1])
            T1_img /= np.std(T1_img[brain_mask_T1 == 1])
        if train:
            T1_img = T1_img[5:43, :, :]

    # extrace slices (reduce first and last few of slices)
    if train:
        img = img[5:43, :, :]
        brain_mask = brain_mask[5:43, :, :]
        label = label[5:43, :, :]
    if T1 is not None:
        return img, label, brain_mask, T1_img
    else:
        return img, label, brain_mask


def GE3T_preprocessing(img, label, T1=None, train=True, whitestripe=False):
    start_cut = 46
    num_selected_slice = np.shape(img)[0]
    img_rows_dataset = np.shape(img)[1]
    img_cols_dataset = np.shape(img)[2]
    img = np.float32(img)
    docker_setups = False
    if T1 is not None:
        T1_img = np.float32(T1)
        brain_mask_T1 = np.ndarray(
            (num_selected_slice, img_rows_dataset, img_cols_dataset),
            dtype=np.float32)
        T1_img_resize = np.ndarray(
            (num_selected_slice, rows_standard, cols_standard),
            dtype=np.float32)
        imgs_two_channels = np.ndarray(
            (num_selected_slice, rows_standard, cols_standard, 2),
            dtype=np.float32)

    brain_mask = np.ndarray(
        (num_selected_slice, img_rows_dataset, img_cols_dataset),
        dtype=np.float32)
    img_resize = np.ndarray((num_selected_slice, rows_standard, cols_standard),
                            dtype=np.float32)
    brain_mask_resize = np.ndarray(
        (num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    label_resize = np.ndarray(
        (num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR ---------------------------------------
    brain_mask[img >= thresh_FLAIR] = 1
    brain_mask[img < thresh_FLAIR] = 0

    for i in range(np.shape(img)[0]):
        brain_mask[i, :, :] = scipy.ndimage.morphology.binary_fill_holes(
            brain_mask[i, :, :])

    # Gaussian Normalization -------------------
    if not whitestripe:
        img -= np.mean(img[brain_mask == 1])
        img /= np.std(img[brain_mask == 1])
    if not docker_setups:
        img_resize[...] = np.min(img)
        img_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 +
            img_cols_dataset // 2)] = img[:, start_cut:start_cut +
                                          rows_standard, :]
        brain_mask_resize[...] = np.min(brain_mask)
        brain_mask_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 +
            img_cols_dataset // 2)] = brain_mask[:, start_cut:start_cut +
                                                 rows_standard, :]

        label_resize[...] = np.min(label)
        label_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 +
            img_cols_dataset // 2)] = label[:, start_cut:start_cut +
                                            rows_standard, :]
    else:
        img_resize[...] = np.min(img)
        img_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 + img_cols_dataset // 2)] = img[:, (
                img_rows_dataset // 2 -
                rows_standard // 2):(img_rows_dataset // 2 +
                                     rows_standard // 2), :]
        brain_mask_resize[...] = np.min(brain_mask)
        brain_mask_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 + img_cols_dataset // 2)] = brain_mask[:, (
                img_rows_dataset // 2 -
                rows_standard // 2):(img_rows_dataset // 2 +
                                     rows_standard // 2), :]
        label_resize[...] = np.min(label)
        label_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 + img_cols_dataset // 2)] = label[:, (
                img_rows_dataset // 2 -
                rows_standard // 2):(img_rows_dataset // 2 +
                                     rows_standard // 2), :]
    # T1 --------------------------------------
    if T1 is not None:
        brain_mask_T1[T1_img >= thresh_T1] = 1
        brain_mask_T1[T1_img < thresh_T1] = 0

        for i in range(np.shape(T1_img)[0]):
            brain_mask_T1[
                i, :, :] = scipy.ndimage.morphology.binary_fill_holes(
                    brain_mask_T1[i, :, :])
        # Gaussian Normalization
        if not whitestripe:
            T1_img -= np.mean(T1_img[brain_mask_T1 == 1])
            T1_img /= np.std(T1_img[brain_mask_T1 == 1])
        T1_img_resize[...] = np.min(T1_img)
        T1_img_resize[:, :, (cols_standard // 2 - img_cols_dataset // 2):(
            cols_standard // 2 +
            img_cols_dataset // 2)] = T1_img[:, start_cut:start_cut +
                                             rows_standard, :]
        if train:
            T1_img_resize = T1_img_resize[12:75, :, :]

    # extract slices
    if train:
        img_resize = img_resize[12:75, :, :]
        label_resize = label_resize[12:75, :, :]
        brain_mask_resize = brain_mask_resize[12:75, :, :]
    if T1 is not None:
        return img_resize, label_resize, brain_mask_resize, T1_img_resize
    else:
        return img_resize, label_resize, brain_mask_resize


class LazyLoader:
    def __init__(self, initializer, *args, **kwargs):
        self.initializer = initializer
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.initializer(*self.args, **self.kwargs)


class ImageSet(torch.utils.data.Dataset):
    def __init__(self,
                 path_f,
                 parent_dir='',
                 domain='',
                 T1=False,
                 train=True,
                 whitestripe=False):
        lines = [p.strip().split() for p in open(path_f, 'r')]
        self.paths = [f'{parent_dir}{path}' for path, _ in lines]
        self.data = []
        self.domain = domain
        for img_subj_pth in self.paths:
            if not whitestripe:
                img = sitk.ReadImage('./data/wmh_challenge/' + img_subj_pth + '/pre/FLAIR.nii.gz')
            else:
                img = sitk.ReadImage('./data/wmh_challenge/' +img_subj_pth +
                                     '/pre/FLAIR_ws_masked.nii.gz')
            img = sitk.GetArrayFromImage(img)

            if T1:
                if not whitestripe:
                    T1_img = sitk.ReadImage('./data/wmh_challenge/' +img_subj_pth + '/pre/T1.nii.gz')
                else:
                    T1_img = sitk.ReadImage('./data/wmh_challenge/' +img_subj_pth +
                                            '/pre/T1_ws_masked.nii.gz')
                T1_img = sitk.GetArrayFromImage(T1_img)
            else:
                T1_img = None
            try:
                label = sitk.ReadImage('./data/wmh_challenge/' +img_subj_pth + '/wmh.nii.gz')
            except:
                label = sitk.ReadImage('./data/wmh_challenge/' +img_subj_pth + '/wmh.nii')
            label = sitk.GetArrayFromImage(label)
            if self.domain == 'Local':
                label = label[(24, 25, 26, 27, 28), :, :]
            if self.domain != 'GE3T':
                if T1:
                    img, label, mask, T1_img = Utrecht_preprocessing(
                        img,
                        label,
                        T1_img,
                        train=train,
                        whitestripe=whitestripe)
                else:
                    img, label, mask = Utrecht_preprocessing(
                        img, label, train=train, whitestripe=whitestripe)

                for i in range(0, img.shape[0]):
                    self.data.append({
                        'img':
                        img[i, :, :],
                        'label':
                        label[i, :, :],
                        'mask':
                        mask[i, :, :],
                        'domain_s':
                        DOMAINS2IDS[self.domain]
#                        0 if self.domain == 'Singapore' else 1
                    })
                    if T1:
                        self.data[-1]['T1_img'] = T1_img[i, :, :]
            else:
                if T1:
                    img, label, mask, T1_img = GE3T_preprocessing(
                        img,
                        label,
                        T1_img,
                        train=train,
                        whitestripe=whitestripe)
                else:
                    img, label, mask = GE3T_preprocessing(
                        img, label, train=train, whitestripe=whitestripe)

                for i in range(0, img.shape[0]):
                    self.data.append({
                        'img': img[i, :, :],
                        'label': label[i, :, :],
                        'mask': mask[i, :, :],
                        'domain_s': DOMAINS2IDS[self.domain]
                    })
                    if T1:
                        self.data[-1]['T1_img'] = T1_img[i, :, :]
            print(f'finish: {img_subj_pth}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Mixed(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.domains = [domain() for domain in args]
        self.lengths = [len(d) for d in self.domains]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        for d, domain in enumerate(self.domains):
            if index >= len(domain):
                index -= len(domain)
            else:
                x = domain[index]
                # x['domain'] = d
                return x


class WmhChallenge(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 base_and_aug=True,
                 do_aug=True,
                 intensity_rescale=None):
        self.dataset = dataset
        self.base_and_aug = base_and_aug
        self.do_aug = do_aug
        self.intensity_rescale = intensity_rescale

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset[index]
        if 'T1_img' in x:
            if self.do_aug:
                if self.base_and_aug:
                    # x['img'], x['label'], x['mask'], x['T1_img'] = self.aug(
                    #     x['img'], x['label'], x['mask'], T1=x['T1_img'])
                    img_aug, label_aug, mask_aug, t1_aug = self.aug(
                        x['img'], x['label'], x['mask'], T1=x['T1_img'])
                    img_baseline, label_baseline, mask_baseline, t1_baseline = baseline(
                        x['img'], x['label'], x['mask'], T1=x['T1_img'])
                    img_aug = np.concatenate((img_aug, t1_aug), axis=0)
                    img_baseline = np.concatenate((img_baseline, t1_baseline),
                                                  axis=0)
                    # x['img'] = np.stack((img_baseline, img_aug), axis=0)
                    # x['label'] = np.stack((label_baseline, label_aug), axis=0)
                    # x['mask'] = np.stack((mask_baseline, mask_aug), axis=0)
                    # x['T1_img'] = t1_aug
                    x['img'] = img_aug
                    x['label'] = label_aug
                    x['mask'] = mask_aug
                    x['T1_img'] = t1_aug
                    x['img_base'] = img_baseline
                    x['label_base'] = label_baseline
                    x['mask_base'] = mask_baseline

                else:
                    x['img'], x['label'], x['mask'], x['T1_img'] = self.aug(
                        x['img'], x['label'], x['mask'], T1=x['T1_img'])
                    x['img'] = np.concatenate((x['img'], x['T1_img']), axis=0)
            else:
                x['img'], x['label'], x['mask'], x['T1_img'] = baseline(
                    x['img'], x['label'], x['mask'], T1=x['T1_img'])
                x['img'] = np.concatenate((x['img'], x['T1_img']), axis=0)
            # x['img'] = np.concatenate((x['img'], x['T1_img']), axis=0)

        else:
            if self.do_aug:
                if self.base_and_aug:
                    # x['img'], x['label'], x['mask'] = self.aug(
                    #     x['img'], x['label'], x['mask'])
                    img_aug, label_aug, mask_aug = self.aug(
                        x['img'], x['label'], x['mask'])
                    img_baseline, label_baseline, mask_baseline = baseline(
                        x['img'], x['label'], x['mask'])
                    # x['img'] = np.stack((img_baseline, img_aug), axis=0)
                    # x['label'] = np.stack((label_baseline, label_aug), axis=0)
                    # x['mask'] = np.stack((mask_baseline, mask_aug), axis=0)
                    x['img'] = img_aug
                    x['label'] = label_aug
                    x['mask'] = mask_aug
                    x['img_base'] = img_baseline
                    x['label_base'] = label_baseline
                    x['mask_base'] = mask_baseline
                else:
                    x['img'], x['label'], x['mask'] = self.aug(
                        x['img'], x['label'], x['mask'])
            else:
                x['img'], x['label'], x['mask'] = baseline(
                    x['img'], x['label'], x['mask'])
        return x

    def intensity_rescaling(self, img, c_factor):
        img = img - img.mean()
        img = np.multiply(img, c_factor)
        img = img + img.mean()
        return img

    def aug(self, img, label, mask, T1=None):
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(.9, 1.1)
        shear = np.random.uniform(-18, 18)

        if self.intensity_rescale is not None:
            c_factor = np.random.uniform(
                1 - self.intensity_rescale,
                1 + self.intensity_rescale)  # contrast rescale factor
            img = self.intensity_rescaling(img, c_factor)
            if T1 is not None:
                T1 = self.intensity_rescaling(T1, c_factor)

        img = augmentor.to_pil_image(img, mode='F')

        # label[label == 2] = 0
        label = augmentor.to_pil_image(label, mode='F')
        mask = augmentor.to_pil_image(mask, mode='F')
        if T1 is not None:
            T1_img = augmentor.to_pil_image(T1, mode='F')
            T1_img = augmentor.affine(T1_img,
                                      angle=angle,
                                      translate=(0, 0),
                                      shear=shear,
                                      scale=scale)
            T1_img = augmentor.to_tensor(T1_img).float()

        img = augmentor.affine(img,
                               angle=angle,
                               translate=(0, 0),
                               shear=shear,
                               scale=scale)
        label = augmentor.affine(label,
                                 angle=angle,
                                 translate=(0, 0),
                                 shear=shear,
                                 scale=scale)
        mask = augmentor.affine(mask,
                                angle=angle,
                                translate=(0, 0),
                                shear=shear,
                                scale=scale)

        img = augmentor.to_tensor(img).float()
        label = augmentor.to_tensor(label).float()
        label = (label > 0).float()
        mask = augmentor.to_tensor(mask).float()
        mask = (mask > 0).float()
        if T1 is not None:
            return img, label, mask, T1_img
        else:
            return img, label, mask


def baseline(img, label, mask, T1=None):
    img = augmentor.to_pil_image(img, mode='F')
    if label.dtype != np.float32:
        label = label.astype(np.float32)

    # label[label == 2] = 0
    label = augmentor.to_pil_image(label, mode='F')
    mask = augmentor.to_pil_image(mask, mode='F')
    img = augmentor.to_tensor(img).float()
    label = augmentor.to_tensor(label).float()
    mask = augmentor.to_tensor(mask).float()
    if T1 is not None:
        T1_img = augmentor.to_pil_image(T1, mode='F')
        T1_img = augmentor.to_tensor(T1_img).float()
        return img, label, mask, T1_img
    else:
        return img, label, mask


def tensorize(x):
    x = augmentor.to_pil_image(x, mode='F')
    x = augmentor.to_tensor(x).float()
    return x

if __name__ == '__main__':
    print("here")
    from dataset.wmh_utils import get_splits

    splits, num_domains = get_splits(
        'WMH_SEG',  # get data of different domains
        T1=True,
        whitestripe=False,
        supRatio=5,
        seed=41,
        experiment_mode='semi_alternate')
    trainset_sup = splits['train_sup']()
    #trainset_unsup =splits['train_unsup']()
    # testset = splits['Singapore']['test']()
    valset = splits['val']()
    print(len(trainset_sup))
    trainset = WmhChallenge(trainset_sup,
                            base_and_aug=False)

    train = torch.utils.data.DataLoader(trainset,
                                        batch_size=16,
                                        drop_last=True,
                                        num_workers=4,
                                        shuffle=True)
    it = iter(train)
    # print(len(trainset_unsup))
    print(len(valset))

    print("dfasfs")