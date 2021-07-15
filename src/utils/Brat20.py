import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as augmentor

import os
import nibabel as nib


def train_val_split(all_data_csv, train_dir_path, val_dir_path, val_size=69):
    df = pd.read_csv(all_data_csv)
    all_ids = df.BraTS_2020_subject_ID
    val_ids = np.random.choice(all_ids, val_size, replace=False)
    tr_ids = np.setdiff1d(all_ids, val_ids)
    print("here")
    np.savetxt(train_dir_path + "/training_ids.csv", tr_ids.astype(np.str), delimiter=',', fmt='%s')
    np.savetxt(val_dir_path + "/val_ids.csv", val_ids.astype(np.str), delimiter=',', fmt='%s')


class Brat20(torch.utils.data.Dataset):

    def __init__(self, dataroot_dir, mode, min_slice_index, max_slice_index, augment=False, intensity_aug=False):
        super(Brat20, self).__init__()

        # NOTE: if dataloader does not shuffle
        # and batch size is kept to 60, then
        # each batch equates to a single subject
        self.augment = augment
        self.intensity_aug = intensity_aug
        self.weights = {}
        if mode == 'train':
            ids_path = os.path.join(dataroot_dir, 'trainset/training_ids.csv')
        else:
            ids_path = os.path.join(dataroot_dir, 'valset/val_ids.csv')

        if mode == 'train':
            subjects_root_dir = os.path.join(dataroot_dir, 'MICCAI_BraTS2020_TrainingData')
        else:
            subjects_root_dir = os.path.join(dataroot_dir, 'MICCAI_BraTS2020_TrainingData')

        self.subjects_name = np.asarray(pd.read_csv(ids_path, header=None)).reshape(-1)
        self.subjects_id = [int(subject.strip('-')[-1]) for subject in self.subjects_name
                            for _ in range(max_slice_index - min_slice_index)]

        flair_paths = [os.path.join(subjects_root_dir, str(subj_name) + '/{}_flair.nii.gz'.format(subj_name)) for
                       subj_name in self.subjects_name]
        label_paths = [os.path.join(subjects_root_dir, str(subj_name) + '/{}_seg.nii.gz'.format(subj_name)) for
                       subj_name in self.subjects_name]

        paths = zip(flair_paths, label_paths)
        self.data = []
        for data, label in paths:

            X = self._extract(data, slices=list(range(min_slice_index, max_slice_index)))
            Y = self._extract(label, slices=list(range(min_slice_index, max_slice_index)))

            for sl in range(Y.shape[2]):
                self.data.append({
                    'data': X[:, :, sl],
                    'label': Y[:, :, sl]})

    def __len__(self):
        return len(self.data)

    def _extract(self, f, slices=(24, 25, 26, 27, 28)):
        x = nib.load(f).get_data()
        slices = np.array(slices)
        return x[:, :, slices].astype('float32')

    def __getitem__(self, index):

        x = self.data[index]['data']
        y = self.data[index]['label']
        # only Whole Tumor (WT) segmentation
        y[y >= 1] = 1

        if self.augment:
            None
            x, y, m = augment(
                x=x, y=y, intensity_aug=self.intensity_aug)
        else:
            x, y = tensorize(x, y)

        x = rescale_intensity(x)

        return {'data': x, 'label': y, 'subject': self.subjects_id[index]}



def augment(x, y, m=None, t1=None, intensity_aug=None):
    # NOTE: method expects numpy float arrays
    # to_pil_image makes assumptions based on input when mode = None
    # i.e. it should infer that mode = 'F'
    # manually setting mode to 'F' in this function

    # print(x.shape); exit()
    # NOTE: accepts np.ndarray of size H x W x C
    # x.shape = 64x64
    # torch implicitly expands last dim as below:
    # elif pic.ndim == 2:
    # if 2D image, add channel dimension (HWC)
    # pic = np.expand_dims(pic, 2)
    # BUT!!!!!!!
    # if x was a tensor this would be different:
    # elif pic.ndimension() == 2:
    # if 2D image, add channel dimension (CHW)
    # pic = pic.unsqueeze(0)

    angle = np.random.uniform(-180, 180)
    scale = np.random.uniform(.8, 1.2)
    shear = np.random.uniform(-30, 30)
    c_factor = np.random.uniform(.5, 1.5)  # contrast factor
    # c_factor = 1.2     ' # contrast factor

    ori_x = x
    ori_t1 = None
    if intensity_aug is not None:
        x = adjust_contrast(x, c_factor)

    x = augmentor.to_pil_image(x, mode='F')
    y = augmentor.to_pil_image(y, mode='F')

    if m is not None:
        m = augmentor.to_pil_image(m, mode='F')
    if t1 is not None:
        ori_t1 = t1
        if intensity_aug is not None:
            t1 = adjust_contrast(t1, c_factor)
        t1 = augmentor.to_pil_image(t1, mode='F')

    x = augmentor.affine(x,
                         angle=angle, translate=(0, 0), shear=shear, scale=scale)
    y = augmentor.affine(y,
                         angle=angle, translate=(0, 0), shear=shear, scale=scale)
    if m is not None:
        m = augmentor.affine(m,
                             angle=angle, translate=(0, 0), shear=shear, scale=scale)
    if t1 is not None:
        t1 = augmentor.affine(t1,
                              angle=angle, translate=(0, 0), shear=shear, scale=scale)
    x = augmentor.to_tensor(x).float()
    y = augmentor.to_tensor(y).float()
    y = (y > 0).float()

    if m is not None:
        m = augmentor.to_tensor(m).float()
        m = (m > 0).float()

    if t1 is not None:
        t1 = augmentor.to_tensor(t1).float()

    if m is not None and t1 is not None:
        return x, y, m, t1
    elif m is not None and t1 is None:
        return x, y, m
    elif m is None and t1 is not None:
        return x, y, t1
    else:
        return x, y


def adjust_contrast(x, c_factor):
    x_np = x - x.mean()
    x_np = np.multiply(x_np, c_factor)
    x_np = x_np + x.mean()
    return x_np


def tensorize(*args):
    return tuple(torch.Tensor(arg).float().unsqueeze(0) for arg in args)


def rescale_intensity(x):
    return normalize_quantile(x, 0.99)
    maximum = x.max()
    minimum = x.min()
    return (x - minimum + 0.01) / (maximum - minimum + 0.01)


def normalize_quantile(x, threshold):
    q = torch.quantile(x, threshold)
    mask = x[x <= q]
    return x / max(mask)
