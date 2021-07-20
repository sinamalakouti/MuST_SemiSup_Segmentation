import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as augmentor
import torchvision.transforms as transformer
import os
import nibabel as nib


def get_name_mapping(name_mapping_csv_path):
    df = pd.read_csv(name_mapping_csv_path)
    name_map = {}

    for index, row in df.iterrows():
        name_map[row['BraTS_2020_subject_ID']] = {'brats18': row['BraTS_2018_subject_ID'],
                                                  'brats19': row['BraTS_2019_subject_ID']}

    return name_map


def id19_to_id20(name_map):
    ids = []
    for id_2020 in name_map:
        if name_map[id_2020]['brats19'] is not np.nan:
            ids.append(id_2020)
    return ids


def id18_to_id20(name_map):
    ids = []
    for id_2020 in name_map:
        if name_map[id_2020]['brats18'] is not np.nan:
            ids.append(id_2020)
    return ids


# train: whole 2018  and test: new 2019 subjects (5)
def train_val_split(traindata_id, testdata_id, train_dir_path, test_dir_path):
    np.savetxt(train_dir_path + "/brats2018.csv", traindata_id.astype(np.str), delimiter=',', fmt='%s')
    np.savetxt(test_dir_path + "/brats2019_new.csv", testdata_id.astype(np.str), delimiter=',', fmt='%s')


# randomly picking  from the 2020 data
def train_val_split(all_data_csv, train_dir_path, val_dir_path, val_size=69):
    df = pd.read_csv(all_data_csv)
    all_ids = df.BraTS_2020_subject_ID
    val_ids = np.random.choice(all_ids, val_size, replace=False)
    tr_ids = np.setdiff1d(all_ids, val_ids)
    print("here")
    np.savetxt(train_dir_path + "/training_ids.csv", tr_ids.astype(np.str), delimiter=',', fmt='%s')
    np.savetxt(val_dir_path + "/val_ids.csv", val_ids.astype(np.str), delimiter=',', fmt='%s')


def semi_sup_split(train_dir_csv, sup_dir_path, unsup_dir_path, ratio=0.5):
    all_train = np.asarray(pd.read_csv(train_dir_csv, header=None)).reshape(-1)
    unsup_size = int(ratio * len(all_train))
    unsup_ids = np.random.choice(all_train, unsup_size, replace=False)
    sup_ids = np.setdiff1d(all_train, unsup_ids)
    np.savetxt(sup_dir_path + "/train_sup_ids.csv", sup_ids.astype(np.str), delimiter=',', fmt='%s')
    np.savetxt(unsup_dir_path + "/train_unsup_ids.csv", unsup_ids.astype(np.str), delimiter=',', fmt='%s')


class Brat20(torch.utils.data.Dataset):

    def __init__(self, dataroot_dir, mode,
                 min_slice_index, max_slice_index, augment=False, intensity_aug=False, center_cropping=False):
        super(Brat20, self).__init__()

        self.augment = augment
        self.intensity_aug = intensity_aug
        self.center_cropping = center_cropping
        self.weights = {}

        if mode == "train2020_sup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats20_training_ids.csv')
        elif mode == "train2020_semi_sup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats20_training_sup_ids.csv')
        elif mode == "train2020_semi_unsup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats20_training_unsup_ids.csv')
        elif mode == "test2020":  # validation
            ids_path = os.path.join(dataroot_dir, 'valset/brats20_val_ids.csv')
        elif mode == "train2018_sup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats2018.csv')
        elif mode == "train2018_semi_sup":
            None  #todo
        elif mode == "train2018_semi_unsup":
            None  #todo
        elif mode == "test2019_new":
            ids_path = os.path.join(dataroot_dir, 'valset/brats2019_new.csv')

        subjects_root_dir = os.path.join(dataroot_dir, 'MICCAI_BraTS2020_TrainingData')

        # if mode == 'train':
        #     subjects_root_dir = os.path.join(dataroot_dir, 'MICCAI_BraTS2020_TrainingData')

        # else:
        #     subjects_root_dir = os.path.join(dataroot_dir, 'MICCAI_BraTS2020_TrainingData')

        self.subjects_name = np.asarray(pd.read_csv(ids_path, header=None)).reshape(-1)
        self.subjects_id = [int(subject.split('_')[-1]) for subject in self.subjects_name
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
                if Y[:, :, sl].sum() == 0 or X[:, :, sl].sum() == 0 or np.sum((X[:, :, sl] >0))/(240 * 240) * 100 < 20:
                    None
                else:
                    self.data.append({
                        'data': X[:, :, sl],
                        'label': Y[:, :, sl]
                    })

    def __len__(self):
        return len(self.data)

    def _extract(self, f, slices=(24, 25, 26, 27, 28)):
        x = nib.load(f).get_data()
        slices = np.array(slices)
        return x[:, :, slices].astype('float32')

    def __getitem__(self, index):

        x = self.data[index]['data']
        y = self.data[index]['label']
        x, y = tensorize(x, y)
        if self.center_cropping:
            x, y = center_crop(x, y)

        # only Whole Tumor (WT) segmentation
        # y[y >= 1] = 1

        y[y == 4] = 3

        if self.augment:
            x, y, m = augment(
                x=x, y=y, intensity_aug=self.intensity_aug)
        else:
            x, y = tensorize(x, y)

        x = rescale_intensity(x)
        if x.isnan().sum() > 0:
            print("dfafadfads")

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


def center_crop(x, y, size=200):
    cropper = transformer.CenterCrop(size)
    x_cropped = cropper(x).reshape((size, size))
    y_cropped = cropper(y).reshape((size, size))
    assert y.sum() == y_cropped.sum(), "cropped label part!!!!"
    return x_cropped, y_cropped


def rescale_intensity(x):
    return normalize_quantile(x, 0.99)
    maximum = x.max()
    minimum = x.min()
    return (x - minimum + 0.01) / (maximum - minimum + 0.01)


def normalize_quantile(x, threshold):
    q = torch.quantile(x, threshold)
    mask = x[x <= q]
    return x / max(mask)
