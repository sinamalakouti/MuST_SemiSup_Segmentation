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


# return brats2020 ids that were in brats 2019
def id19_to_id20(name_map):
    ids = []
    for id_2020 in name_map:
        if name_map[id_2020]['brats19'] is not np.nan:
            ids.append(id_2020)
    return ids


# return brats2020 ids that were in brats2018
def id18_to_id20(name_map):
    ids = []
    for id_2020 in name_map:
        if name_map[id_2020]['brats18'] is not np.nan:
            ids.append(id_2020)
    return ids


def generate_train_val_test_2018(all_data_csv, root_dir, val_size=50, test_size=50):
    name_map = get_name_mapping(all_data_csv)
    all_ids = id18_to_id20(name_map)
    val_test_ids = np.random.choice(all_ids, val_size + test_size, replace=False)
    tr_ids = np.setdiff1d(all_ids, val_test_ids)
    test_ids = np.random.choice(val_test_ids, test_size, replace=False)
    val_ids = np.setdiff1d(val_test_ids, test_ids)
    np.savetxt(root_dir + "only2018/some_train2018.csv", tr_ids.astype(np.str), delimiter=',', fmt='%s')
    np.savetxt(root_dir + "only2018/val2018_ids.csv", val_ids.astype(np.str), delimiter=',', fmt='%s')
    np.savetxt(root_dir + "only2018/test2018_ids.csv", test_ids.astype(np.str), delimiter=',', fmt='%s')


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

    def __init__(self, dataroot_dir, mode, min_slice_index, max_slice_index,
                 augment=False, intensity_aug=None, center_cropping=False, t1=None, t2=None, t1ce=None):
        super(Brat20, self).__init__()

        self.augment = augment
        self.intensity_aug = intensity_aug
        self.center_cropping = center_cropping
        self.weights = {}
        self.t1 = t1
        self.t2 = t2
        self.t1ce = t1ce

        if mode == "train2020_sup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats20_training_ids.csv')
        elif mode == "train2020_semi_sup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats20_training_sup_ids.csv')
        elif mode == "train2020_semi_unsup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats20_training_unsup_ids.csv')
        elif mode == "val2020":  # validation
            ids_path = os.path.join(dataroot_dir, 'valset/brats20_val_ids.csv')
        elif mode == "all_train2018_sup":
            ids_path = os.path.join(dataroot_dir, 'trainset/brats2018.csv')
        elif mode == "only_train2018_sup":
            ids_path = os.path.join(dataroot_dir, 'only2018/some_train2018_ids.csv')
        elif mode == "only_test2018":
            ids_path = os.path.join(dataroot_dir, 'only2018/test2018_ids.csv')
        elif mode == "only_val2018":
            ids_path = os.path.join(dataroot_dir, 'only2018/val2018_ids.csv')
        elif mode == "train2018_semi_sup":
            None  # todo
        elif mode == "train2018_semi_unsup":
            None  # todo
        elif mode == "test2019_new":
            ids_path = os.path.join(dataroot_dir, 'valset/brats2019_new.csv')

        subjects_root_dir = os.path.join(dataroot_dir, 'MICCAI_BraTS2020_TrainingData')
        self.subjects_name = np.asarray(pd.read_csv(ids_path, header=None)).reshape(-1)

        label_paths = [os.path.join(subjects_root_dir, str(subj_name) + '/{}_seg.nii.gz'.format(subj_name)) for
                       subj_name in self.subjects_name]
        flair_paths = [os.path.join(subjects_root_dir, str(subj_name) + '/{}_flair.nii.gz'.format(subj_name)) for
                       subj_name in self.subjects_name]
        if self.t1:
            t1_paths = [os.path.join(subjects_root_dir, str(subj_name) + '/{}_t1.nii.gz'.format(subj_name)) for
                        subj_name in self.subjects_name]
        else:
            t1_paths = [None for _ in self.subjects_name]
        if self.t2:
            t2_paths = [os.path.join(subjects_root_dir, str(subj_name) + '/{}_t2.nii.gz'.format(subj_name)) for
                        subj_name in self.subjects_name]
        else:
            t2_paths = [None for _ in self.subjects_name]

        if self.t1ce:
            t1ce_paths = [os.path.join(subjects_root_dir, str(subj_name) + '/{}_t1ce.nii.gz'.format(subj_name)) for
                          subj_name in self.subjects_name]
        else:
            t1ce_paths = [None for _ in self.subjects_name]
        paths = zip(flair_paths, t1_paths, t2_paths, t1ce_paths, label_paths, self.subjects_name)

        self.data = []
        for data, t1_data, t2_data, t1ce_data, label, subject_name in paths:

            X = self._extract(data, slices=list(range(min_slice_index, max_slice_index)))
            Y = self._extract(label, slices=list(range(min_slice_index, max_slice_index)))
            if self.t1:
                X_t1 = self._extract(t1_data, slices=list(range(min_slice_index, max_slice_index)))
            if self.t2:
                X_t2 = self._extract(t2_data, slices=list(range(min_slice_index, max_slice_index)))
            if self.t1ce:
                X_t1ce = self._extract(t1ce_data, slices=list(range(min_slice_index, max_slice_index)))
            subject_id = int(subject_name.split('_')[-1])
            for sl in range(Y.shape[2]):
                if X[:, :, sl].sum() == 0 or np.sum((X[:, :, sl] > 0)) / (240 * 240) * 100 < 10:
                    None
                else:
                    data_map = {'data': X[:, :, sl],
                                'label': Y[:, :, sl],
                                'subject_id': subject_id,
                                'slice': sl + min_slice_index
                                }
                    if self.t1:
                        data_map['data_t1'] = X_t1[:, :, sl]
                    else:
                        data_map['data_t1'] = None
                    if self.t2:
                        data_map['data_t2'] = X_t2[:, :, sl]
                    else:
                        data_map['data_t2'] = None
                    if self.t1ce:
                        data_map['data_t1ce'] = X_t1ce[:, :, sl]
                    else:
                        data_map['data_t1ce'] = None

                    self.data.append(data_map)

    def __len__(self):
        return len(self.data)

    def _extract(self, f, slices=(24, 25, 26, 27, 28)):
        x = nib.load(f).get_data()
        slices = np.array(slices)
        return x[:, :, slices].astype('float32')

    def __getitem__(self, index):

        x = self.data[index]['data']
        x_t1 = self.data[index]['data_t1']
        x_t2 = self.data[index]['data_t2']
        x_t1ce = self.data[index]['data_t1ce']
        y = self.data[index]['label']

        x, x_t1, x_t2, x_t1ce, y = tensorize(x, x_t1, x_t2, x_t1ce, y)
        if self.center_cropping:
            x, x_t1, x_t2, x_t1ce, y = center_crop(x, x_t1, x_t2, x_t1ce, y)

        # only Whole Tumor (WT) segmentation
        # y[y >= 1] = 1

        y[y == 4] = 3  # for simplicity in training, substitute label = 3 with 4
        # rand = np.random.rand(1)
        if self.augment:  # and rand < 0.5:

            x, x_t1, x_t2, x_t1ce, y = augment(
                x=x, y=y, t1=x_t1, t2=x_t2, t1ce=x_t1ce, intensity_aug=self.intensity_aug)
        else:
            x, x_t1, x_t2, x_t1ce, y = tensorize(x, x_t1, x_t2, x_t1ce, y)

        x = rescale_intensity(x)
        x_t1 = rescale_intensity(x_t1) if x_t1 is not None else None
        x_t2 = rescale_intensity(x_t2) if x_t2 is not None else None
        x_t1ce = rescale_intensity(x_t1ce) if x_t1ce is not None else None

        # result = {'data': x, 'label': y, 'subject': self.subjects_id[index]}
        data_modalities = []
        if x is not None:
            data_modalities.append(x)
        if x_t1 is not None:
            # result['data_t1'] = x_t1
            data_modalities.append(x_t1)
        if x_t2 is not None:
            # result['data_t2'] = x_t2
            data_modalities.append(x_t2)
        if x_t1ce is not None:
            # result['data_t1ce'] = x_t1ce
            data_modalities.append(x_t1ce)

        x_final = torch.cat(data_modalities, dim=0)
        result = {'data': x_final, 'label': y, 'subject': self.data[index]['subject_id'],
                  'slice': self.data[index]['slice']}
        return result


def augment(x, y, m=False, t1=False, t2=False, t1ce=False, intensity_aug=False):
    # NOTE: method expects numpy float arrays
    # to_pil_image makes assumptions based on input when mode = None
    # i.e. it should infer that mode = 'F'
    # manually setting mode to 'F' in this function

    angle = np.random.uniform(-180, 180)
    scale = np.random.uniform(.8, 1.2)
    shear = np.random.uniform(-30, 30)
    c_factor = np.random.uniform(.5, 1.5)  # contrast factor
    # c_factor = 1.2     ' # contrast factor

    ori_x = x
    ori_t1 = None
    if intensity_aug:
        x = adjust_contrast(x, c_factor)

    x = augmentor.to_pil_image(x, mode='F')
    y = augmentor.to_pil_image(y, mode='F')

    if m:
        m = augmentor.to_pil_image(m, mode='F')
    if t1:
        ori_t1 = t1
        if intensity_aug:
            t1 = adjust_contrast(t1, c_factor)
        t1 = augmentor.to_pil_image(t1, mode='F')
    if t2:
        if intensity_aug:
            t2 = adjust_contrast(t2, c_factor)
        t2 = augmentor.to_pil_image(t2, mode='F')
    if t1ce:
        if intensity_aug:
            t1ce = adjust_contrast(t1ce, c_factor)
        t1ce = augmentor.to_pil_image(t1ce, mode='F')

    x = augmentor.affine(x,
                         angle=angle, translate=(0, 0), shear=shear, scale=scale)
    y = augmentor.affine(y,
                         angle=angle, translate=(0, 0), shear=shear, scale=scale)
    if m:
        m = augmentor.affine(m,
                             angle=angle, translate=(0, 0), shear=shear, scale=scale)
    if t1:
        t1 = augmentor.affine(t1,
                              angle=angle, translate=(0, 0), shear=shear, scale=scale)
        t1 = augmentor.to_tensor(t1).float()
    if t2:
        t2 = augmentor.affine(t2,
                              angle=angle, translate=(0, 0), shear=shear, scale=scale)
        t2 = augmentor.to_tensor(t2).float()
    if t1ce:
        t1ce = augmentor.affine(t1ce,
                                angle=angle, translate=(0, 0), shear=shear, scale=scale)
        t1ce = augmentor.to_tensor(t1ce).float()

    x = augmentor.to_tensor(x).float()
    y = augmentor.to_tensor(y).float()

    if m:
        m = augmentor.to_tensor(m).float()
        m = (m > 0).float()

    return x, t1, t2, t1ce, y


def adjust_contrast(x, c_factor):
    x_np = x - x.mean()
    x_np = np.multiply(x_np, c_factor)
    x_np = x_np + x.mean()
    return x_np


def tensorize(*args):
    return tuple(torch.Tensor(arg).float().unsqueeze(0) if arg is not None else None for arg in args)


def center_crop(x, x_t1, x_t2, x_t1ce, y, size=200):
    cropper = transformer.CenterCrop(size)
    x_cropped = cropper(x).reshape((size, size))
    x_t1_cropped = cropper(x_t1).reshape((size, size)) if x_t1 is not None else None
    x_t2_cropped = cropper(x_t2).reshape((size, size)) if x_t2 is not None else None
    x_t1ce_cropped = cropper(x_t1ce).reshape((size, size)) if x_t1ce is not None else None
    y_cropped = cropper(y).reshape((size, size))

    assert y.sum() == y_cropped.sum(), "cropped label part!!!!"
    return x_cropped, x_t1_cropped, x_t2_cropped, x_t1ce_cropped, y_cropped


def rescale_intensity(x):
    return normalize_quantile(x, 0.99)
    maximum = x.max()
    minimum = x.min()
    return (x - minimum + 0.01) / (maximum - minimum + 0.01)


def normalize_quantile(x, threshold):
    q = torch.quantile(x, threshold)
    mask = x[x <= q]
    return x / max(mask)
