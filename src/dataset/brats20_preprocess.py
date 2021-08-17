import nibabel as nib
import torch
import numpy as np
import os
import pandas as pd
import torchvision.transforms.functional as augmentor
import torchvision.transforms as transformer





def rescale_intensity(x):
    return normalize_quantile(x, 0.99)
    maximum = x.max()
    minimum = x.min()
    return (x - minimum + 0.01) / (maximum - minimum + 0.01)


def normalize_quantile(x, threshold):
    q = torch.quantile(x, threshold)
    mask = x[x <= q]
    return x / max(mask)


def preprocess_data(raw_dataroot, new_dataroot, all_ids_file):
    ids_file = pd.read_csv(all_ids_file)
    ids_path = ids_file['BraTS_2020_subject_ID']
    subjects_name = np.asarray(ids_path).reshape(-1)
    label_paths = [os.path.join(raw_dataroot, str(subj_name) + '/{}_seg.nii.gz'.format(subj_name)) for
                   subj_name in subjects_name]
    flair_paths = [os.path.join(raw_dataroot, str(subj_name) + '/{}_flair.nii.gz'.format(subj_name)) for
                   subj_name in subjects_name]

    t1_paths = [os.path.join(raw_dataroot, str(subj_name) + '/{}_t1.nii.gz'.format(subj_name)) for
                subj_name in subjects_name]

    t2_paths = [os.path.join(raw_dataroot, str(subj_name) + '/{}_t2.nii.gz'.format(subj_name)) for
                subj_name in subjects_name]

    t1ce_paths = [os.path.join(raw_dataroot, str(subj_name) + '/{}_t1ce.nii.gz'.format(subj_name)) for
                  subj_name in subjects_name]

    paths = zip(flair_paths, t1_paths, t2_paths, t1ce_paths, label_paths, subjects_name)

    for path in paths:
        flair_path, t1_path, t2_path, t1ce_path, label_path, subject_name = path
        subject_dir_path = os.path.join(new_dataroot, subject_name)
        os.mkdir(subject_dir_path, 0o777)

        output_shape = (200, 200, 155)
        data_shape = (155, 240, 240)
        X_flair = nib.load(flair_path).get_fdata().astype('float32').reshape(data_shape)
        X_t1 = nib.load(t1_path).get_fdata().astype('float32').reshape(data_shape)
        X_t2 = nib.load(t2_path).get_fdata().astype('float32').reshape(data_shape)
        X_t1ce = nib.load(t1ce_path).get_fdata().astype('float32').reshape(data_shape)
        X_seg = nib.load(t1ce_path).get_fdata().astype('float32').reshape(data_shape)

        X_flair = torch.tensor(X_flair)
        X_t1 = torch.tensor(X_t1)
        X_t2 = torch.tensor(X_t2)
        X_t1ce = torch.tensor(X_t1ce)
        X_seg = torch.tensor(X_seg)

        cropper = transformer.CenterCrop(200)
        X_flair = cropper(X_flair)
        X_t1 = cropper(X_t1)
        X_t2 = cropper(X_t2)
        X_t1ce = cropper(X_t1ce)
        X_seg = cropper(X_seg)

        X_flair = rescale_intensity(X_flair)
        X_t1 = rescale_intensity(X_t1)
        X_t2 = rescale_intensity(X_t2)
        X_t1ce = rescale_intensity(X_t1ce)
        X_seg = rescale_intensity(X_seg)

        X_flair = X_flair.reshape(output_shape)
        X_t1 = X_t1.reshape(output_shape)
        X_t2 = X_t2.reshape(output_shape)
        X_t1ce = X_t1ce.reshape(output_shape)
        X_seg = X_seg .reshape(output_shape)

        flair_out_path = os.path.join(subject_dir_path,'{}_flair.npy'.format(subject_name))
        X_t1_out_path = os.path.join(subject_dir_path, '{}_t1.npy'.format(subject_name))
        X_t2_out_path = os.path.join(subject_dir_path, '{}_t2.npy'.format(subject_name))
        X_t1ce_out_path = os.path.join(subject_dir_path, '{}_t1ce.npy'.format(subject_name))
        X_seg_out_path = os.path.join(subject_dir_path, '{}_seg.npy'.format(subject_name))

        X_flair = np.array(X_flair)
        X_t1 = np.array(X_t1)
        X_t2 = np.array(X_t2)
        X_t1ce = np.array(X_t1ce)
        X_seg = np.array(X_seg)

        np.save(flair_out_path, X_flair)
        np.save(X_t1_out_path, X_t1)
        np.save(X_t2_out_path, X_t2)
        np.save(X_t1ce_out_path, X_t1ce)
        np.save(X_seg_out_path, X_seg)




raw_dataroot = ''
new_dataroot = ''
all_ids_file = ''
preprocess_data(raw_dataroot, new_dataroot, all_ids_file)