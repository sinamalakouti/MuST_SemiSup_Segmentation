import os
from torchvision import transforms
import torch
from .wmhChallenge import LazyLoader, ImageSet, Mixed
import numpy as np
import random
# Utrecht', 'GE3T',
WMH_SEG = ['Utrecht', 'GE3T','Singapore']
DOMAINS2IDS = {'Utrecht': 1,
               'GE3T': 2,
               'Singapore': 0}
IDS2DOMAINS = {1: 'Utrecht',
               2: 'GE3T',
               0: 'Singapore'}
DATASETS = {'WMH_SEG': WMH_SEG}
DOMAINS = {'WMH_SEG': 3}
SPLITS = ['train', 'test', 'val']
f'./data/wmh_challenge/paths/Singapore/train.txt'


def create_semi_sup_split(all_data_path, sup_dir_path, unsup_dir_path, ratio=0.5, seed=None, domain='Singapore'):
    if seed is not None:
        np.random.seed(seed)
    all_size = 48  # 3 datasets together
    lines = [p.strip().split() for p in open(all_data_path, 'r')]
    n_ids = len(lines)
    all_ids = [p + " " + str(id) for p, id in lines]
    print("all ids is ")
    print(all_ids)
    sup_size = np.ceil(ratio * all_size) if np.ceil(ratio * all_size) % 3 == 0 else np.ceil(ratio * all_size) + 1
    sup_size = int(sup_size / 3)
    print("sup size")
    print(sup_size)
    unsup_size = n_ids - sup_size
    np.random.shuffle(all_ids)
    sup_ids = all_ids[0:sup_size]
    unsup_ids = all_ids[sup_size:]
    assert len(sup_ids) == sup_size, "len(sup_ids) != sup_size"
    assert len(unsup_ids) == unsup_size, "len(sup_ids) != sup_size"

    if seed is None:
        joined_string = "\n".join(sup_ids)
        file = open(sup_dir_path + "/train_{}_sup_ids{}.txt".format(domain, int(ratio * 100)), "w")
        file.write(joined_string)
    else:
        joined_string = "\n".join(sup_ids)
        file = open(sup_dir_path + "/train_{}_sup_ids{}_seed{}.txt".format(domain, int(ratio * 100), seed), "w")
        file.write(joined_string)
    if seed is None:
        joined_string = "\n".join(unsup_ids)
        file = open(unsup_dir_path + "/train_{}_unsup_ids{}.txt".format(domain, int(ratio * 100)), "w")
        file.write(joined_string)
    else:
        joined_string = "\n".join(unsup_ids)
        file = open(unsup_dir_path + "/train_{}_unsup_ids{}_seed{}.txt".format(domain, int(ratio * 100), seed), "w")
        file.write(joined_string)


def get_file(supRatio, seed, experiment_mode='semi', isSupervised=True, dset=None):
    if experiment_mode == 'semi' or experiment_mode == 'semi_downSample' or experiment_mode == 'semi_alternate':
        f = 'train_{}_{}_ids{}_seed{}.csv'.format(dset, 'sup' if isSupervised else 'unsup', supRatio, seed)
    elif experiment_mode == 'partially_sup':
        f = 'train_{}_{}_ids{}_seed{}.csv'.format(dset, 'sup', supRatio, seed)
    elif experiment_mode == 'fully_sup':
        f = 'train.txt'
    return f


def get_splits(name, T1=False, whitestripe=False, is_multidomain=True, supRatio=5, seed=41, experiment_mode=None):
    if is_multidomain:  # different seeds and
        if experiment_mode in ['semi', 'semi_downSample', 'semi_alternate']:
            return {  # todo
                           'train_sup':
                               LazyLoader(
                                   Mixed,
                                   *tuple(
                                       LazyLoader(ImageSet,
                                                  './data/wmh_challenge/paths/{}/{}'.format(dset,
                                                                                               get_file(supRatio, seed,
                                                                                                        experiment_mode=experiment_mode,
                                                                                                        isSupervised=True,
                                                                                                        dset=dset)
                                                                                               ),
                                                  domain=dset,
                                                  T1=T1,
                                                  whitestripe=whitestripe)
                                       for dset in DATASETS[name])),
                           'train_unsup':
                               LazyLoader(
                                   Mixed,
                                   *tuple(
                                       LazyLoader(ImageSet,
                                                  './data/wmh_challenge/paths/{}/{}'.format(dset,
                                                                                               get_file(supRatio, seed,
                                                                                                        experiment_mode=experiment_mode,
                                                                                                        isSupervised=False,
                                                                                                        dset=dset)
                                                                                               ),
                                                  domain=dset,
                                                  T1=T1,
                                                  whitestripe=whitestripe)
                                       for dset in DATASETS[name])),

                           'val':
                               LazyLoader(
                                   Mixed,
                                   *tuple(
                                       LazyLoader(ImageSet,
                                                  f'./data/wmh_challenge/paths/{dset}/val.txt',
                                                  domain=dset,
                                                  T1=T1,
                                                  train=False,
                                                  whitestripe=whitestripe)
                                       for dset in DATASETS[name])),



                   }, DOMAINS[name]

        elif experiment_mode in ['partially_sup', 'fully_sup']:
            return {  # todo

                           'train_sup':
                               LazyLoader(
                                   Mixed,
                                   *tuple(
                                       LazyLoader(ImageSet,
                                                  './data/wmh_challenge/paths/{}/{}'.format(dset,
                                                                                               get_file(supRatio, seed,
                                                                                                        experiment_mode=experiment_mode,
                                                                                                        isSupervised=True,
                                                                                                        dset=dset)
                                                                                               ),
                                                  domain=dset,
                                                  T1=T1,
                                                  whitestripe=whitestripe)
                                       for dset in DATASETS[name])),
                           'train_unsup': None,
                           'val':
                               LazyLoader(
                                   Mixed,
                                   *tuple(
                                       LazyLoader(ImageSet,
                                                  f'./data/wmh_challenge/paths/{dset}/val.txt',
                                                  domain=dset,
                                                  T1=T1,
                                                  train=False,
                                                  whitestripe=whitestripe)
                                       for dset in DATASETS[name])),



                   }, DOMAINS[name]





def postprocessing(pred, domain):  # TODO:remove postprocessing
    if domain == 0 or domain == 1:
        start_slice = 6
        num_selected_slice = pred.shape[0]
        pred[0:start_slice, :, :, :] = 0
        pred[(num_selected_slice - start_slice - 1):(num_selected_slice -
                                                     1), :, :, :] = 0
    else:
        start_slice = 11
        num_selected_slice = pred.shape[0]
        pred[0:start_slice, :, :, :] = 0
        pred[(num_selected_slice - start_slice - 1):(num_selected_slice -
                                                     1), :, :, :] = 0

    return pred
