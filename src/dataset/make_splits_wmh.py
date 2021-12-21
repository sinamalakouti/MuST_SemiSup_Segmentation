

import numpy as np


# /home/sina/WMH_semisup_segmentation/WMH_Unsupervised_Segmentation/src/data/wmh_challenge/paths/GE3T


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
        file = open(sup_dir_path + "/train_{}_sup_ids{}.csv".format(domain, int(ratio * 100)), "w")
        file.write(joined_string)
    else:
        joined_string = "\n".join(sup_ids)
        file = open(sup_dir_path + "/train_{}_sup_ids{}_seed{}.csv".format(domain, int(ratio * 100), seed), "w")
        file.write(joined_string)
    if seed is None:
        joined_string = "\n".join(unsup_ids)
        file = open(unsup_dir_path + "/train_{}_unsup_ids{}.csv".format(domain, int(ratio * 100)), "w")
        file.write(joined_string)
    else:
        joined_string = "\n".join(unsup_ids)
        file = open(unsup_dir_path + "/train_{}_unsup_ids{}_seed{}.csv".format(domain, int(ratio * 100), seed), "w")
        file.write(joined_string)


domains = ['Singapore', 'GE3T', 'Utrecht']

ratios = [0.25, 0.3]
seeds = [41,42,43]
for ratio in ratios:
    for seed in seeds:
        for domain in domains:
            all_data_path = '/home/sina/WMH_semisup_segmentation/WMH_Unsupervised_Segmentation/src/data/wmh_challenge/paths/{}/train.txt'.format(domain)
            sup_path = '/home/sina/WMH_semisup_segmentation/WMH_Unsupervised_Segmentation/src/data/wmh_challenge/paths/{}'.format(domain)
            unsup_path = '/home/sina/WMH_semisup_segmentation/WMH_Unsupervised_Segmentation/src/data/wmh_challenge/paths/{}'.format(domain)
            create_semi_sup_split(all_data_path, sup_path, unsup_path, ratio=ratio, seed=seed,
                                  domain=domain)