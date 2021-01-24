from PIL import Image
import torch
import torch.utils.data as Data
import os
import glob
import numpy as cp
import nibabel as nib
import numpy as np
from utils import soft_n_cut_loss
from utils import Constants
from utils.soft_n_cut_loss import *
import fcm
from utils import utils


class DataLoader():
    # initialization
    # datapath : the data folder of bsds500
    # mode : train/test/val
    def __init__(self, datapath, mode):
        # image container
        self.raw_data = []
        self.mode = mode

        # navigate to the image directory
        # images_path = os.path.join(datapath,'images')
        train_image_path = os.path.join(datapath, mode)
        file_list = []
        if (mode != "train"):
            train_image_regex = os.path.join(train_image_path, '*.jpg')
            file_list = glob.glob(train_image_regex)
        # find all the images
        else:
            train_list_file = os.path.join("../VOC2012", Constants.imagelist)
            with open(train_list_file) as f:
                for line in f.readlines():
                    file_list.append(os.path.join(train_image_path, line[0:-1] + ".jpg"))
        # load the images
        for file_name in file_list:
            with Image.open(file_name) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                self.raw_data.append(
                    cp.array(image.resize((Constants.inputsize[0], Constants.inputsize[1]), Image.BILINEAR)))
        # resize and align
        self.scale()
        # normalize
        self.transfer()

        # calculate weights by 2
        if (mode == "train"):
            self.dataset = self.get_dataset(self.raw_data, self.raw_data.shape, 75)
        else:
            self.dataset = self.get_dataset(self.raw_data, self.raw_data.shape, 75)

    def scale(self):
        for i in range(len(self.raw_data)):
            image = self.raw_data[i]
            self.raw_data[i] = cp.stack((image[:, :, 0], image[:, :, 1], image[:, :, 2]), axis=0)
        self.raw_data = cp.stack(self.raw_data, axis=0)

    def transfer(self):
        # just for RGB 8-bit color
        self.raw_data = self.raw_data.astype(cp.float)
        # for i in range(self.raw_data.shape[0]):
        #    Image.fromarray(self.raw_data[i].swapaxes(0,-1).astype(np.uint8)).save("./reconstruction/input_"+str(i)+".jpg")

    def torch_loader(self):
        return Data.DataLoader(
            self.dataset,
            batch_size=Constants.BatchSize,
            shuffle=Constants.Shuffle,
            num_workers=Constants.LoadThread,
            pin_memory=True,
        )

    def get_dataset(self, raw_data, shape, batch_size):
        dataset = []
        for batch_id in range(0, shape[0], batch_size):
            print(batch_id)
            batch = raw_data[batch_id:min(shape[0], batch_id + batch_size)]
            if (self.mode == "train"):
                print("shidfihfif")
                # weight = compute_weigths(batch, batch.shape)
                # weight = cp.asnumpy(tmp_weight)
                # dataset.append(Data.TensorDataset(torch.from_numpy(batch / 256).float(), weight))
            else:
                dataset.append(Data.TensorDataset(torch.from_numpy(batch / 256).float()))
        cp.get_default_memory_pool().free_all_blocks()
        return Data.ConcatDataset(dataset)


class PittLocalFull(torch.utils.data.Dataset):

    def __init__(self, ws, T1, mixup_threshold, intensity_aug, intensity_rescale, data_paths, label_paths, mask_paths,
                 is_ncut=False, is_FCM=False, is_train=True, augment=False, data_paths_t1=None):
        super(PittLocalFull, self).__init__()

        # NOTE: if dataloader does not shuffle
        # and batch size is kept to 5, then
        # each batch equates to a single subject
        self.T1 = T1
        self.ws = ws
        self.intensity_aug = intensity_aug
        self.intensity_rescale = intensity_rescale
        self.augment = augment
        self.mixup_threshold = mixup_threshold
        self.is_train = is_train
        self.weights = {}
        self.is_ncut = is_ncut
        self.is_FCM = is_FCM
        # self.iter = 0
        if self.T1 is not None:
            self.order = [f.strip().split('/')[-1].strip('_FL_preproc.nii.gz')  # change strip later if other exp
                          for p in data_paths for f in open(p) for _ in range(5)]

            data_paths = [f.strip() for p in data_paths
                          for f in open(p).readlines()]
            label_paths = [f.strip() for p in label_paths
                           for f in open(p).readlines()]
            mask_paths = [f.strip() for p in mask_paths
                          for f in open(p).readlines()]
            data_paths_t1 = [f.strip() for p in data_paths_t1
                             for f in open(p).readlines()]

            paths = zip(data_paths, label_paths, mask_paths, data_paths_t1)
            self.data = []
            for data_f, label_f, mask_f, data_t1_f in paths:
                X = None
                X_t1 = None
                if self.ws is not None:
                    X = self._extract(data_f, slices=(0, 1, 2, 3, 4))
                    X_t1 = self._extract(data_t1_f, slices=(0, 1, 2, 3, 4))
                else:
                    X = self._extract(data_f)
                    X_t1 = self._extract(data_t1_f)
                Y = self._extract(label_f)
                M = self._extract(mask_f)
                for sl in range(Y.shape[2]):
                    self.data.append({
                        'data': X[:, :, sl],
                        'label': Y[:, :, sl],
                        'mask': M[:, :, sl],
                        'data_t1': X_t1[:, :, sl]})
        else:
            self.order = [f.strip().split('/')[-1].strip('_flair.nii.gz')
                          for p in data_paths for f in open(p) for _ in range(5)]

            data_paths = [f.strip() for p in data_paths
                          for f in open(p).readlines()]
            label_paths = [f.strip() for p in label_paths
                           for f in open(p).readlines()]
            mask_paths = [f.strip() for p in mask_paths
                          for f in open(p).readlines()]

            paths = zip(data_paths, label_paths, mask_paths)
            self.data = []
            for data_f, label_f, mask_f in paths:
                X = None
                if self.ws is not None:
                    X = self._extract(data_f, slices=(0, 1, 2, 3, 4))
                else:
                    X = self._extract(data_f)
                Y = self._extract(label_f)
                M = self._extract(mask_f)
                for sl in range(Y.shape[2]):
                    self.data.append({
                        'data': X[:, :, sl],
                        'label': Y[:, :, sl],
                        'mask': M[:, :, sl]})

    def __len__(self):
        return len(self.data)

    def _extract(self, f, slices=(24, 25, 26, 27, 28)):
        x = nib.load(f).get_data()
        slices = np.array(slices)
        return x[:, :, slices].astype('float32')

    def __getitem__(self, index):
        if self.T1 is not None:
            x = self.data[index]['data']
            y = self.data[index]['label']
            m = self.data[index]['mask']  # mask no need to do intensity rescale
            x_t1 = self.data[index]['data_t1']
            # x = rescale_intensity(x)  # chg
            #            y = rescale_intensity(y) #chg
            #x_t1 = rescale_intensity(x_t1)  # chg
            if self.augment:
                None
                # x, y, m, x_t1 = augment(
                #     x=x, y=y, m=m, t1=x_t1, intensity_aug=self.intensity_aug)
            else:
                x, y, m, x_t1 = tensorize(x, y, m, x_t1)
            output_arr = []
            output_arr.append(x)
            output_arr.append(x_t1)
            # lamda * output_arr[0] + (1-lamda)*output_arr[1]
            if self.mixup_threshold is not None:  # chg
                x_final = self.mixup_threshold * output_arr[0] + (1 - self.mixup_threshold) * output_arr[1]
                # print("mixup is not, none")
                # print(self.mixup_threshold)
                # print("mixup is not, none")
                # print(self.intensity_aug)

            else:
                if self.intensity_rescale:
                    output_arr[0] = rescale_intensity(output_arr[0])  # chg/
                    output_arr[1] = rescale_intensity(output_arr[1])
                x_final = np.concatenate(output_arr[0:2], axis=0)
            # print(self.mixup_threshold)
            # print("mixup is none")
            # print(self.intensity_aug)
            WMH_cluster = None
            if self.is_FCM:
                import matplotlib.pyplot as plt
                if 'fcm_seg' not in self.data[index]:
                    WMH_cluster = fcm.fcm_WMH_segmentation(x[0], 2, 0.03, 1)

                    WMH_cluster = torch.Tensor(WMH_cluster)
                    self.data[index]['fcm_seg'] = WMH_cluster.type(torch.DoubleTensor)
                else:
                    WMH_cluster = self.data[index]['fcm_seg']

            if self.is_FCM:
                return {'data': x_final, 'label': y, 'mask': m.bool(),
                        'subject': self.order[index], 'wmh_cluster': WMH_cluster}
            return {'data': x_final, 'label': y, 'mask': m.bool(),
                    'subject': self.order[index]}
        else:
            x = self.data[index]['data']
            y = self.data[index]['label']
            m = self.data[index]['mask']  # mask no need to do intensity rescale

            if self.augment:
                None
                # x, y, m = augment(
                #     x=x, y=y, m=m, intensity_aug=self.intensity_aug)
            else:
                x, y, m = tensorize(x, y, m)

            WMH_cluster = None
            if self.is_FCM:
                import matplotlib.pyplot as plt
                if 'fcm_seg' not in self.data[index]:
                    WMH_cluster = fcm.fcm_WMH_segmentation(x[0], 2, 0.03, 1)

                    WMH_cluster = torch.Tensor(WMH_cluster)
                    self.data[index]['fcm_seg'] = WMH_cluster.type(torch.DoubleTensor)
                else:
                    WMH_cluster = self.data[index]['fcm_seg']

                # plt.imshow(WMH_cluster, 'gray')
                # image_path =  "../images/wmh_{}.png".format(self.iter)
                # plt.savefig(image_path)
                #
                # plt.imshow(y.reshape(212,256), 'gray')
                # image_path = "../images/label_{}.png".format(self.iter)
                # plt.savefig(image_path)

                # self.iter += 1

            if self.intensity_rescale:
                x = normalize_quantile(x, 0.99)
                # x = rescale_intensity(x) #chg/
            #            y = rescale_intensity(y) #chg

            # if self.is_train and self.is_ncut:
            #     if index not in self.weights:
            #
            #         r = x.shape[1]
            #         c = x.shape[2]
            #         self.weights[index] = compute_weigths(x.flatten(),r,c)

            if self.is_FCM:
                return {'data': x, 'label': y, 'mask': m.bool(),
                        'subject': self.order[index], 'wmh_cluster': WMH_cluster}
            return {'data': x, 'label': y, 'mask': m.bool(),
                    'subject': self.order[index]}


def tensorize(*args):
    return tuple(torch.Tensor(arg).float().unsqueeze(0) for arg in args)


def rescale_intensity(x):
    return normalize_quantile(x,0.99)
    maximum = x.max()
    minimum = x.min()
    return (x - minimum + 0.01) / (maximum - minimum + 0.01)


def normalize_quantile(x, threshold):
    q = torch.quantile(x, threshold)
    mask = x[x <= q]
    return x / max(mask)
