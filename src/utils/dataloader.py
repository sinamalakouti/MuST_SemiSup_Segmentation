from PIL import Image
import torch
import torch.utils.data as Data
import os
import glob
import numpy as cp
import nibabel as nib
import numpy as np
from utils import Constants
import fcm
from utils import utils

import torchvision.transforms.functional as augmentor


def adjust_contrast(x, c_factor):
    x_np = x - x.mean()
    x_np = np.multiply(x_np, c_factor)
    x_np = x_np + x.mean()
    return x_np


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
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 3)
    # ax.flat[0].imshow(x)
    if intensity_aug is not None:
        x = adjust_contrast(x, c_factor)

    x = augmentor.to_pil_image(x, mode='F')
    # ax.flat[1].imshow(x)
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
    # ax.flat[2].imshow(x.squeeze(0))
    # plt.show(); exit()
    y = augmentor.to_tensor(y).float()
    y = (y > 0).float()

    if m is not None:
        m = augmentor.to_tensor(m).float()
        m = (m > 0).float()

    if t1 is not None:
        t1 = augmentor.to_tensor(t1).float()

        # returns 1xHxW

    # x_numpy = x.numpy()
    # y_numpy = y.numpy()
    # if m is not None:
    #     m_numpy = m.numpy()
    # if t1 is not None:
    #     t1_numpy = t1.numpy()
    # plt.rcParams.update({'figure.max_open_warning': 0})
    # fig, ax = plt.subplots(1, 2)
    # ax.flat[0].imshow(np.rot90(x_numpy.squeeze(0), 3), vmin=0, vmax=800,
    #                   interpolation='none', origin='lower', cmap='gray')
    # ax.flat[1].imshow(np.rot90(ori_x, 3), vmin=0, vmax=800,
    #                   interpolation='none', origin='lower', cmap='gray')
    # # ax.flat[2].imshow(np.rot90(y_numpy.squeeze(0), 3),
    # #                   interpolation='none', origin='lower', cmap='gray')
    # # if m is not None:
    # #     ax.flat[3].imshow(np.rot90(m_numpy.squeeze(0), 3),
    # #                       interpolation='none', origin='lower', cmap='gray')
    # # if t1 is not None:
    # #     ax.flat[4].imshow(np.rot90(t1_numpy.squeeze(0), 3), vmin=0, vmax=800,
    # #                       interpolation='none', origin='lower', cmap='gray')
    # #     ax.flat[5].imshow(np.rot90(ori_t1, 3), vmin=0, vmax=800,
    # #                       interpolation='none', origin='lower', cmap='gray')
    # fig.savefig('temp_fig/fig.jpg')

    if m is not None and t1 is not None:
        return x, y, m, t1
    elif m is not None and t1 is None:
        return x, y, m
    elif m is None and t1 is not None:
        return x, y, t1
    else:
        return x, y


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

    def perform_fcm(self, index, x):
        WMH_cluster = None
        if 'fcm_seg' not in self.data[index]:
            WMH_cluster = fcm.fcm_WMH_segmentation(x[0], 2, 0.03, 1)

            WMH_cluster = torch.Tensor(WMH_cluster)
            self.data[index]['fcm_seg'] = WMH_cluster.type(torch.DoubleTensor)
        else:
            WMH_cluster = self.data[index]['fcm_seg']
        return WMH_cluster

    def __getitem__(self, index):
        if self.T1 is not None:
            x = self.data[index]['data']
            y = self.data[index]['label']
            m = self.data[index]['mask']  # mask no need to do intensity rescale
            x_t1 = self.data[index]['data_t1']

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
                x_mixup = self.mixup_threshold * output_arr[0] + (1 - self.mixup_threshold) * output_arr[1]
                output_arr.append(x_mixup)

            if self.intensity_rescale:
                output_arr[0] = rescale_intensity(output_arr[0])
                output_arr[1] = rescale_intensity(output_arr[1])
                if len(output_arr) == 3:
                    output_arr[2] = rescale_intensity(output_arr[2])
                x_final = np.concatenate(output_arr[0:], axis=0)

            if self.is_FCM:
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
                x, y, m = augment(
                    x=x, y=y, m=m, intensity_aug=self.intensity_aug)
            else:
                x, y, m = tensorize(x, y, m)

            WMH_cluster = None
            if self.is_FCM:
                WMH_cluster = self.perform_fcm(index, x)
            x = rescale_intensity(x)

            if self.is_FCM:
                return {'data': x, 'label': y, 'mask': m.bool(),
                        'subject': self.order[index], 'wmh_cluster': WMH_cluster}
            return {'data': x, 'label': y, 'mask': m.bool(),
                    'subject': self.order[index]}


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
