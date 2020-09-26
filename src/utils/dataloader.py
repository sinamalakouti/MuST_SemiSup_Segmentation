from PIL import Image
import torch
import torch.utils.data as Data
import os
import glob
import numpy as cp
import soft_n_cut_loss.py
import Constatns.py

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
                self.raw_data.append(cp.array(image.resize((Constants.inputsize[0], Constants.inputsize[1]), Image.BILINEAR)))
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
                weight = compute_weigths(batch, batch.shape)
                # weight = cp.asnumpy(tmp_weight)
                dataset.append(Data.TensorDataset(torch.from_numpy(batch / 256).float(), weight))
            else:
                dataset.append(Data.TensorDataset(torch.from_numpy(batch / 256).float()))
        cp.get_default_memory_pool().free_all_blocks()
        return Data.ConcatDataset(dataset)
