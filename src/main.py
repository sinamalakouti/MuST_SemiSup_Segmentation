import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from src.utils import utils
from src.utils.reconstruction_loss import ReconstructionLoss
from src import Wnet
import matplotlib.pyplot as plt

utils.Constants.USE_CUDA = True

def train(dataset):
    # read data
    # dataset = utils.get_dataset(dataset)
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    dataset = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('path/to/mnist_root/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('/files/', train=False, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ])),
    #     batch_size=batch_size_test, shuffle=True)

    # TODO: preprocessing?
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    wnet = Wnet.Wnet(18, 5, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                separables=separables)

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("device is     ", dev)
    device = torch.device(dev)
    wnet.to(device)
    wnet.build()

    Recon_loss = ReconstructionLoss()
    optimizer = torch.optim.Adam(wnet.parameters(), 0.01)

    for iter in range(utils.Constants.N_ITERATION):
        wnet.train()
        print(iter)
        i = 0
        for batch in dataset:
            batch = batch[0]
            # print(batch[0])
            # print("data shape is ", batch[0].shape)
            pred = wnet(batch.to(device))

            # recon_loss = Recon_loss.compute_loss(pred, batch['data'])
            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, batch)
            print("evaluation")
            print(recon_loss)
            recon_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                wnet.eval()
                p = wnet(batch)
                print("hereeerreererr   ", p.shape )
                plt.imshow(p[0,:].data[0])
                plt.show()
                plt.imshow(batch[0].data[0])
                plt.show()
                wnet.train()

                # plt.imshow(batch['data'].data[0])
                # plt.show()
                # input()





            i +=1
    return wnet


# def save_model(model):
#


if __name__ == '__main__':
    train(utils.Constants.Datasets.PittLocalFull)
