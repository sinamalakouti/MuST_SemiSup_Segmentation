import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')


for p in sys.path:
    print("path  ", p)
from utils import utils
from utils.reconstruction_loss import ReconstructionLoss
import Wnet
import matplotlib.pyplot as plt

utils.Constants.USE_CUDA = True
utils.Constants.N_ITERATION = 20000

def train(dataset):
    # read data
    dataset = utils.get_dataset(dataset)
    trainset = dataset.dataset.data[0:90][:]

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
    wnet.to(device)
    Recon_loss = ReconstructionLoss()
    optimizer = torch.optim.Adam(wnet.parameters(), 0.001)

    test = torch.tensor(dataset.dataset.data[0]['data']).reshape((1,1,212,256))
    for iter in range(utils.Constants.N_ITERATION):
        print("iteration: ", iter)
        wnet.train()
        print(iter)
        if iter % 200 == 0:
            # wnet.save("model.pt")
            checkname = os.path.join("../models/")
            path = checkname + 'model_epoch_{}_.model'.format(iter)
            with open(path,'wb') as f:
            # path = os.path.join("./models/")
                print(path)
                torch.save(wnet, f)


        if iter % 10 == 0:
            with torch.no_grad():
                wnet.eval()
                test.reshape((1,1,212,256))
                p = wnet(test.to(device))
                p = p.reshape((test.shape[2], test.shape[3]))
                p.cpu()
                plt.imshow(p)
                plt.savefig("../images/image_{}.png".format(iter))
                plt.imshow(test.reshape((212,256)))
                plt.savefig("../images/image_{}_original.png".format(iter))
                wnet.train()
        j = 0
        for batch in dataset:
            b = batch['data']
            b.to(device)
            pred = wnet(b)
            # recon_loss = Recon_loss.compute_loss(pred, batch['data'])
            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, b)
            if j == 0:
                # p = p.reshape((test.shape[2], test.shape[3]))
                # p.cpu()
                plt.imshow(pred[0 ,:].data.reshape(212,256))
                plt.show()
                plt.imshow(b.data[0].reshape(212,256))
                plt.show()
            print(recon_loss)
            recon_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            j +=1
    return wnet


# def save_model(model):
#


if __name__ == '__main__':
    train(utils.Constants.Datasets.PittLocalFull)
