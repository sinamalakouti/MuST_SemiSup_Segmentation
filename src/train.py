import torch
import sys
import os
import torch
from utils import utils
from utils.reconstruction_loss import ReconstructionLoss

import Wnet
import matplotlib.pyplot as plt

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')

for p in sys.path:
    print("path  ", p)


utils.Constants.USE_CUDA = True
utils.Constants.N_ITERATION = 20000


def test(dataset, model_path):
    testset = utils.get_testset(dataset)
    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("device is     ", dev)

    # TODO: preprocessing?
    # inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]
    # outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    # kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    # paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # separables = [False, True, True, True, True, True, True, True, False]

    wnet = utils.load_model(model_path)
    wnet.to(device)
    wnet.eval()
    with torch.no_grad():
        for batch in testset:
            b = batch['data']
            b = b.to(device)
            X_in_intermediate = wnet.Uenc(b)
            X_in_intermediate = wnet.conv1(X_in_intermediate)
            segmentation = wnet.softmax(X_in_intermediate)
            X_in_final = wnet.Udec(segmentation)
            X_out_final = wnet.conv2(X_in_final)
        utils.save_segment_images(segmentation.cpu(), "../test/segmentation")
        utils.save_images(b.cpu(), X_out_final.cpu(), "../test/reconstruction")


def train_reconstruction(dataset):
    # read data
    trainset = utils.get_trainset(dataset)
    testset = utils.get_testset(dataset)

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
    wnet.build()
    wnet.to(device)
    Recon_loss = ReconstructionLoss()
    optimizer = torch.optim.Adam(wnet.parameters(), 0.001)

    test = torch.tensor(testset.dataset.data[0]['data']).reshape((1, 1, 212, 256))

    for iter in range(utils.Constants.N_ITERATION):
        print("iteration: ", iter)
        wnet.train()
        print(iter)
        if iter % 200 == 0:
            checkname = os.path.join("../models/")
            path = checkname + 'model_epoch_{}_.model'.format(iter)
            with open(path, 'wb') as f:
                print(path)
                torch.save(wnet, f)

        if iter % 10 == 0:
            with torch.no_grad():
                wnet.eval()

                test.reshape((1, 1, 212, 256))
                p = wnet(test.to(device))
                p = p.reshape((test.shape[2], test.shape[3]))
                p = p.cpu()
                plt.imshow(p)
                plt.savefig("../images/image_{}.png".format(iter))
                plt.imshow(test.reshape((212, 256)))
                plt.savefig("../images/image_{}_original.png".format(iter))
                wnet.train()

        for batch in trainset:
            b = batch['data']
            b = b.to(device)
            pred = wnet(b)
            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, b)
            print(recon_loss)
            recon_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return wnet


def train_with_ncut(dataset):
    testset = utils.get_testset(dataset)
    trainset = utils.get_trainset(dataset)

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
    wnet.build()
    wnet.to(device)

    optimizer = torch.optim.Adam(wnet.parameters(), 0.001)

    test = torch.tensor(testset.dataset.data[0]['data']).reshape((1, 1, 212, 256))



    for iter in range(utils.Constants.N_ITERATION):
        print("iteration: ", iter)
        wnet.train()
        print(iter)
        if iter % 200 == 0:
            checkname = os.path.join("../models/")
            path = checkname + 'model_epoch_{}_.model'.format(iter)
            with open(path, 'wb') as f:
                print(path)
                torch.save(wnet, f)

        if iter % 10 == 0:
            with torch.no_grad():
                wnet.eval()

                test.reshape((1, 1, 212, 256))
                p = wnet(test.to(device))
                p = p.reshape((test.shape[2], test.shape[3]))
                p = p.cpu()
                plt.imshow(p)
                plt.savefig("../images/image_{}.png".format(iter))
                plt.imshow(test.reshape((212, 256)))
                plt.savefig("../images/image_{}_original.png".format(iter))
                wnet.train()

        for batch in trainset:
            b = batch['data']
            b = b.to(device)
            X_in_intermediate = wnet.Uenc(b)
            X_in_intermediate = wnet.conv1(X_in_intermediate)
            X_out_intermediate = wnet.softmax(X_in_intermediate)
            X_in_final = wnet.Udec(X_out_intermediate)
            pred = wnet.conv2(X_in_final)
            ncutLoss = utils.soft_n_cut_loss(b,X_out_intermediate,5)
            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, b)
            ncutLoss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(recon_loss)
            recon_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return wnet


#


if __name__ == '__main__':
    train_with_ncut(utils.Constants.Datasets.PittLocalFull)
    # train_reconstruction(utils.Constants.Datasets.PittLocalFull)
    # test(utils.Constants.Datasets.PittLocalFull, '/Users/sina/PycharmProjects/W-Net/models/model_epoch_0_.model')
