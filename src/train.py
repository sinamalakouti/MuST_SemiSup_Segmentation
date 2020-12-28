import torch
import sys
import os
import torch
from utils import utils

import Wnet
import matplotlib.pyplot as plt
from utils import reconstruction_loss

sys.path.append('src')
sys.path.append('src/utils/Constants')
sys.path.append('srs/utils')

for p in sys.path:
    print("path  ", p)

utils.Constants.USE_CUDA = True
utils.Constants.N_ITERATION = 20000


def train_reconstruction(dataset):
    # read data
    trainset = utils.get_trainset(dataset, False)
    testset = utils.get_testset(dataset, False)

    # TODO: preprocessing?
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]

    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    wnet = Wnet.Wnet(18, 4, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
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
    wnet = Wnet.Wnet(18, 4, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
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
            ncutLoss = utils.soft_n_cut_loss(b, X_out_intermediate, 5)
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


def train_with_two_reconstruction(dataset):
    utils.Constants.FCM = False
    testset = utils.get_testset(dataset, True)
    trainset = utils.get_trainset(dataset, True)

    # TODO: preprocessing?
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    k = 5
    wnet = Wnet.Wnet(18, k, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                     separables=separables)

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("device is     ", dev)
    device = torch.device(dev)
    wnet.build()
    wnet.to(device)
    # linear_combination.to(device)

    optimizer = torch.optim.Adam(wnet.parameters(), 0.001)

    test0 = torch.tensor(testset.dataset.data[0]['data']).reshape((1, 1, 212, 256))
    test1 = torch.tensor(trainset.dataset.data[0]['data']).reshape((1, 1, 212, 256))

    test = test0
    minim = test.min()
    maxim = test.max()
    test = (test - minim) / (maxim - minim)
    plt.imshow(test.reshape((212, 256)))
    plt.savefig("../images/image_ii{}_{}_original.png".format(0, 0))

    test = test1
    minim = test.min()
    maxim = test.max()
    test = (test - minim) / (maxim - minim)
    plt.imshow(test.reshape((212, 256)))
    plt.savefig("../images/image_ii{}_{}_original.png".format(1, 0))

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
            for ii in range(0, 1):
                if ii == 0:
                    test = test0
                else:
                    test = test1

                minim = test.min()
                maxim = test.max()
                test = (test - minim) / (maxim - minim)

                with torch.no_grad():
                    wnet.eval()

                    test.reshape((1, 1, 212, 256))
                    p = wnet(test.to(device))
                    p = p.reshape((test.shape[2], test.shape[3]))
                    p = p.cpu()
                    plt.imshow(p)

                    plt.savefig("../images/image_ii{}_{}.png".format(ii, iter))

                    X_out_intermediate = wnet.U_enc_fw(test.to(device))

                    sample_dir = '../images/segmentation/iter_{}_{}'.format(ii, iter)
                    if not os.path.isdir(sample_dir):
                        try:
                            os.mkdir(sample_dir)
                        except OSError:
                            print("Creation of the directory %s failed" % path)
                    else:
                        None

                    utils.save_segment_images(X_out_intermediate.cpu(),
                                              "../images/segmentation/iter_{}_{}".format(ii, iter))
                    intermediate_pred = wnet.linear_combination(X_out_intermediate)
                    plt.imshow(intermediate_pred.cpu().reshape((212, 256)))
                    plt.savefig("../images/segmentation/iter_{}_{}/linear_comb_{}.png".format(ii, iter, iter))

        wnet.train()
        for batch in trainset:
            b = batch['data']
            b = b.to(device)

            X_out_intermediate = wnet.U_enc_fw(b)
            intermediate_pred = wnet.linear_combination(X_out_intermediate)
            pred = wnet.U_dec_fw(X_out_intermediate)

            intermediate_loss = torch.nn.MSELoss().to(device)
            intermediate_recon_loss = intermediate_loss(intermediate_pred, b)

            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, b)
            # regularization = reconstruction_loss.regularizaton(X_out_intermediate)
            final_loss = recon_loss + intermediate_recon_loss
            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                for p in wnet.linear_combination.parameters():
                    p.data.clamp_(0.0)
            print(final_loss)

    return wnet


def train_only_first_part(dataset):
    trainset = utils.get_trainset(dataset, True)
    testset = utils.get_testset(dataset, True)

    # TODO: preprocessing?
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    k = 4
    wnet = Wnet.Wnet(18, k, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                     separables=separables)

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("device is     ", dev)
    device = torch.device(dev)
    wnet.build()
    wnet.to(device)
    # linear_combination.to(device)

    optimizer = torch.optim.Adam(wnet.parameters(), 0.001)
    intermediate_loss = torch.nn.MSELoss().to(device)

    # test_instance = torch.cat([torch.tensor(testset.dataset.data[1]['data']),torch.tensor(testset.dataset.data[1]['data_t1'])])
    # train_instance = torch.cat([torch.tensor(trainset.dataset.data[0]['data']),torch.tensor(trainset.dataset.data[0]['data_t1'])])
    test_instance = torch.tensor(testset.dataset.data[0]['data'])
    train_instance = torch.tensor(testset.dataset.data[0]['data'])
    test0 = test_instance.reshape((1, 1, 212, 256))
    test1 = train_instance.reshape((1, 1, 212, 256))

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
            for ii in range(1):

                if ii == 0:
                    test = test0
                else:
                    test = test1

                minim = test.min()
                maxim = test.max()
                test = (test - minim) / (maxim - minim)

                with torch.no_grad():
                    wnet.eval()

                    test.reshape((1, 1, 212, 256))
                    X_out_intermediate = wnet.U_enc_fw(test.to(device))
                    # p = p.reshape((test.shape[2], test.shape[3]))
                    # p = p.cpu()
                    # plt.imshow(p)

                    # plt.savefig("../images/image_{}.png".format(iter))
                    plt.imshow(test[0, 0].reshape((212, 256)), 'gray')
                    plt.savefig("../images/image_{}_{}_original.png".format(0, iter))
                    # plt.imshow(test[0, 1].reshape((212, 256)),'gray')
                    # plt.savefig("../images/image_{}_{}_original.png".format(1, iter))

                    # X_in_intermediate = wnet.Uenc(test.to(device))
                    # X_in_intermediate = wnet.conv1(X_in_intermediate)
                    # X_out_intermediate = wnet.softmax(X_in_intermediate)

                    sample_dir = '../images/segmentation/iter_{}_{}'.format(0, iter)
                    if not os.path.isdir(sample_dir):
                        try:
                            os.mkdir(sample_dir)
                        except OSError:
                            print("Creation of the directory %s failed" % path)
                        else:
                            None

                    sample_dir = '../images/segmentation/iter_{}_{}'.format(1, iter)
                    if not os.path.isdir(sample_dir):
                        try:
                            os.mkdir(sample_dir)
                        except OSError:
                            print("Creation of the directory %s failed" % path)
                        else:
                            None

                    utils.save_segment_images(X_out_intermediate.cpu(),
                                              "../images/segmentation/iter_{}_{}".format(0, iter))
                    intermediate_pred = wnet.linear_combination(X_out_intermediate)
                    plt.imshow(intermediate_pred[0, 0].cpu().reshape((212, 256)))
                    plt.savefig("../images/segmentation/iter_{}_{}/linear_comb_{}_{}.png".format(ii, iter, ii, iter))
                    intermediate_pred = wnet.linear_combination(X_out_intermediate)
                    # plt.imshow(intermediate_pred[0,1].cpu().reshape((212, 256)))
                    # plt.savefig("../images/segmentation/iter_{}_{}/linear_comb_{}_{}.png".format(ii, iter,1, iter))

                    wnet.train()

        for batch in trainset:
            f_b = batch['data']
            b = f_b.to(device)
            # t1_b = batch['data_t1']
            # t1_b = t1_b.to(device)
            # b = torch.cat([f_b,t1_b],axis = 1)

            brain = batch['mask']
            brain = brain.to(device)
            X_out_intermediate = wnet.U_enc_fw(b)

            X_out_intermediate = torch.mul(X_out_intermediate, brain)
            X_out_intermediate = X_out_intermediate.type(torch.float)
            intermediate_pred = wnet.linear_combination(X_out_intermediate)
            regularization = reconstruction_loss.regularizaton(X_out_intermediate)
            intermediate_recon_loss = intermediate_loss(intermediate_pred, b) + regularization
            intermediate_recon_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #
            with torch.no_grad():
                for p in wnet.linear_combination.parameters():
                    p.data.clamp_(0.0)
            print(intermediate_recon_loss)

    return wnet


def train_with_two_reconstruction_old(dataset):
    testset = utils.get_testset(dataset, False)
    trainset = utils.get_trainset(dataset, False)

    # TODO: preprocessing?
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    wnet = Wnet.Wnet(18, 4, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                     separables=separables)

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("device is     ", dev)
    device = torch.device(dev)
    wnet.build()
    wnet.to(device)
    # linear_combination.to(device)

    optimizer1 = torch.optim.Adam(wnet.parameters(), 0.001)
    optimizer2 = torch.optim.Adam(wnet.parameters(), 0.001)

    test = torch.tensor(testset.dataset.data[0]['data']).reshape((1, 1, 212, 256))
    minim = test.min()
    maxim = test.max()
    test = (test - minim) / (maxim - minim)

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

                X_in_intermediate = wnet.Uenc(test.to(device))
                X_in_intermediate = wnet.conv1(X_in_intermediate)
                X_out_intermediate = wnet.softmax(X_in_intermediate)
                utils.save_segment_images(X_out_intermediate.cpu(), "../images/segmentation")
                intermediate_pred = wnet.linear_combination(X_out_intermediate)
                plt.imshow(intermediate_pred.cpu().reshape((212, 256)))
                plt.savefig("../images/segmentation/linear_comb_{}.png".format(iter))

                wnet.train()
        torch.autograd.set_detect_anomaly(True)
        for batch in trainset:
            b = batch['data']
            b = b.to(device)
            X_in_intermediate = wnet.Uenc(b)
            X_in_intermediate = wnet.conv1(X_in_intermediate)
            X_out_intermediate = wnet.softmax(X_in_intermediate)
            intermediate_loss = torch.nn.MSELoss().to(device)
            intermediate_pred = wnet.linear_combination(X_out_intermediate)
            intermediate_recon_loss = intermediate_loss(intermediate_pred, b)
            intermediate_recon_loss.backward(retain_graph=True)
            #           optimizer1.step()
            #            optimizer1.zero_grad()
            optimizer1.step()
            optimizer1.zero_grad()
            # for p in wnet.linear_combination.parameters():
            #    p.data.clamp_(0.01)

            X_in_final = wnet.Udec(X_out_intermediate)
            pred = wnet.conv2(X_in_final)
            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, b)
            final_loss = recon_loss

            final_loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            print(final_loss)

    return wnet


def train_with_fcm(dataset):
    utils.Constants.FCM = True
    testset = utils.get_testset(dataset, True)
    trainset = utils.get_trainset(dataset, True)

    # TODO: preprocessing?
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    k = 5
    wnet = Wnet.Wnet(18, k, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                     separables=separables)

    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("device is     ", dev)
    device = torch.device(dev)
    wnet.build()
    wnet.to(device)
    # linear_combination.to(device)

    optimizer = torch.optim.Adam(wnet.parameters(), 0.001)

    test = torch.tensor(testset.dataset.data[0]['data']).reshape((1, 1, 212, 256))
    minim = test.min()
    maxim = test.max()
    test = (test - minim) / (maxim - minim)

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

                X_out_intermediate = wnet.U_enc_fw(b)

                sample_dir = '../images/segmentation/iter_{}'.format(iter)
                if not os.path.isdir(sample_dir):
                    try:
                        os.mkdir(sample_dir)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                else:
                    None

                utils.save_segment_images(X_out_intermediate.cpu(),
                                          "../images/segmentation/iter_{}".format(iter))

                intermediate_pred = wnet.linear_combination(X_out_intermediate)

                plt.imshow(intermediate_pred.cpu().reshape((212, 256)))
                plt.savefig("../images/segmentation/iter_{}/linear_comb_{}.png".format(iter, iter))

                wnet.train()

        # torch.autograd.set_detect_anomaly(True)
        for batch in trainset:
            b = batch['data']
            b = b.to(device)

            prior = batch['wmh_cluster']
            prior = prior.to(device)
            prior = prior.type(torch.DoubleTensor)

            X_out_intermediate = wnet.U_enc_fw(b)
            intermediate_loss = torch.nn.MSELoss().to(device)
            intermediate_pred = wnet.linear_combination(X_out_intermediate)
            intermediate_recon_loss = intermediate_loss(intermediate_pred, b)

            pred = wnet.U_dec_fw(X_out_intermediate)
            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, b)

            # regularization = reconstruction_loss.regularizaton(X_out_intermediate)
            X_out_intermediate = X_out_intermediate.to(device)
            prior = prior.to(device)
            fcm_loss = reconstruction_loss.soft_dice_loss(prior, X_out_intermediate[:, 1, :, :])
            final_loss = recon_loss + intermediate_recon_loss + fcm_loss

            final_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                for p in wnet.linear_combination.parameters():
                    p.data.clamp_(0.0)
            print(final_loss)

    return wnet


if __name__ == '__main__':
    # None
    train_with_fcm(utils.Constants.Datasets.PittLocalFull)
    # train_with_two_reconstruction_old(utils.Constants.Datasets.PittLocalFull)
    # train_only_first_part(utils.Constants.Datasets.PittLocalFull)
    # train_with_two_reconstruction(utils.Constants.Datasets.PittLocalFull)
    # train_reconstruction(utils.Constants.Datasets.PittLocalFull)
    # test(utils.Constants.Datasets.PittLocalFull, '/Users/sinamalakouti/PycharmProjects/WMH_Unsupervised_Segmentation/models/model_epoch_0_.model')
