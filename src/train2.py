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


def train(dataset):
    utils.FCM = False
    # read data
    trainset = utils.get_trainset(dataset,5, False, None, None)
    testset = utils.get_testset(dataset,5, False, None, None)
    experiment_path = "only_unet"

    cuda_number = 1
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]

    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:{}".format(cuda_number)
    else:
        dev = "cpu"
    print("device is     ", dev)
    device = torch.device(dev)

    wnet = Wnet.Wnet(18, 4, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                     separables=separables)

    wnet.build()
    wnet.to(device)

    optimizer = torch.optim.Adam(wnet.parameters(), 0.001)
    #get dataset
    test = torch.tensor(testset.dataset.data[0]['data']).reshape((1, 1, 212, 256))
    plt.imshow(test.reshape((212, 256)))
    plt.savefig("../" + experiment_path + "/images/image_original.png")

    for iter in range(utils.Constants.N_ITERATION):
        print("iteration: ", iter)
        wnet.train()
        print(iter)
        if iter % 200 == 0:
            checkname = os.path.join("../" + experiment_path + "/models/")
            path = checkname + 'model_epoch_{}_.model'.format(iter)
            with open(path, 'wb') as f:
                print(path)
                torch.save(wnet, f)

        if iter % 10 == 0:
            save_test(wnet, experiment_path, test, device, path,iter)



        for batch in trainset:
            wnet.train()
            b = batch['data']
            b = b.to(device)
            segments = wnet.U_enc_fw(b)
            pred = wnet.linear_combination(segments)

            loss = torch.nn.MSELoss().to(device)
            recon_loss = loss(pred, b)
            recon_loss.backward()
            regularization = reconstruction_loss.regularization(segments)

            optimizer.step()
            optimizer.zero_grad()

    return wnet



def save_test(wnet, experiment_path, test, device, path, iter):
    with torch.no_grad():
        wnet.eval()

        test.reshape((1, 1, 212, 256))
        X_out_intermediate = wnet.U_enc_fw(test.to(device))

        sample_dir = '../' + experiment_path + '/images/segmentation/iter_{}'.format(iter)
        if not os.path.isdir(sample_dir):
            try:
                os.mkdir(sample_dir)
            except OSError:
                print("Creation of the directory %s failed" % path)
        else:
            None

        utils.save_segment_images(X_out_intermediate.cpu(),
                                  "../" + experiment_path + "/images/segmentation/iter_{}".format(iter))

        intermediate_pred = wnet.linear_combination(X_out_intermediate)

        plt.imshow(intermediate_pred[0, 0].cpu().reshape((212, 256)))
        plt.savefig("../" + experiment_path + "/images/segmentation/iter_{}/linear_comb_{}.png".format(iter, iter))

if __name__ == '__main__':
    dataset = utils.Datasets.PittLocalFull
    train(dataset)
