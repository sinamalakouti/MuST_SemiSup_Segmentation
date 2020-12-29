import torch
from utils import utils
from evaluation_metrics import dice_coef
import numpy as np
import Wnet

def open_net(net):
    s = torch.nn.BatchNorm2d(3)

    h = [module for module in net.modules() if type(module) != torch.nn.Sequential]

    for m in h:
        if type(m) == type(Wnet.Unet) or type(m) == type(torch.nn.ModuleList) or type(m) ==  type(torch.nn.Sequential):
            open_net(m)
        elif type(m) == type(s):
            print("here")
            m.track_running_stats=False
            m.momentum=0
            m.training=True
            print(m)
        else:
            print(type(m))

def test(dataset, model_path):
    testset = utils.get_testset(dataset,True)
    if torch.cuda.is_available() and utils.Constants.USE_CUDA:
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("device is     ", dev)

    # TODO: preprocessing?
    inputs_dim = [1, 64, 128, 256,12, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]

    wnet = utils.load_model(model_path)
    wnet.to(device)
    wnet.eval()
    # open_net(wnet)

    print("here1  ")
    with torch.no_grad():
        for batch in testset:
            x_test = batch['data']
            y_true = batch['label']
            x_test = x_test.to(device)
            X_in_intermediate = wnet.Uenc(x_test)
            X_in_intermediate = wnet.conv1(X_in_intermediate)
            print("here2")
            segmentation = wnet.softmax(X_in_intermediate)
            segments = segmentation.argmax(1)
            wmh_segment = segments == 1
            dice_score = dice_coef(y_true.reshape(wmh_segment.shape), wmh_segment)
            dice_arr  = dice_score.numpy()
            text = np.array2string(dice_arr)
            text += '\n mean is :    '+ str(np.mean(dice_arr))
            text_file = open("../test/final_result.txt", "w")
            text_file.write(text)
            text_file.close()
            X_in_final = wnet.Udec(segmentation)
            X_out_final = wnet.conv2(X_in_final)
            utils.save_segment_images(segmentation.cpu(), "../test/segmentation")
            utils.save_images(x_test.cpu(), X_out_final.cpu(), "../test/reconstruction")
            utils.save_label(y_true.reshape((20,212,256)), "../test/labels")


#
# def evaluate_dice(model:Wnet,testset):
#     testdata = testset.dataset.data
#     n_testdata = len(testdata)
#     dice_score = np.zeros(n_testdata)
#
#     for test_sample in testdata:
#         x_test = test_sample['data'].reshape(1, 1, 212, 256)
#         y_test = test_sample['label']
#         pred =


if __name__ == '__main__':
    test(utils.Constants.Datasets.PittLocalFull, '../models_enc_withoutMask/model_epoch_600_.model')
