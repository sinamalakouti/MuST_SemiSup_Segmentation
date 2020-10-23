import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from utils.reconstruction_loss import ReconstructionLoss


from utils import  Constants


class Module(nn.Module):
    def __init__(self, dim_in, dim_out, stride, padding, kernel_size=3, separable=False):
        super(Module, self).__init__()
        self.separable = separable
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.net = self.__build_module()

    def __depthWise_separable_conv(self, in_dim, out_dim, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, padding=1, kernel_size=kernel_size, groups=in_dim),
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, groups=1)
        )

    def __build_module(self):
        if not self.separable:
            return nn.Sequential(
                nn.Conv2d(self.dim_in, self.dim_out, kernel_size=self.kernel_size, padding=self.padding,
                          stride=self.stride),
                nn.ReLU(),
                nn.BatchNorm2d(self.dim_out),
                nn.Conv2d(self.dim_out, self.dim_out, kernel_size=self.kernel_size, padding=self.padding,
                          stride=self.stride),
                nn.ReLU(),
                nn.BatchNorm2d(self.dim_out)
            )
        else:
            return nn.Sequential(
                self.__depthWise_separable_conv(self.dim_in, self.dim_out, self.kernel_size),
                nn.ReLU(),
                nn.BatchNorm2d(self.dim_out),
                self.__depthWise_separable_conv(self.dim_out, self.dim_out, self.kernel_size),
                nn.ReLU(),
                nn.BatchNorm2d(self.dim_out)
            )

    def forward(self, X):
       # module = self.__build_module()
        return self.net(X)


class Unet(nn.Module):
    def __init__(self, n_modules, dim_in, dim_out, stride, padding, kernel_size, separable):
        super(Unet, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.stride = stride
        self.padding = padding
        self.separable = separable
        self.kernel_size = kernel_size
        self.n_modules = n_modules
        self.__build()

    def __build(self):
        n_contracting_path = self.n_modules // 2 + 1
        # n_expanding_path = self.n_modules - n_contracting_path
        contracting_modules = []
        expanding_modules = []
        for i in range(self.n_modules):
            if i < n_contracting_path:
                m = Module(self.dim_in[i], self.dim_out[i], stride=self.stride[i], padding=self.padding[i],
                           kernel_size=self.kernel_size[i], separable=self.separable[i])
                contracting_modules.append(m)
            else:
                m = Module(self.dim_in[i], self.dim_out[i], stride=self.stride[i], padding=self.padding[i],
                           kernel_size=self.kernel_size[i], separable=self.separable[i])
                expanding_modules.append(m)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(Constants.DROP_OUT)
        self.down_samplers = [None] + [self.pool] * (n_contracting_path - 1)
        self.dropOuts = [None] + [self.dropout] * (n_contracting_path)
        self.up_samplers = [nn.ConvTranspose2d(self.dim_out[i], self.dim_in[i],
                                               kernel_size=3, stride=2, padding=1, output_padding=1)
                            for i in range(n_contracting_path - 1, -1, -1)]
        self.up_samplers = torch.nn.ModuleList(self.up_samplers)
        self.contracting_modules = torch.nn.ModuleList(contracting_modules)
        self.expanding_modules = torch.nn.ModuleList(expanding_modules)
        # return self.modules

    def forward(self, X):
        n_contracting_path = self.n_modules // 2 + 1

        Xs = []
        for i, module in enumerate(self.contracting_modules):
            if self.down_samplers[i] is None:
                X_in = X
            else:
                if Constants.DROP_OUT:
                    X_in = self.dropOuts[i](Xs[-1])
                else:
                    X_in = Xs[-1]
                # X_in = Xs[-1]
                X_in = self.down_samplers[i](X_in)
            X_out = module(X_in)
            if i != n_contracting_path - 1:
                Xs.append(X_out)

        # expanding modules
        Xs.reverse()
        counter = self.n_modules - n_contracting_path
        for i, modules in enumerate(self.expanding_modules):
            j = i + n_contracting_path
            X_in = self.up_samplers.__getitem__(i)(X_out)
            h_diff = Xs[i].size()[2] - X_in.size()[2]
            w_diff = Xs[i].size()[3] - X_in.size()[3]

            if (h_diff != 0 or w_diff != 0):
                X_in = F.pad(X_in, (0, w_diff, 0 ,h_diff ))
                # print("shape shpae  ", X_in.shape)
            X_in = torch.cat([Xs[i], X_in], 1)
            X_out = modules(X_in)

        return X_out


class Wnet(nn.Module):

    def __init__(self, n_modules, k, dim_inputs, dim_outputs, kernels, strides, paddings, separables):
        super(Wnet, self).__init__()
        self.strides = strides
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.kernels = kernels
        self.paddings = paddings
        self.separables = separables
        self.n_modules = n_modules
        self.k = k
        self.build()

    def build(self):
        self.Uenc = Unet(self.n_modules // 2, self.dim_inputs, self.dim_outputs, self.strides, self.paddings,
                       self.kernels, self.separables)
        self.conv1 = nn.Conv2d(self.dim_outputs[-1], self.k, kernel_size=1)
        dec_dim_inputs = [0] * len(self.dim_inputs)
        dec_dim_inputs[0] = self.k
        dec_dim_inputs[1:] = self.dim_inputs[1:]
        self.Udec = Unet(self.n_modules // 2, dec_dim_inputs, self.dim_outputs, self.strides, self.paddings,
                         self.kernels, self.separables)
        self.conv2 = nn.Conv2d(self.dim_outputs[-1], self.dim_inputs[0], kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, X):
        # if torch.cuda.is_available():
        #     dev = "cuda:0"
        # else:
        #     dev = "cpu"
        # print("11111device is     ", dev)
        # print(" input , ", X.is_cuda)
        # device = torch.device(dev)
       # print("device 1   ",  next(self.Uenc.parameters()))
       #  self.Unec = self.Uenc.to(device)
      #:  print("deviece 22 ", self.Uenc.is_cuda)
        X_in_intermediate = self.Uenc(X)
        X_in_intermediate = self.conv1(X_in_intermediate)
        X_out_intermediate = self.softmax(X_in_intermediate)
        X_in_final = self.Udec(X_out_intermediate)
        X_out_final = self.conv2(X_in_final)
        return X_out_final


if __name__ == '__main__':
    inputs_dim = [1, 64, 128, 256, 512, 1024, 512, 256, 128]
    outputs_dim = [64, 128, 256, 512, 1024, 512, 256, 128, 64]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    separables = [False, True, True, True, True, True, True, True, False]
    wnet = Wnet(18, 5, inputs_dim, outputs_dim, strides=strides, paddings=paddings, kernels=kernels,
                separables=separables)
    wnet.build()

    a = np.random.rand(20, 1, 212, 256)
    X = list(a)
    X = torch.FloatTensor(X)
    Y = wnet(X)
    optimizer = torch.optim.SGD(wnet.parameters(), 0.001)
    print(Y.shape)
