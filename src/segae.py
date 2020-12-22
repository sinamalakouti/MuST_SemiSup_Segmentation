import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from utils import Constants

class Conv_Module(nn.Module):

    def __init__(self, in_channels, out_channels, kernel,stride):
        super(Conv_Module, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel - kernel
        self.stride = stride
        self.__build()

    def __build(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, stride=self.stride, kernel_size=self.kernel),
            nn.LeakyReLU(),
            nn.BatchNorm2d()
        )


class Wnet(nn.Module):

    def __init__(self):
        super(Wnet, self).__init__()
        self.__build()

    def __build(self):
        self.block1 = Conv_Module(2,32,stride=1,kernel=3)
        self.down1 = Conv_Module(32, 64, stride=2, kernel=3)

        self.block2 = Conv_Module(64, 64, stride=1, kernel=3)
        self.down2 = Conv_Module(64,128,stride=2,kernel=3)

        self.block3 = Conv_Module(128,128,stride=1, kernel=3)
        self.up1 = nn.ConvTranspose2d(128,128,kernel_size=2)

        self.block4 = Conv_Module(192, 64, stide= 1, kernel =3)
        self.block5 = Conv_Module(64,64, stride=1, kernel=3)

        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2)

        self.block6 = Conv_Module(96,32, stride=1, kernel =3)
        self.block7 = Conv_Module(32,23,stride=1, kernel=3)

        self.block7=  Conv_Module(32,16,stride=1, kernel=1)


