

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, stride, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.stride = stride
        if kernel_size == 3:
            self.padding = 1
        elif kernel_size == 5:
            self.padding = 2
        self.net = self.__build_module()

    def __build_module(self):
        return nn.Sequential(
            nn.Conv2d(self.dim_in, self.dim_out, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.dim_out),
            nn.ELU(),
            nn.Conv2d(self.dim_out, self.dim_out, self.kernel_size, self.stride, self.padding),
            nn.BatchNorm2d(self.dim_out),
            nn.ELU()
        )

    def forward(self, X):
        return self.net(X)

class CLS (nn.Module):
    def  __init__(self, dim_in, dim_out):
        super(CLS, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.net = self.__build_module()

    def __build_module(self):

        return nn.Sequential(
            nn.Conv2d(self.dim_in, self.dim_out, 1),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.net(X)


class Up(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False):

        ic = in_channels
        oc = out_channels
        super(Up,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2,
                                        mode='bilinear', align_corners=True)
        else:
            self.up = torch.nn.ConvTranspose2d(ic, oc,
                                               kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, X):
        x1, x2 = X
        x1 = self.up(x1)
        # bxcxhxw
        h_diff = x2.size()[2] - x1.size()[2]
        w_diff = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (w_diff // 2, w_diff - w_diff // 2,
                        h_diff // 2, h_diff - h_diff // 2))

        return torch.cat([x2, x1], dim=1)

class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.net = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, X):
        return self.net(X)



class PGS(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, kernel_sizes, strides):
        super(PGS, self).__init__()
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.__build_net()


    def __build_net(self):
        # contracting path
        self.conv1 = ConvBlock(self.dim_inputs[0], self.dim_outputs[0], self.strides[0], self.kernel_sizes[0])
        self.down1 = Down()
        self.conv2 = ConvBlock(self.dim_inputs[1], self.dim_outputs[1], self.strides[1], self.kernel_sizes[1])
        self.down2 = Down()
        self.conv3 = ConvBlock(self.dim_inputs[2], self.dim_outputs[2], self.strides[2], self.kernel_sizes[2])
        self.down3 = Down()
        self.conv4 = ConvBlock(self.dim_inputs[3], self.dim_outputs[3], self.strides[3], self.kernel_sizes[3])
        self.down4 = Down()

        # bottleneck

        self.conv5 = ConvBlock(self.dim_inputs[4], self.dim_outputs[4], self.strides[4], self.kernel_sizes[4])

        # Expanding path

        self.up1 = Up(self.dim_outputs[4], self.dim_outputs[4], False)
        self.conv6  = ConvBlock(self.dim_inputs[5], self.dim_outputs[5], self.strides[5], self.kernel_sizes[5])
        self.up2 = Up(self.dim_outputs[5], self.dim_outputs[5], False)
        self.conv7 = ConvBlock(self.dim_inputs[6], self.dim_outputs[6], self.strides[6], self.kernel_sizes[6])
        self.up3 = Up(self.dim_outputs[6], self.dim_outputs[6], False)
        self.conv8 = ConvBlock(self.dim_inputs[7], self.dim_outputs[7], self.strides[7], self.kernel_sizes[7])
        self.up4 = Up(self.dim_outputs[7], self.dim_outputs[7], False)
        self.conv9 = ConvBlock(self.dim_inputs[8], self.dim_outputs[8], self.strides[8], self.kernel_sizes[8])


        # classifiers
        print(self.dim_outputs[4])
        self.cls5 = CLS(self.dim_outputs[4], self.dim_inputs[0])
        self.cls6 = CLS(self.dim_outputs[5], self.dim_inputs[0])
        self.cls7 = CLS(self.dim_outputs[6], self.dim_inputs[0])
        self.cls8 = CLS(self.dim_outputs[7], self.dim_inputs[0])
        self.cls9 = CLS(self.dim_outputs[8], self.dim_inputs[0])  #main classifier

    def forward(self, X):
        # contracting path

        c1 = self.conv1(X)
        d1 = self.down1(c1)
        c2 = self.conv2(d1)
        d2 = self.down2(c2)
        c3 = self.conv3(d2)
        d3 = self.down3(c3)
        c4 = self.conv4(d3)
        d4 = self.down4(c4)

        # bottleneck

        c5 = self.conv5(d4)
        output5 = self.cls5(c5)

        # expanding path

        u1 = self.up1((c5, c4))
        c6 = self.conv6(u1)
        output6 = self.cls6(c6)

        u2 = self.up2((c6, c3))
        c7 = self.conv7(u2)
        output7 = self.cls7(c7)

        u3 = self.up3((c7, c2))
        c8 = self.conv8(u3)
        output8 = self.cls8(c8)

        u4 = self.up4((c8, c1))
        c9 = self.conv9(u4)
        output9 = self.cls9(c9)
        return output5, output6, output7, output8, output9

    def compute_loss(self, y_preds, y_true, loss_functions,  is_supervised):
        (sup_loss, unsup_loss) = loss_functions
        total_loss = 0
        if is_supervised:

            # return sup_loss(y_preds, y_true)

            for output in y_preds:
                ratio = int(np.round(y_true.shape[2] / output.shape[2]))
                maxpool = nn.MaxPool2d(kernel_size=2, stride=ratio, padding=0)
            #
                target = maxpool(y_true)
                if target.shape != output.shape:
                    h_diff = output.size()[2] - target.size()[2]
                    w_diff = output.size()[3] - target.size()[3]
            #
                    target = F.pad(target, (w_diff // 2, w_diff - w_diff // 2,
                                                                    h_diff // 2, h_diff - h_diff // 2))
            #
                    # print("SUPERVISED : padded target!!!")
            #
            #
                assert output.shape == target.shape, "output and target shape is not similar!!"
                total_loss += sup_loss(output, target)
        else:
            main_output = y_preds[-1].detach()

            for i in range(len(y_preds) - 1):
                # if out
                assert not (main_output.shape == y_preds[i].shape), "Wrong output: comparing main output with itself"
                with torch.no_grad():
                    ratio = int(np.round(main_output.shape[2] / y_preds[i].shape[2]))

                    maxpool = nn.MaxPool2d(kernel_size=2, stride=ratio, padding=0)

                    
                    pooled_main_output = maxpool(main_output)
                    if pooled_main_output.shape != y_preds[i].shape:
                        h_diff = y_preds[i].size()[2] - pooled_main_output.size()[2]
                        w_diff = y_preds[i].size()[3] - pooled_main_output.size()[3]

                        pooled_main_output = F.pad(pooled_main_output, (w_diff // 2, w_diff - w_diff // 2,
                                                                        h_diff // 2, h_diff - h_diff // 2))
                        print(pooled_main_output)
                        print("Unsupervised: Padded OUTPUT!!!")



                    assert pooled_main_output.shape == y_preds[i].shape, \
                        "Error! shapes has to be equal but got {} and {}".format(pooled_main_output.shape, y_preds[i].shape)
                total_loss += unsup_loss(y_preds[i], pooled_main_output)

        return total_loss



if __name__ == '__main__':
    inputs_dim = [1, 64, 96, 128, 256, 768, 384, 224, 160]
    outputs_dim = [64, 96, 128, 256, 512, 256, 128, 96, 64]
    kernels = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # dim_inputs, dim_outputs, kernel_sizes, strides):
    wnet = PGS(inputs_dim, outputs_dim, kernels, strides)
    # wnet.build()

    a = np.random.rand(20, 1, 212, 256)
    X = list(a)
    X = torch.FloatTensor(X)
    Y = wnet(X)
    # optimizer = torch.optim.SGD(wnet.parameters(), 0.001)
    print("hererere")
    for y in Y:
        print(y.shape)

    loss = wnet.compute_loss(Y, Y, (nn.MSELoss(),nn.MSELoss()),  False)
    print("loss")
    print(loss)