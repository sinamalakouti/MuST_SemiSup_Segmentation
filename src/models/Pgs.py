import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.uniform import Uniform
from dataset.Augmentation import Perturbator
import random


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


class CLS(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CLS, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.net = self.__build_module()

    def __build_module(self):
        return nn.Sequential(
            nn.Conv2d(self.dim_in, self.dim_out, 1)
        )

    def forward(self, X):
        logits = self.net(X)
        return logits


class Up(torch.nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=False):

        ic = in_channels
        oc = out_channels
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2,
                                        mode='bilinear', align_corners=True)
        else:
            self.up = torch.nn.ConvTranspose2d(ic, oc,
                                               kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, X, transformer=None):
        x1, x2 = X
        if transformer is None:
            x1 = self.up(x1)
        else:
            x1 = self.up(transformer(x1)[0])
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
    def __init__(self, dim_inputs, dim_outputs, kernel_sizes, strides, cfg):
        super(PGS, self).__init__()
        self.dim_inputs = dim_inputs
        self.dim_outputs = dim_outputs
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.config = cfg
        self.transformer = Perturbator(cfg)
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
        self.conv6 = ConvBlock(self.dim_inputs[5], self.dim_outputs[5], self.strides[5], self.kernel_sizes[5])
        self.up2 = Up(self.dim_outputs[5], self.dim_outputs[5], False)
        self.conv7 = ConvBlock(self.dim_inputs[6], self.dim_outputs[6], self.strides[6], self.kernel_sizes[6])
        self.up3 = Up(self.dim_outputs[6], self.dim_outputs[6], False)
        self.conv8 = ConvBlock(self.dim_inputs[7], self.dim_outputs[7], self.strides[7], self.kernel_sizes[7])
        self.up4 = Up(self.dim_outputs[7], self.dim_outputs[7], False)
        self.conv9 = ConvBlock(self.dim_inputs[8], self.dim_outputs[8], self.strides[8], self.kernel_sizes[8])

        # classifiers
        print(self.dim_outputs[4])
        self.cls5 = CLS(self.dim_outputs[4], self.dim_outputs[-1])
        self.cls6 = CLS(self.dim_outputs[5], self.dim_outputs[-1])
        self.cls7 = CLS(self.dim_outputs[6], self.dim_outputs[-1])
        self.cls8 = CLS(self.dim_outputs[7], self.dim_outputs[-1])
        self.cls9 = CLS(self.dim_outputs[8], self.dim_outputs[-1])  # main classifier

    def forward(self, X, is_supervised):
        type_unsup = 'layerwise'
        if is_supervised:
            sup_outputs = self.__fw_supervised(X)
            return sup_outputs, None

        elif type_unsup == 'layerwise':
            return self.__fw_unsupervised_layerwise2(X)

        else:

            # get supervised outputs

            self.training = False
            # with torch.no_grad():
            sup_outputs = self.__fw_supervised(X)

            # get unsupervised outputs

            self.training = True
            unsup_outputs = self.__fw_unsupervised(X)

            return sup_outputs, unsup_outputs

    def __fw_unsupervised_layerwise(self, X):
        cascade = False
        # contracting path
        c1, d1, c2, d2, c3, d3, c4, d4 = self.__fw_contracting_path(X)

        # bottleneck

        with torch.no_grad():
            c5_teach = self.__fw_bottleneck(d4).detach()
            output5_teach = self.cls5(c5_teach).detach()

        d4_stud, aug_output5_teach = self.transformer(d4, output5_teach, cascade=cascade)
        c5_stud = self.__fw_bottleneck(d4_stud)
        output5_stud = self.cls5(c5_stud)

        # expanding path
        up1 = self.__fw_up(c5_teach, c4, self.up1) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c5_stud, c4, self.up1)

        with torch.no_grad():
            c6_teach = self.__fw_expand_4layer(up1).detach()
            output6_teach = self.cls6(c6_teach).detach()

        aug_up1, aug_output6_teach = self.transformer(up1, output6_teach, cascade=cascade)
        c6_stud = self.__fw_expand_4layer(aug_up1)
        output6_stud = self.cls6(c6_stud)
        ######
        up2 = self.__fw_up(c6_teach, c3, self.up2) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c6_stud, c3, self.up2)

        with torch.no_grad():
            c7_teach = self.__fw_expand_3layer(up2).detach()
            output7_teach = self.cls7(c7_teach).detach()

        aug_up2, aug_output7_teach = self.transformer(up2, output7_teach, cascade=cascade)
        c7_stud = self.__fw_expand_3layer(aug_up2)
        output7_stud = self.cls7(c7_stud)

        #####
        up3 = self.__fw_up(c7_teach, c2, self.up3) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c7_stud, c2, self.up3)

        with torch.no_grad():
            c8_teach = self.__fw_expand_2layer(up3).detach()
            output8_teach = self.cls8(c8_teach).detach()

        aug_up3, aug_output8_teach = self.transformer(up3, output8_teach, cascade=cascade)
        c8_stud = self.__fw_expand_2layer(aug_up3)
        output8_stud = self.cls8(c8_stud)

        ####
        up4 = self.__fw_up(c8_teach, c1, self.up4) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c8_stud, c1, self.up4)

        with torch.no_grad():
            # output9 is the main output of the network
            c9_teach = self.__fw_expand_1layer(up4).detach()
            output9_teach = self.cls9(c9_teach).detach()

        aug_up4, aug_output9_teach = self.transformer(up4, output9_teach, cascade=cascade)
        c9_stud = self.__fw_expand_1layer(aug_up4)
        output9_stud = self.cls9(c9_stud)

        supervised_outputs = aug_output5_teach, aug_output6_teach, aug_output7_teach, aug_output8_teach, aug_output9_teach
        unsupervised_outputs = output5_stud, output6_stud, output7_stud, output8_stud, output9_stud
        return supervised_outputs, unsupervised_outputs

    def __fw_unsupervised_layerwise2(self, X):
        cascade = False
        # contracting path
        c1, d1, c2, d2, c3, d3, c4, d4 = self.__fw_contracting_path(X)

        # bottleneck

        with torch.no_grad():
            c5_teach = self.__fw_bottleneck(d4).detach()
            aug_output5_teach = self.cls5(c5_teach).detach()

        d4_stud, _ = self.transformer(d4, None, cascade=cascade)
        c5_stud = self.__fw_bottleneck(d4_stud)
        output5_stud = self.cls5(c5_stud)

        # expanding path
        teach_up1 = self.__fw_up(c5_teach, c4, self.up1,
                                 None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c5_stud, c4, self.up1, None)

        stud_up1 = self.__fw_up(c5_teach, c4, self.up1,
                                self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c5_stud, c4, self.up1, self.transformer)

        with torch.no_grad():
            c6_teach = self.__fw_expand_4layer(teach_up1).detach()
            aug_output6_teach = self.cls6(c6_teach).detach()

        c6_stud = self.__fw_expand_4layer(stud_up1)
        output6_stud = self.cls6(c6_stud)
        ######
        teach_up2 = self.__fw_up(c6_teach, c3, self.up2,
                                 None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c6_stud, c3, self.up2, None)

        stud_up2 = self.__fw_up(c6_teach, c3, self.up2,
                                self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c6_stud, c3, self.up2, self.transformer)

        with torch.no_grad():
            c7_teach = self.__fw_expand_3layer(teach_up2).detach()
            aug_output7_teach = self.cls7(c7_teach).detach()

        c7_stud = self.__fw_expand_3layer(stud_up2)
        output7_stud = self.cls7(c7_stud)

        #####
        teach_up3 = self.__fw_up(c7_teach, c2, self.up3, None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c7_stud, c2, self.up3, None)

        stud_up3 = self.__fw_up(c7_teach, c2, self.up3,
                                self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c7_stud, c2, self.up3, self.transformer)

        with torch.no_grad():
            c8_teach = self.__fw_expand_2layer(teach_up3).detach()
            aug_output8_teach = self.cls8(c8_teach).detach()

        c8_stud = self.__fw_expand_2layer(stud_up3)
        output8_stud = self.cls8(c8_stud)

        ####
        teach_up4 = self.__fw_up(c8_teach, c1, self.up4, None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c8_stud, c1, self.up4, None)

        stud_up4 = self.__fw_up(c8_teach, c1, self.up4,
                                self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(c8_stud, c1, self.up4, self.transformer)

        with torch.no_grad():
            # output9 is the main output of the network
            c9_teach = self.__fw_expand_1layer(teach_up4).detach()
            aug_output9_teach = self.cls9(c9_teach).detach()

        c9_stud = self.__fw_expand_1layer(stud_up4)
        output9_stud = self.cls9(c9_stud)

        supervised_outputs = aug_output5_teach, aug_output6_teach, aug_output7_teach, aug_output8_teach, aug_output9_teach
        unsupervised_outputs = output5_stud, output6_stud, output7_stud, output8_stud, output9_stud
        return supervised_outputs, unsupervised_outputs

    def __fw_supervised(self, X):

        c1, d1, c2, d2, c3, d3, c4, d4 = self.__fw_contracting_path(X)

        # bottleneck

        c5 = self.__fw_bottleneck(d4)
        output5 = self.cls5(c5)
        # expanding path
        # 4th expanding layer

        up1 = self.__fw_up(c5, c4, self.up1)
        c6 = self.__fw_expand_4layer(up1)
        output6 = self.cls6(c6)
        # 3rd expanding layer

        up2 = self.__fw_up(c6, c3, self.up2)
        c7 = self.__fw_expand_3layer(up2)
        output7 = self.cls7(c7)
        # 2nd expanding layer

        up3 = self.__fw_up(c7, c2, self.up3)
        c8 = self.__fw_expand_2layer(up3)
        output8 = self.cls8(c8)

        # 1st expanding layer

        up4 = self.__fw_up(c8, c1, self.up4)
        c9 = self.__fw_expand_1layer(up4)
        output9 = self.cls9(c9)

        return output5, output6, output7, output8, output9

    def __fw_contracting_path(self, X):
        c1 = self.conv1(X)
        d1 = self.down1(c1)
        c2 = self.conv2(d1)
        d2 = self.down2(c2)
        c3 = self.conv3(d2)
        d3 = self.down3(c3)
        c4 = self.conv4(d3)
        d4 = self.down4(c4)

        return c1, d1, c2, d2, c3, d3, c4, d4

    def __fw_bottleneck(self, X):
        c5 = self.conv5(X)
        # out = self.cls5(c5)
        return c5

    def __fw_up(self, X_expand, X_contract, up_module, noise_dist=None, is_supervised=True, transformer=None):
        if is_supervised:
            return up_module((X_expand, X_contract), transformer)
        else:
            # 1st augmentation
            noise_vector = noise_dist.sample(X_expand.shape[1:]).to(X_expand.device)  # .unsqueeze(0)
            noise_vector = noise_vector.reshape(X_expand.shape[1:])
            X_expand = X_expand.mul(noise_vector) + X_expand

            # 2nd augmentation

            noise_vector = noise_dist.sample(X_contract.shape[1:]).to(X_contract.device)  # .unsqueeze(0)
            noise_vector = noise_vector.reshape(X_contract.shape[1:])
            X_contract = X_contract.mul(noise_vector) + X_contract

            up = up_module((X_expand, X_contract))
            # noise_vector = noise_dist.sample(up.shape[1:]).to(up.device)  # .unsqueeze(0)
            # noise_vector = noise_vector.reshape(up.shape[1:])
            # up = up.mul(noise_vector) + up
            return up

    def __fw_expand_4layer(self, X):
        # u1 = self.up1((X_expand, X_contract))
        c6 = self.conv6(X)
        # output6 = self.cls6(c6)
        return c6

    def __fw_expand_3layer(self, X):

        # u2 = self.up2((X_expand, X_contract))
        c7 = self.conv7(X)
        # output7 = self.cls7(c7)
        return c7

    def __fw_expand_2layer(self, X):
        # u3 = self.up3((X_expand, X_contract))
        c8 = self.conv8(X)
        # output8 = self.cls8(c8)
        return c8

    def __fw_expand_1layer(self, X):
        # u4 = self.up4((X_expand, X_contract))
        c9 = self.conv9(X)
        # output9 = self.cls9(c9)
        return c9


def __fw_sup_loss(y_preds, y_true, loss_functions):
    (sup_loss, unsup_loss) = loss_functions
    total_loss = 0
    losses = []
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
        losses.append(sup_loss(output, target))
    total_loss = sum(losses)
    return total_loss


def __fw_self_unsup_loss(y_preds, loss_functions):
    main_output = y_preds[-1].detach()
    (_, unsup_loss) = loss_functions
    total_loss = 0
    losses = []
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
                # print(pooled_main_output)
                # print("Unsupervised: Padded OUTPUT!!!")

            assert pooled_main_output.shape == y_preds[i].shape, \
                "Error! shapes has to be equal but got {} and {}".format(pooled_main_output.shape,
                                                                         y_preds[i].shape)
        losses.append(total_loss + unsup_loss(y_preds[i], pooled_main_output))
    total_loss = sum(losses)
    return total_loss


def __fw_outputwise_unsup_loss(y_noisy, y_orig, loss_functions):
    (_, unsup_loss) = loss_functions
    total_loss = 0
    losses = []
    assert len(y_orig) == len(y_noisy), "Error! unsup_preds and sup_preds have to have same length"
    num_preds = len(y_orig)

    for i in range(num_preds):
        orig_pred = y_orig[i]
        noisy_pred = y_noisy[i]
        assert orig_pred.shape == noisy_pred.shape, "Error! for preds number {}, supervised and unsupervised" \
                                                    " prediction shape is not similar!".format(i)
        losses.append(unsup_loss(noisy_pred, orig_pred.detach()))
    total_loss = sum(losses)
    return total_loss


# def compute_loss(y_preds, y_true, loss_functions, is_supervised):
#     if is_supervised:
#         total_loss = __fw_sup_loss(y_preds, y_true, loss_functions)
#     else:
#         # for comparing outputs together!
#         # total_loss = self.__fw_self_unsup_loss(y_preds, loss_functions)
#
#         # consistency of original output and noisy output
#         total_loss = __fw_outputwise_unsup_loss(y_preds, y_true, loss_functions)
#
#     return total_loss


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

    loss = wnet.compute_loss(Y, Y, (nn.MSELoss(), nn.MSELoss()), False)
    print("loss")
    print(loss)
