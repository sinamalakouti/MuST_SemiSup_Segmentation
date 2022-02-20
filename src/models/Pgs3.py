import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Perturbations import Perturbator


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

    def forward(self, X, transformer):
        x1, x2 = X
        if transformer is None:
            x1 = self.up(x1)
        else:
            # x1 = self.up(transformer(x1, None, perturbation_mode='F')[0])
            x1 = transformer(self.up(x1), None, perturbation_mode='F')[0]
            x2 = transformer(x2, None, perturbation_mode='F')[0]
        # x1 = self.up(x1)
        # bxcxhxw
        h_diff = x2.size()[2] - x1.size()[2]
        w_diff = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (w_diff // 2, w_diff - w_diff // 2,
                        h_diff // 2, h_diff - h_diff // 2))
        res = torch.cat([x2, x1], dim=1)
        return res
        # if transformer is None:
        #     return res


"""
        return res, transformer(res, None, perturbation_mode='F')[0]
"""


class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.net = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, X):
        return self.net(X)


class PGS3(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, kernel_sizes, strides, cfg):
        super(PGS3, self).__init__()
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

        # self.conv5_teach = ConvBlock(self.dim_inputs[4], self.dim_outputs[4], self.strides[4], self.kernel_sizes[4])
        # self.conv5_stud = ConvBlock(self.dim_inputs[4], self.dim_outputs[4], self.strides[4], self.kernel_sizes[4])
        #
        # # Expanding path
        #
        self.up1_teach = Up(self.dim_outputs[4], self.dim_outputs[4], False)
        self.up1_stud = Up(self.dim_outputs[4], self.dim_outputs[4], False)
        # self.conv6_teach = ConvBlock(self.dim_inputs[5], self.dim_outputs[5], self.strides[5], self.kernel_sizes[5])
        # self.conv6_stud = ConvBlock(self.dim_inputs[5], self.dim_outputs[5], self.strides[5], self.kernel_sizes[5])
        self.up2_teach = Up(self.dim_outputs[5], self.dim_outputs[5], False)
        self.up2_stud = Up(self.dim_outputs[5], self.dim_outputs[5], False)
        # self.conv7_teach = ConvBlock(self.dim_inputs[6], self.dim_outputs[6], self.strides[6], self.kernel_sizes[6])
        # self.conv7_stud = ConvBlock(self.dim_inputs[6], self.dim_outputs[6], self.strides[6], self.kernel_sizes[6])
        self.up3_teach = Up(self.dim_outputs[6], self.dim_outputs[6], False)
        self.up3_stud = Up(self.dim_outputs[6], self.dim_outputs[6], False)
        # self.conv8_teach = ConvBlock(self.dim_inputs[7], self.dim_outputs[7], self.strides[7], self.kernel_sizes[7])
        # self.conv8_stud = ConvBlock(self.dim_inputs[7], self.dim_outputs[7], self.strides[7], self.kernel_sizes[7])
        self.up4_teach = Up(self.dim_outputs[7], self.dim_outputs[7], False)
        self.up4_stud = Up(self.dim_outputs[7], self.dim_outputs[7], False)
        # self.conv9_teach = ConvBlock(self.dim_inputs[8], self.dim_outputs[8], self.strides[8], self.kernel_sizes[8])
        # self.conv9_stud = ConvBlock(self.dim_inputs[8], self.dim_outputs[8], self.strides[8], self.kernel_sizes[8])

        # classifiers
        print(self.dim_outputs[4])
        self.cls5_teach = CLS(self.dim_inputs[4], self.dim_outputs[-1])
        self.cls5_stud = CLS(self.dim_inputs[4], self.dim_outputs[-1])
        self.cls6_teach = CLS(self.dim_inputs[5], self.dim_outputs[-1])
        self.cls6_stud = CLS(self.dim_inputs[5], self.dim_outputs[-1])
        self.cls7_teach = CLS(self.dim_inputs[6], self.dim_outputs[-1])
        self.cls7_stud = CLS(self.dim_inputs[6], self.dim_outputs[-1])
        self.cls8_teach = CLS(self.dim_inputs[7], self.dim_outputs[-1])
        self.cls8_teach = CLS(self.dim_inputs[7], self.dim_outputs[-1])
        self.cls9_teach = CLS(self.dim_inputs[8], self.dim_outputs[-1])  # main classifier
        self.cls9_stud = CLS(self.dim_inputs[8], self.dim_outputs[-1])  # main classifier

    def forward(self, X, is_supervised):
        type_unsup = 'layerwise'
        # type_unsup = 'unsupervised'  # both feature_level (F) and input level (G) augmentation
        if is_supervised:
            sup_outputs = self.__fw_supervised(X)
            return sup_outputs, None

        elif type_unsup == 'layerwise':
            return self.__fw_unsupervised_layerwise5(X)
            # return self.__fw_unsupervised_layerwislayerwise2e2(X)

        elif type_unsup == 'unsupervised':
            return self.__fw_unsupervised_feautre_sapce(X)

        else:

            # get supervised outputs

            self.training = False
            # with torch.no_grad():
            sup_outputs = self.__fw_supervised(X)

            # get unsupervised outputs

            self.training = True
            unsup_outputs = self.__fw_unsupervised(X)

            return sup_outputs, unsup_outputs

    def __fw_unsupervised_layerwise2(self, X):  # only_feature space aug

        # contracting path
        c1, d1, c2, d2, c3, d3, c4, d4 = self.__fw_contracting_path(X)

        # bottleneck

        with torch.no_grad():
            # c5_teach = self.__fw_bottleneck(d4, self.conv5_teach).detach()
            aug_output5_teach = self.cls5_teach(d4).detach()

        d4_stud, _ = self.transformer(d4, None, perturbation_mode='F')
        # c5_stud = self.__fw_bottleneck(d4_stud, self.conv5_stud)
        output5_stud = self.cls5_stud(d4_stud)

        # expanding path
        teach_up1 = self.__fw_up(aug_output5_teach, c4, self.up1_teach,
                                 transformer=None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(aug_output5_teach, c4, self.up1_teach, transformer=None)

        stud_up1 = self.__fw_up(output5_stud, c4, self.up1_stud,
                                transformer=self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(output5_stud, c4, self.up1_stud, transformer=self.transformer)

        with torch.no_grad():
            # c6_teach = self.__fw_expand_4layer(teach_up1,self.conv6_teach).detach()
            aug_output6_teach = self.cls6_teach(teach_up1).detach()

        # c6_stud = self.__fw_expand_4layer(stud_up1, self.conv6_stud)
        output6_stud = self.cls6_stud(stud_up1)
        ######
        teach_up2 = self.__fw_up(aug_output6_teach, c3, self.up2_teach,
                                 transformer=None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(aug_output6_teach, c3, self.up2_teach, transformer=None)

        stud_up2 = self.__fw_up(output6_stud, c3, self.up2_stud,
                                transformer=self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(output6_stud, c3, self.up2_stud, transformer=self.transformer)

        with torch.no_grad():
            # c7_teach = self.__fw_expand_3layer(teach_up2, self.conv7_teach).detach()
            aug_output7_teach = self.cls7_teach(teach_up2).detach()

        # c7_stud = self.__fw_expand_3layer(stud_up2,self.conv7_stud)
        output7_stud = self.cls7_stud(stud_up2)

        #####
        teach_up3 = self.__fw_up(aug_output7_teach, c2, self.up3_teach,
                                 transformer=None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(aug_output7_teach, c2, self.up3_teach, transformer=None)

        stud_up3 = self.__fw_up(output7_stud, c2, self.up3_stud,
                                transformer=self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(output7_stud, c2, self.up3_stud, transformer=self.transformer)

        with torch.no_grad():
            # c8_teach = self.__fw_expand_2layer(teach_up3,self.conv8_teach).detach()
            aug_output8_teach = self.cls8_teach(teach_up3).detach()

        # c8_stud = self.__fw_expand_2layer(stud_up3,self.conv8_stud)
        output8_stud = self.cls8_stud(stud_up3)

        ####
        teach_up4 = self.__fw_up(aug_output8_teach, c1, self.up4_teach,
                                 transformer=None) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(aug_output8_teach, c1, self.up4_teach, transformer=None)

        stud_up4 = self.__fw_up(output8_stud, c1, self.up4_stud,
                                transformer=self.transformer) if self.config.information_passing_strategy == 'teacher' \
            else self.__fw_up(output8_stud, c1, self.up4_stud, transformer=self.transformer)

        with torch.no_grad():
            # output9 is the main output of the network
            # c9_teach = self.__fw_expand_1layer(teach_up4,self.conv9_teach).detach()
            aug_output9_teach = self.cls9_teach(teach_up4).detach()

        # c9_stud = self.__fw_expand_1layer(stud_up4,self.conv9_stud)
        output9_stud = self.cls9_stud(stud_up4)

        supervised_outputs = aug_output5_teach, aug_output6_teach, aug_output7_teach, aug_output8_teach, aug_output9_teach
        unsupervised_outputs = output5_stud, output6_stud, output7_stud, output8_stud, output9_stud
        return supervised_outputs, unsupervised_outputs


    def __fw_supervised(self, X):

        c1, d1, c2, d2, c3, d3, c4, d4 = self.__fw_contracting_path(X)

        # bottleneck

        # c5 = self.__fw_bottleneck(d4)
        output5 = self.cls5_teach(d4)
        # expanding path
        # 4th expanding layer

        up1 = self.__fw_up(output5, c4, self.up1_teach)
        # c6 = self.__fw_expand_4layer(up1)
        output6 = self.cls6_tech(up1)
        # 3rd expanding layer

        up2 = self.__fw_up(output6, c3, self.up2_teach)
        # c7 = self.__fw_expand_3layer(up2)
        output7 = self.cls7_teach(up2)
        # 2nd expanding layer

        up3 = self.__fw_up(output7, c2, self.up3_teach)
        # c8 = self.__fw_expand_2layer(up3)
        output8 = self.cls8_teach(up3)

        # 1st expanding layer

        up4 = self.__fw_up(output8, c1, self.up4_teach)
        # c9 = self.__fw_expand_1layer(up4)
        output9 = self.cls9_teach(up4)

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

    def __fw_up(self, X_expand, X_contract, up_module, transformer=None):
        return up_module((X_expand, X_contract), transformer)
