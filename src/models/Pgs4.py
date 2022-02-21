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
            # x2 = transformer(x2, None, perturbation_mode='F')[0]
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


class PGS4(nn.Module):
    def __init__(self, dim_inputs, dim_outputs, kernel_sizes, strides, cfg):
        super(PGS4, self).__init__()
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
        # self.conv5_stud = ConvBlock(self.dim_inputs[4], self.dim_outputs[4], self.strides[4], self.kernel_sizes[4])

        # Expanding path

        self.up1 = Up(self.dim_outputs[4], self.dim_outputs[4], False)
        # self.up1_stud = Up(self.dim_outputs[4], self.dim_outputs[4], False)
        self.conv6 = ConvBlock(self.dim_inputs[5], self.dim_outputs[5], self.strides[5], self.kernel_sizes[5])
        # self.conv6_stud = ConvBlock(self.dim_inputs[5], self.dim_outputs[5], self.strides[5], self.kernel_sizes[5])
        self.up2 = Up(self.dim_outputs[5], self.dim_outputs[5], False)
        # self.up2_stud = Up(self.dim_outputs[5], self.dim_outputs[5], False)
        self.conv7 = ConvBlock(self.dim_inputs[6], self.dim_outputs[6], self.strides[6], self.kernel_sizes[6])
        # self.conv7_stud = ConvBlock(self.dim_inputs[6], self.dim_outputs[6], self.strides[6], self.kernel_sizes[6])
        self.up3 = Up(self.dim_outputs[6], self.dim_outputs[6], False)
        # self.up3_stud = Up(self.dim_outputs[6], self.dim_outputs[6], False)
        self.conv8 = ConvBlock(self.dim_inputs[7], self.dim_outputs[7], self.strides[7], self.kernel_sizes[7])
        # self.conv8_stud = ConvBlock(self.dim_inputs[7], self.dim_outputs[7], self.strides[7], self.kernel_sizes[7])
        self.up4 = Up(self.dim_outputs[7], self.dim_outputs[7], False)
        # self.up4_stud = Up(self.dim_outputs[7], self.dim_outputs[7], False)
        self.conv9 = ConvBlock(self.dim_inputs[8], self.dim_outputs[8], self.strides[8], self.kernel_sizes[8])
        # self.conv9_stud = ConvBlock(self.dim_inputs[8], self.dim_outputs[8], self.strides[8], self.kernel_sizes[8])

        # INTERMEIDATE  decoders
        self.decode4_stud = CLS(self.dim_outputs[3], self.dim_outputs[-1])
        self.decode3_stud = CLS(self.dim_outputs[2], self.dim_outputs[-1])
        self.decode2_stud = CLS(self.dim_outputs[1], self.dim_outputs[-1])
        self.decode1_stud = CLS(self.dim_outputs[0], self.dim_outputs[-1])

        self.decode4_teach = CLS(self.dim_outputs[3], self.dim_outputs[-1])
        self.decode3_teach = CLS(self.dim_outputs[2], self.dim_outputs[-1])
        self.decode2_teach = CLS(self.dim_outputs[1], self.dim_outputs[-1])
        self.decode1_teach = CLS(self.dim_outputs[0], self.dim_outputs[-1])

        # classifiers
        print(self.dim_outputs[4])
        self.cls5 = CLS(self.dim_outputs[4], self.dim_outputs[-1])

        self.cls6 = CLS(self.dim_outputs[5], self.dim_outputs[-1])
        # self.decode6_stud = CLS(self.dim_outputs[5], self.dim_outputs[-1])
        self.cls7 = CLS(self.dim_outputs[6], self.dim_outputs[-1])
        # self.cls7_stud = CLS(self.dim_outputs[6], self.dim_outputs[-1])
        self.cls8 = CLS(self.dim_outputs[7], self.dim_outputs[-1])
        # self.cls8_stud = CLS(self.dim_outputs[7], self.dim_outputs[-1])
        self.cls9 = CLS(self.dim_outputs[8], self.dim_outputs[-1])  # main classifier
        # self.cls9_stud = CLS(self.dim_outputs[8], self.dim_outputs[-1])  # main classifier

    def forward(self, X, is_supervised):
        type_unsup = 'layerwise'
        # type_unsup = 'unsupervised'  # both feature_level (F) and input level (G) augmentation
        if is_supervised:
            main_outputs, decoder_outputs = self.__fw_supervised(X)
            return main_outputs, decoder_outputs

        elif type_unsup == 'layerwise':
            return self.__fw_unsupervised_layerwise2(X)
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
        c1_teach, d1, c2_teach, d2, c3_teach, d3, c4_teach, d4 = self.__fw_contracting_path(X)

        out4_teach = self.decode4_teach(c4_teach)
        out3_teach = self.decode3_teach(c3_teach)
        out2_teach = self.decode2_teach(c2_teach)
        out1_teach = self.decode1_teach(c1_teach)

        c1_stud = self.transformer(c1_teach, None, perturbation_mode='F')[0]
        c2_stud = self.transformer(c2_teach, None, perturbation_mode='F')[0]
        c3_stud = self.transformer(c3_teach, None, perturbation_mode='F')[0]
        c4_stud = self.transformer(c4_teach, None, perturbation_mode='F')[0]

        out4_stud = self.decode4_stud(c4_stud)
        out3_stud = self.decode3_stud(c3_stud)
        out2_stud = self.decode2_stud(c2_stud)
        out1_stud = self.decode1_stud(c1_stud)

        unsupervised_outputs = out4_stud, out3_stud, out2_stud, out1_stud
        supervised_outputs = out4_teach, out3_teach, out2_teach, out1_teach
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
        main_outputs = (output5, output6, output7, output8, output9)

        dec_out_6 = self.decode4_teach(c4)
        dec_out_7 = self.decode3_teach(c3)
        dec_out_8 = self.decode2_teach(c2)
        dec_out_9 = self.decode1_teach(c1)
        decoder_outputs = (dec_out_6, dec_out_7, dec_out_8, dec_out_9)
        return main_outputs, decoder_outputs

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
        # c5 = conv(X)
        # out = self.cls5(c5)
        return c5

    def __fw_up(self, X_expand, X_contract, up_module, transformer=None):
        return up_module((X_expand, X_contract), transformer)

    def __fw_expand_4layer(self, X):
        c6 = self.conv6(X)
        # c6 = conv(X)
        return c6

    def __fw_expand_3layer(self, X):
        c7 = self.conv7(X)
        # c7 = conv(X)
        return c7

    def __fw_expand_2layer(self, X):
        c8 = self.conv8(X)
        # c8 = conv(X)
        return c8

    def __fw_expand_1layer(self, X):
        c9 = self.conv9(X)
        # c9 = conv(X)
        return c9
