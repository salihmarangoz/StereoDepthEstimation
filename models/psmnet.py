import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import norm
import numpy as np


def conv2d_block(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, add_bn=True, add_relu=True, return_sequential=False):
    seq = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation = dilation, bias=bias),]

    if add_bn:
        seq.append(nn.BatchNorm2d(out_channels))

    if add_relu:
        seq.append(nn.ReLU(inplace=True))

    if return_sequential:
        return nn.Sequential(*seq)
    else:
        return tuple(seq)


def conv3d_block(in_channels, out_channels, kernel_size, stride, padding, bias=False, add_bn=True, add_relu=True, return_sequential=False):
    seq = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),]

    if add_bn:
        seq.append(nn.BatchNorm3d(out_channels))

    if add_relu:
        seq.append(nn.ReLU(inplace=True))

    if return_sequential:
        return nn.Sequential(*seq)
    else:
        return tuple(seq)


class Interpolate(nn.Module):
    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode
        
    def forward(self, x, size):
        return F.interpolate(x, size=size, mode=self.mode, align_corners=True)


class DisparityRegression(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.trainable = trainable
        if trainable:
            self.idx = nn.Parameter(torch.arange(0, 192, dtype=torch.float ,requires_grad=True))

    def forward(self, x):
        x = F.softmax(x, dim=2)
        if not self.trainable:
            idx = torch.arange(0, x.shape[2], dtype=x.dtype).to(x.device)
        else:
            idx = self.idx
        return torch.tensordot(x, idx, dims=([2,],[0,]))


class FeatureExtraction(nn.Module):
    def __init__(self, interpolation_mode="bilinear"):
        super().__init__()
        ################ CNN #########################
        self.conv0_x = nn.Sequential(*conv2d_block(3, 32, 3, stride=2, padding=1),
                                     *conv2d_block(32, 32, 3, stride=1, padding=1),
                                     *conv2d_block(32, 32, 3, stride=1, padding=1),)

        conv1_x = []
        for i in range(3): 
            conv1_x.append(nn.Sequential(*conv2d_block(32, 32, 3, stride=1, padding=1),
                                         *conv2d_block(32, 32, 3, stride=1, padding=1, add_relu=False),))
        self.conv1_x = nn.Sequential(*conv1_x)

        conv2_x = []
        conv2_x.append(nn.Sequential(*conv2d_block(32, 64, 3, stride=2, padding=1),
                                     *conv2d_block(64, 64, 3, stride=1, padding=1, add_relu=False),)) # todo: downsampling?
        for i in range(15):
            conv2_x.append(nn.Sequential(*conv2d_block(64, 64, 3, stride=1, padding=1),
                                         *conv2d_block(64, 64, 3, stride=1, padding=1, add_relu=False),))
        self.conv2_x = nn.Sequential(*conv2_x)

        conv3_x = []
        conv3_x.append(nn.Sequential(*conv2d_block(64, 128, 3, stride=1, padding=2, dilation=2),
                                     *conv2d_block(128, 128, 3, stride=1, padding=2, dilation=2, add_relu=False),)) # todo: downsampling?
        for i in range(2):
            conv3_x.append(nn.Sequential(*conv2d_block(128, 128, 3, stride=1, padding=2, dilation=2),
                                         *conv2d_block(128, 128, 3, stride=1, padding=2, dilation=2, add_relu=False),))
        self.conv3_x = nn.Sequential(*conv3_x)

        conv4_x = []
        for i in range(3): # todo: downsampling?
            conv4_x.append(nn.Sequential(*conv2d_block(128, 128, 3, stride=1, padding=4, dilation=4),
                                         *conv2d_block(128, 128, 3, stride=1, padding=4, dilation=4, add_relu=False),))
        self.conv4_x = nn.Sequential(*conv4_x)


        ################ SPP MODULE #########################
        self.branch_1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.branch_2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.branch_3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.branch_4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.upsampling = Interpolate(mode=interpolation_mode)

        self.fusion = nn.Sequential(*conv2d_block(320, 128, 3, stride=1, padding=1),
                                    nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False),)


    def forward(self, x):
        conv0_x = self.conv0_x(x)
        conv1_x = self.conv1_x(conv0_x)
        conv2_x = self.conv2_x(conv1_x)
        conv3_x = self.conv3_x(conv2_x)
        conv4_x = self.conv4_x(conv3_x)

        branch_1 = self.upsampling(self.branch_1(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))
        branch_2 = self.upsampling(self.branch_2(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))
        branch_3 = self.upsampling(self.branch_3(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))
        branch_4 = self.upsampling(self.branch_4(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))

        concat_layers = torch.cat((conv2_x, conv4_x, branch_1, branch_2, branch_3, branch_4), 1)

        fusion = self.fusion(concat_layers)

        return fusion



class CostVolume(nn.Module):
    def __init__(self, max_slide, stride=1):
        super().__init__()
        self.max_slide = max_slide
        self.stride = stride

    def forward(self, left_feats, right_feats, device=None): 
        B = left_feats.shape[0]
        C = left_feats.shape[1]
        H = left_feats.shape[2]
        W = left_feats.shape[3]

        if device is None:
            device = left_feats.device

        cost = torch.Tensor(B, C*2, self.max_slide//self.stride, H, W).to(device)

        for i in range(self.max_slide//self.stride):
            if(i==0):
                cost[:, :C, i, :, :] = left_feats
                cost[:, C:, i, :, :] = right_feats
            else:
                cost[:, :C, i, :, i*self.stride:] = left_feats[:, :, :, i*self.stride:]
                cost[:, C:, i, :, i*self.stride:] = right_feats[:, :, :, :-i*self.stride]
        return cost



class Hourglass(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self._3Dstackx_1 = nn.Sequential(*conv3d_block(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1),
                                         *conv3d_block(in_channels*2, in_channels*2, kernel_size=3, stride=1, padding=1, add_relu=False),) # todo: add_relu=False?
        self._3Dstackx_2 = nn.Sequential(*conv3d_block(in_channels*2, in_channels*2, kernel_size=3, stride=2, padding=1),
                                         *conv3d_block(in_channels*2, in_channels*2, kernel_size=3, stride=1, padding=1),)
        self._3Dstackx_3 = nn.Sequential(nn.ConvTranspose3d(in_channels*2, in_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                         nn.BatchNorm3d(in_channels*2),)
        self._3Dstackx_4 = nn.Sequential(nn.ConvTranspose3d(in_channels*2, in_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                         nn.BatchNorm3d(in_channels),)


    def forward(self, x, add1=None ,add2=None, add3=None, add4=None, add_layer1_to_layer3=False):
        layer1 = self._3Dstackx_1(x)
        if add1 is not None:
            layer1 = layer1 + add1

        layer2 = self._3Dstackx_2(layer1)
        if add2 is not None:
            layer2 = layer2 + add2

        layer3 = self._3Dstackx_3(layer2)
        if add_layer1_to_layer3:
            if add3 is not None:
                raise Exception("add_layer1_to_layer3 and add3 cant be defined at the same time!")
            add3 = layer1
        if add3 is not None:
            layer3 = layer3 + add3

        layer4 = self._3Dstackx_4(layer3)
        if add4 is not None:
            layer4 = layer4 + add4
        return layer1, layer2, layer3, layer4


class PSMNet(nn.Module):
    def __init__(self, maxdisp=192, costvolume_slide=192//4, costvolume_stride=1, name="psm_stacked_hourglass"):
        super().__init__()
        self.maxdisp = maxdisp
        self.name = name

        self.feature_extraction = FeatureExtraction()

        self.cost_volume = CostVolume(costvolume_slide, costvolume_stride)

        self._3Dconv0 = nn.Sequential(*conv3d_block(64, 32, 3, stride=1, padding=1),
                                      *conv3d_block(32, 32, 3, stride=1, padding=1),)
        self._3Dconv1 = nn.Sequential(*conv3d_block(32, 32, 3, stride=1, padding=1),
                                      *conv3d_block(32, 32, 3, stride=1, padding=1, add_relu=False),)

        self._3Dstack1 = Hourglass(32)
        self._3Dstack2 = Hourglass(32)
        self._3Dstack3 = Hourglass(32)

        self.output_1 = nn.Sequential(*conv3d_block(32, 32, 3, stride=1, padding=1),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.output_2 = nn.Sequential(*conv3d_block(32, 32, 3, stride=1, padding=1),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.output_3 = nn.Sequential(*conv3d_block(32, 32, 3, stride=1, padding=1),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.upsampling = Interpolate(mode="trilinear") # todo: bilinear?

        self.before_regression = nn.Identity() # for future modifications

        self.disparity_regression = DisparityRegression()

        self.after_regression = nn.Identity() # for future modifications



    def forward(self, left, right):
        # concat left and right images, calculate feature extraction in one go, split the output
        left_features, right_features = torch.chunk( self.feature_extraction( torch.cat((left, right), dim=0) ), chunks=2, dim=0)

        cost = self.cost_volume(left_features, right_features)

        _3Dconv0 = self._3Dconv0(cost)
        _3Dconv1 = self._3Dconv1(_3Dconv0)

        _3Dstack1_1, _, _3Dstack1_3, _3Dstack1_4 = self._3Dstack1(_3Dconv1                                       , add4=_3Dconv1, add_layer1_to_layer3=True)
        _,           _, _3Dstack2_3, _3Dstack2_4 = self._3Dstack2(_3Dstack1_4, add1=_3Dstack1_3, add3=_3Dstack1_1, add4=_3Dconv1)
        _,           _,           _, _3Dstack3_4 = self._3Dstack3(_3Dstack2_4, add1=_3Dstack2_3, add3=_3Dstack1_1, add4=_3Dconv1)

        output_1 = self.output_1(_3Dstack1_4)
        output_2 = self.output_2(_3Dstack2_4) + output_1
        output_3 = self.output_3(_3Dstack3_4) + output_2

        output_1 = self.upsampling(output_1, (self.maxdisp, left.size()[2], left.size()[3]))
        output_2 = self.upsampling(output_2, (self.maxdisp, left.size()[2], left.size()[3]))
        output_3 = self.upsampling(output_3, (self.maxdisp, left.size()[2], left.size()[3]))


        if self.training:
            output_1 = self.after_regression( self.disparity_regression( self.before_regression(output_1) ) )
            output_2 = self.after_regression( self.disparity_regression( self.before_regression(output_2) ) )
        output_3 = self.after_regression( self.disparity_regression( self.before_regression(output_3) ) )

        if self.training:
            return output_1, output_2, output_3
        return output_3



########################

class FeatureExtractionPlusResNet50(nn.Module):
    def __init__(self, interpolation_mode="bilinear"):
        super().__init__()

        self.resnet50 = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True)
        self.resnet50.classifier = nn.Identity()
        self.resnet50.aux_classifier = nn.Identity()
        self.downsampling = Interpolate(mode="bicubic")

        ################ CNN #########################
        self.conv0_x = nn.Sequential(*conv2d_block(3, 32, 3, stride=2, padding=1),
                                     *conv2d_block(32, 32, 3, stride=1, padding=1),
                                     *conv2d_block(32, 32, 3, stride=1, padding=1),)

        conv1_x = []
        for i in range(3): 
            conv1_x.append(nn.Sequential(*conv2d_block(32, 32, 3, stride=1, padding=1),
                                         *conv2d_block(32, 32, 3, stride=1, padding=1, add_relu=False),))
        self.conv1_x = nn.Sequential(*conv1_x)

        conv2_x = []
        conv2_x.append(nn.Sequential(*conv2d_block(32, 64, 3, stride=2, padding=1),
                                     *conv2d_block(64, 64, 3, stride=1, padding=1, add_relu=False),)) # todo: downsampling?
        for i in range(15):
            conv2_x.append(nn.Sequential(*conv2d_block(64, 64, 3, stride=1, padding=1),
                                         *conv2d_block(64, 64, 3, stride=1, padding=1, add_relu=False),))
        self.conv2_x = nn.Sequential(*conv2_x)

        conv3_x = []
        conv3_x.append(nn.Sequential(*conv2d_block(64, 128, 3, stride=1, padding=2, dilation=2),
                                     *conv2d_block(128, 128, 3, stride=1, padding=2, dilation=2, add_relu=False),)) # todo: downsampling?
        for i in range(2):
            conv3_x.append(nn.Sequential(*conv2d_block(128, 128, 3, stride=1, padding=2, dilation=2),
                                         *conv2d_block(128, 128, 3, stride=1, padding=2, dilation=2, add_relu=False),))
        self.conv3_x = nn.Sequential(*conv3_x)

        conv4_x = []
        for i in range(3): # todo: downsampling?
            conv4_x.append(nn.Sequential(*conv2d_block(128, 128, 3, stride=1, padding=4, dilation=4),
                                         *conv2d_block(128, 128, 3, stride=1, padding=4, dilation=4, add_relu=False),))
        self.conv4_x = nn.Sequential(*conv4_x)


        ################ SPP MODULE #########################
        self.branch_1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.branch_2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.branch_3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.branch_4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                      *conv2d_block(128, 32, 1, stride=1, padding=0),)

        self.upsampling = Interpolate(mode=interpolation_mode)

        self.fusion = nn.Sequential(*conv2d_block(320+2048, 128, 3, stride=1, padding=1),
                                    nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False),)


    def forward(self, x):
        conv0_x = self.conv0_x(x)
        conv1_x = self.conv1_x(conv0_x)
        conv2_x = self.conv2_x(conv1_x)
        conv3_x = self.conv3_x(conv2_x)
        conv4_x = self.conv4_x(conv3_x)

        branch_1 = self.upsampling(self.branch_1(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))
        branch_2 = self.upsampling(self.branch_2(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))
        branch_3 = self.upsampling(self.branch_3(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))
        branch_4 = self.upsampling(self.branch_4(conv4_x), (conv4_x.size()[2],conv4_x.size()[3]))

        resnet50_out = self.resnet50(self.downsampling(x, (x.size()[2]//4, x.size()[3]//4)))
        x_sem = resnet50_out['out']

        concat_layers = torch.cat((conv2_x, conv4_x, branch_1, branch_2, branch_3, branch_4, x_sem), 1)

        fusion = self.fusion(concat_layers)

        return fusion


class HarderDisparityRegression(nn.Module):
    def __init__(self, multiplier=1.0, scale=50):
        super().__init__()
        self.multiplier = multiplier
        self.scale = scale

    def forward(self, x):
        if x.shape[0] > 1:
            raise Exception("Only batch_size=1 is supported!")

        x = F.softmax(x, dim=2)
        max_idx = torch.argmax(x, dim=2)#.reshape((1,1,1,256,512))
        dist = self._norm_pdf(x, loc=max_idx, scale=self.scale)
        x = (x*dist) / torch.sum(x*dist, dim=2)
        idx = torch.arange(0, x.shape[2], dtype=x.dtype, device=x.device) * self.multiplier
        return torch.tensordot(x, idx, dims=([2,],[0,]))


    def _norm_pdf(self, x, loc=0, scale=1):
        #return torch.ones(x.shape).to(x.device) / 192.0
        return np.e**(-((x-loc)/scale)**2/2)