import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Dilation_conv(nn.Module):
    def __init__(self, cin, cout):
        super(Dilation_conv, self).__init__()
        self.dilated1_conv = nn.Sequential(
                             nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, stride=1, padding="same", dilation=1),
                             nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))
        self.dilated3_conv = nn.Sequential(
                             nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, stride=1, padding="same", dilation=3),
                             nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))
        self.dilated5_conv = nn.Sequential(
                             nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, stride=1, padding="same", dilation=5),
                             nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.dilated1_conv(x)
        out2 = self.dilated3_conv(x)
        out3 = self.dilated5_conv(x)

        out = torch.add(out1, out2)
        out = torch.add(out, out3)

        return self.act(out)


class UCMBlock_encoder(nn.Module):
    def __init__(self, input, out):
        super(UCMBlock_encoder, self).__init__()
        self.conv1 = nn.Sequential(
                             nn.Conv3d(input, input, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm3d(input),
                             nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
                             nn.Conv3d(input, input, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm3d(input),
                             nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
                             nn.Conv3d(input, input, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm3d(input))
        self.conv4 = nn.Sequential(
                             nn.Conv3d(input, out, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm3d(out),
                             nn.ReLU(inplace=True))

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)

        out += residual
        out = self.conv4(out)


        return out


class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        y += x
        return self.act2(y)

class DecoderResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm3 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act3 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        residual = y
        y = self.conv2(y)
        y = self.act2(self.norm2(y))
        y = self.conv3(y)
        y = self.norm3(y)
        y = residual + y
        return self.act3(y)

class DecoderResBlock24(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm3 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act3 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        residual = y
        y = self.conv2(y)
        y = self.act2(self.norm2(y))
        y = self.conv3(y)
        y = self.norm3(y)
        y = residual + y
        return y

class DownsamolingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=2, use_1x1conv=True):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size=(2,2,2), mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

class InputChannel_project(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)

        self.conv3 = nn.Conv3d(input_channels, output_channels, 1, stride=stride, padding="valid")
        self.norm3 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.act1(self.norm1(y1))
        y1 = self.norm2(self.conv2(y1))

        y2 = self.conv3(x)
        y2 = self.norm3(y2)

        y = torch.add(y1, y2)
        return self.act2(y)
