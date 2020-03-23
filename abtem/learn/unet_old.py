

import numpy as np
import torch
import torch.nn as nn

from abtem.utils import BatchGenerator


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()

        layers = []

        if batch_norm:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(out_channels)]
        else:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]

        layers += [nn.ReLU(inplace=True)]

        if batch_norm:
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(out_channels)]
        else:
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)]

        layers += [nn.ReLU(inplace=True)]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p=dropout),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.drop = nn.Dropout(p=dropout)
        self.conv = DoubleConv(in_channels, out_channels, batch_norm=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.drop(x)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, init_features=16, dropout=.5, bilinear=True, activation=None):
        super().__init__()

        features = init_features
        self.out_channels = out_channels
        self.activation = activation

        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, 2 * features, dropout=dropout)
        self.down2 = Down(2 * features, 4 * features, dropout=dropout)
        self.down3 = Down(4 * features, 8 * features, dropout=dropout)
        self.down4 = Down(8 * features, 8 * features, dropout=dropout)
        self.up1 = Up(16 * features, 4 * features, bilinear, dropout=dropout)
        self.up2 = Up(8 * features, 2 * features, bilinear, dropout=dropout)
        self.up3 = Up(4 * features, features, bilinear, dropout=dropout)
        self.up4 = Up(2 * features, features, bilinear, dropout=dropout)
        self.out1 = nn.Conv2d(in_channels=features, out_channels=features // 2, kernel_size=1)
        self.out2 = nn.Conv2d(in_channels=features // 2, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out1(x)
        x = self.out2(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

    def get_device(self):
        return next(self.parameters()).device

    def predict_series(self, images, max_batch, num_samples=1):
        batch_generator = BatchGenerator(len(images), max_batch)

        if num_samples > 1:
            outputs = (np.zeros((images.shape[0],) + (self.out_channels,) + images.shape[2:], dtype=np.float32),
                       np.zeros((images.shape[0],) + (self.out_channels,) + images.shape[2:], dtype=np.float32))
            forward = lambda x: (self.mc_predict(x, num_samples=num_samples),)
        else:
            outputs = (np.zeros((images.shape[0],) + (self.out_channels,) + images.shape[2:], dtype=np.float32),)
            forward = lambda x: (self.forward(x),)

        for start, size in batch_generator.generate():
            batch_outputs = forward(images[start:start + size])

            for i, batch_output in enumerate(batch_outputs):
                outputs[i][start:start + size] = batch_output.detach().cpu().numpy()

        return outputs

    def mc_predict(self, images, num_samples):
        mc_output = np.zeros((num_samples,) + (images.shape[0],) + (self.out_channels,) + images.shape[2:])

        for i in range(num_samples):
            outputs = self.forward(images)
            mc_output[i] = outputs.cpu().detach().numpy()

        return np.mean(mc_output, 0), np.std(mc_output, 0)
