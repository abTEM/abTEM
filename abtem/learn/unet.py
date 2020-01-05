from collections import OrderedDict
from abtem.utils import BatchGenerator
import numpy as np
import torch
import torch.nn as nn
from abtem.learn.partialconv2d import PartialConv2d
import torch.nn.functional as F


def extract_patches(images, patch_size):
    dims = len(images.shape)
    patches = images.unfold(dims - 2, patch_size, patch_size).unfold(dims - 1, patch_size, patch_size)
    return patches


def patch_means(images, patch_size):
    if patch_size == 1:
        return images

    return torch.mean(extract_patches(images, patch_size), (-2, -1))


def multi_patch_loss(input, target, loss_obj, patch_sizes, weights=None):
    loss = 0.
    for patch_size in patch_sizes:
        if weights:
            loss += torch.mean(loss_obj(patch_means(input, patch_size), patch_means(target, patch_size)) * weights)
        else:
            loss += torch.mean(loss_obj(patch_means(input, patch_size), patch_means(target, patch_size)))

    return loss / len(patch_sizes)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()

        layers = []

        if batch_norm:
            layers += [PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(out_channels)]
        else:
            layers += [PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]

        layers += [nn.ReLU(inplace=True)]

        if batch_norm:
            layers += [PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(out_channels)]
        else:
            layers += [PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)]

        layers += [nn.ReLU(inplace=True)]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

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
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
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


class Head(nn.Module):

    def __init__(self, features, nclasses, activation):
        super().__init__()
        layer1 = nn.Conv2d(in_channels=features, out_channels=features // 2, kernel_size=1)
        layer2 = nn.Conv2d(in_channels=features // 2, out_channels=nclasses, kernel_size=1)

        self.layers = nn.Sequential(layer1, nn.ReLU(inplace=True), layer2)

        self.activation = activation
        self._nclasses = nclasses

    @property
    def out_channels(self):
        return self._nclasses

    def forward(self, x):
        x = self.layers(x)
        # x = self.conv(x)
        return self.activation(x)


class UNet(nn.Module):

    def __init__(self, heads, in_channels=1, init_features=16, dropout=.5, bilinear=True):
        super().__init__()

        features = init_features

        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, 2 * features, dropout=dropout)
        self.down2 = Down(2 * features, 4 * features, dropout=dropout)
        self.down3 = Down(4 * features, 8 * features, dropout=dropout)
        self.down4 = Down(8 * features, 8 * features, dropout=dropout)
        self.up1 = Up(16 * features, 4 * features, bilinear, dropout=dropout)
        self.up2 = Up(8 * features, 2 * features, bilinear, dropout=dropout)
        self.up3 = Up(4 * features, features, bilinear, dropout=dropout)
        self.up4 = Up(2 * features, features, bilinear, dropout=dropout)

        self.heads = heads

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

        return {key: head(x) for key, head in self.heads.items()}

    def predict_series(self, images, max_batch, num_samples=1):
        batch_generator = BatchGenerator(len(images), max_batch)

        outputs = {}
        for key, head in self.heads.items():

            if num_samples > 1:
                outputs[key + '_mean'] = np.zeros((images.shape[0],) + (head.out_channels,) + images.shape[2:])
                outputs[key + '_std'] = np.zeros((images.shape[0],) + (head.out_channels,) + images.shape[2:])
            else:
                outputs[key] = np.zeros((images.shape[0],) + (head.out_channels,) + images.shape[2:])

        for start, size in batch_generator.generate():

            if num_samples > 1:
                batch_outputs = self.mc_predict(images[start:start + size], num_samples=num_samples)

            else:
                batch_outputs = self.forward(images[start:start + size])
                batch_outputs = {key: output.detach().cpu() for key, output in batch_outputs.items()}

            for key in outputs.keys():
                outputs[key][start:start + size] = batch_outputs[key]

        return outputs

    def mc_predict(self, images, num_samples):
        mc_outputs = {key: np.zeros((num_samples,) + (images.shape[0],) + (head.out_channels,) + images.shape[2:])
                      for key, head in self.heads.items()}

        for i in range(num_samples):
            outputs = self.forward(images)
            for key in mc_outputs.keys():
                mc_outputs[key][i] = outputs[key].cpu().detach().numpy()

        output = {}
        for key, mc_output in mc_outputs.items():
            output[key + '_mean'] = np.mean(mc_output, 0)
            output[key + '_std'] = np.std(mc_output, 0)

        return output

    def all_parameters(self):
        parameters = list(self.parameters())
        for model in self.heads.values():
            parameters += list(model.parameters())
        return parameters

    def save_all(self, path):
        state_dicts = {'unet': self.state_dict()}
        for key, model in self.heads.items():
            state_dicts[key] = model.state_dict()
        torch.save(state_dicts, path)

    def load_all(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['unet'])
        for key, model in self.heads.items():
            model.load_state_dict(checkpoint[key])

    def all_to(self, device):
        self.to(device)
        for model in self.heads.values():
            model.to(device)
