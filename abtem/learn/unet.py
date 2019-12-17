import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DensityMap(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.conv = PartialConv2d(in_channels=features, out_channels=1, kernel_size=1)

    @property
    def out_channels(self):
        return 1

    def forward(self, x, mask):
        x, _ = self.conv(x, mask)
        return torch.sigmoid(x)


class ClassificationMap(nn.Module):

    def __init__(self, features, nclasses):
        super().__init__()
        self.conv = PartialConv2d(in_channels=features, out_channels=nclasses, kernel_size=1)
        self._nclasses = nclasses

    @property
    def out_channels(self):
        return self._nclasses

    def forward(self, x, mask):
        x, _ = self.conv(x, mask)
        return nn.Softmax2d()(x)


class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, multi_channel=True,
                 return_mask=True):

        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)

        self.return_mask = return_mask
        self.multi_channel = multi_channel

        if self.multi_channel:
            self.mask_conv_kernel = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                               self.kernel_size[1])
        else:
            self.mask_conv_kernel = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_window_size = np.prod(list(self.mask_conv_kernel.shape[1:]))

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input: Tensor, mask_in=None):
        assert len(input.shape) == 4

        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.mask_conv_kernel.type() != input.type():
                    self.mask_conv_kernel = self.mask_conv_kernel.to(input)

                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(input.shape).to(input)
                    else:
                        mask = torch.ones(1, 1, input.shape[2], input.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.mask_conv_kernel, bias=None, stride=self.stride,
                                            padding=self.padding)

                self.mask_ratio = self.slide_window_size / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super().forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConvUNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, normalize=True, dropout=0.0, bias=None):
        super().__init__()

        padding = kernel_size // 2
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_in')

        if normalize:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        self.activation = nn.LeakyReLU(0.2)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, mask):
        x, mask = self.conv(x, mask)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x, mask


class PartialConvUNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalize=True, dropout=0.0, bias=None):
        super().__init__()

        padding = kernel_size // 2
        self.conv = PartialConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)

        nn.init.kaiming_normal_(self.conv.weight, a=0, mode='fan_in')

        if normalize:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        self.activation = nn.ReLU()

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, skip_input, mask=None, skip_mask=None):

        if mask is None:
            assert skip_mask is None

        x = F.interpolate(x, scale_factor=2)
        if mask is not None:
            mask = F.interpolate(mask, scale_factor=2, mode='nearest')

        x = torch.cat((x, skip_input), dim=1)
        if mask is not None:
            mask = torch.cat((mask, skip_mask), dim=1)

        x, mask = self.conv(x, mask)
        if self.norm:
            x = self.norm(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        return x, mask


class UNet(nn.Module):

    def __init__(self, mappers, in_channels=1, init_features=16, dropout=.5):
        super().__init__()

        features = init_features

        self.encoder1 = PartialConvUNetEncoder(in_channels, features, 5, stride=1)
        self.encoder2 = PartialConvUNetEncoder(features, features * 2, dropout=dropout)
        self.encoder3 = PartialConvUNetEncoder(features * 2, features * 4, dropout=dropout)
        self.encoder4 = PartialConvUNetEncoder(features * 4, features * 8, dropout=dropout)

        self.bottleneck = PartialConvUNetEncoder(features * 8, features * 16, dropout=dropout)

        self.decoder4 = PartialConvUNetDecoder(features * (16 + 8), features * 8, dropout=dropout)
        self.decoder3 = PartialConvUNetDecoder(features * (8 + 4), features * 4, dropout=dropout)
        self.decoder2 = PartialConvUNetDecoder(features * (4 + 2), features * 2, dropout=dropout)
        self.decoder1 = PartialConvUNetDecoder(features * (2 + 1), features)

        self.mappers = mappers

    def all_parameters(self):
        parameters = list(self.parameters())
        for model in self.mappers.values():
            parameters += list(model.parameters())
        return parameters

    def save_all(self, path):
        state_dicts = {'unet': self.state_dict()}
        for key, model in self.mappers.items():
            state_dicts[key] = model.state_dict()
        torch.save(state_dicts, path)

    def load_all(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['unet'])
        for key, model in self.mappers.items():
            model.load_state_dict(checkpoint[key])

    def all_to(self, device):
        self.to(device)

        for model in self.mappers.values():
            model.to(device)

    def forward(self, x, mask):
        d1, d1_mask = self.encoder1(x, mask)
        d2, d2_mask = self.encoder2(d1, d1_mask)
        d3, d3_mask = self.encoder3(d2, d2_mask)
        d4, d4_mask = self.encoder4(d3, d3_mask)

        bottleneck, bottleneck_mask = self.bottleneck(d4, d4_mask)

        u4, u4_mask = self.decoder4(bottleneck, d4, bottleneck_mask, d4_mask)
        u3, u3_mask = self.decoder3(u4, d3, u4_mask, d3_mask)
        u2, u2_mask = self.decoder2(u3, d2, u3_mask, d2_mask)
        u1, u1_mask = self.decoder1(u2, d1, u2_mask, d1_mask)

        return {key: mapper(u1, u1_mask) for key, mapper in self.mappers.items()}
