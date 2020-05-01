import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()

        layers = []

        if batch_norm:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            # layers += [PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(out_channels)]
        else:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]
            # layers += [PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]

        layers += [nn.ReLU(inplace=True)]

        if batch_norm:
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            # layers += [PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)]
            layers += [nn.BatchNorm2d(out_channels)]
        else:
            layers += [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)]
            # layers += [PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)]

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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels, features=16, dropout=.0, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, 2 * features, dropout=dropout)
        self.down2 = Down(2 * features, 4 * features, dropout=dropout)
        self.down3 = Down(4 * features, 8 * features, dropout=dropout)
        self.down4 = Down(8 * features, 8 * features, dropout=dropout)
        self.up1 = Up(16 * features, 4 * features, bilinear, dropout=dropout)
        self.up2 = Up(8 * features, 2 * features, bilinear, dropout=dropout)
        self.up3 = Up(4 * features, features, bilinear, dropout=dropout)
        self.up4 = Up(2 * features, features, bilinear, dropout=dropout)
        self.out = OutConv(features, 1)

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
        return self.out(x)


class ConvHead(nn.Module):

    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.activation = activation

        layers = []
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.activation is None:
            return self.layers(x)
        else:
            return self.activation(self.layers(x))
