import e2cnn.nn as e2nn
import torch.nn as nn
from e2cnn import gspaces


class R2DoubleConv(nn.Module):

    def __init__(self, in_type, out_type, batch_norm=True):
        super().__init__()

        layers = []

        layers += [e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=not batch_norm)]
        if batch_norm:
            layers += [e2nn.InnerBatchNorm(out_type)]
        layers += [e2nn.ReLU(out_type, inplace=True)]

        layers += [e2nn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=not batch_norm)]
        if batch_norm:
            layers += [e2nn.InnerBatchNorm(out_type)]
        layers += [e2nn.ReLU(out_type, inplace=True)]

        self.double_conv = e2nn.SequentialModule(*layers)

    @property
    def out_type(self):
        return self.double_conv.out_type

    def forward(self, x):
        return self.double_conv(x)


class R2Down(nn.Module):

    def __init__(self, in_type, out_type):
        super().__init__()
        self.pool = e2nn.PointwiseMaxPoolAntialiased(in_type, 2)
        self.conv = R2DoubleConv(self.pool.out_type, out_type)

    @property
    def out_type(self):
        return self.conv.out_type

    def forward(self, x):
        return self.conv(self.pool(x))


class R2Up(nn.Module):

    def __init__(self, in_type, direct_sum_type, out_type):
        super().__init__()

        self.up = e2nn.R2Upsampling(in_type, scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = R2DoubleConv(direct_sum_type, out_type, batch_norm=False)

    @property
    def out_type(self):
        return self.conv.out_type

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = e2nn.tensor_directsum((x1, x2))
        return self.conv(x)


class ConvHead(nn.Module):

    def __init__(self, in_type, out_channels):
        super().__init__()

        self.conv = R2DoubleConv(in_type, in_type)
        self.gpool = e2nn.GroupPooling(in_type)
        self.out = nn.Conv2d(len(in_type), out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.gpool(x)
        x = x.tensor
        return self.out(x)


class R2UNet(nn.Module):

    def __init__(self, in_channels, features, N=8):
        super().__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=N)

        self.in_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        out_type = e2nn.FieldType(self.r2_act, features * [self.r2_act.regular_repr])
        self.inc = R2DoubleConv(self.in_type, out_type)

        out_type = e2nn.FieldType(self.r2_act, 2 * features * [self.r2_act.regular_repr])
        self.down1 = R2Down(self.inc.out_type, out_type)

        out_type = e2nn.FieldType(self.r2_act, 4 * features * [self.r2_act.regular_repr])
        self.down2 = R2Down(self.down1.out_type, out_type)

        out_type = e2nn.FieldType(self.r2_act, 4 * features * [self.r2_act.regular_repr])
        self.down3 = R2Down(self.down2.out_type, out_type)
        #
        # out_type = e2nn.FieldType(self.r2_act, 8 * features * [self.r2_act.regular_repr])
        # self.down4 = R2Down(self.down3.out_type, out_type)
        #
        # out_type = e2nn.FieldType(self.r2_act, 4 * features * [self.r2_act.regular_repr])
        # direct_sum_type = self.down3.out_type + self.down4.out_type
        # self.up1 = R2Up(self.down4.out_type, direct_sum_type, out_type)

        out_type = e2nn.FieldType(self.r2_act, 2 * features * [self.r2_act.regular_repr])
        direct_sum_type = self.down2.out_type + self.down3.out_type
        self.up1 = R2Up(self.down2.out_type, direct_sum_type, out_type)

        out_type = e2nn.FieldType(self.r2_act, 1 * features * [self.r2_act.regular_repr])
        direct_sum_type = self.down1.out_type + self.up1.out_type
        self.up2 = R2Up(self.down1.out_type, direct_sum_type, out_type)

        out_type = e2nn.FieldType(self.r2_act, 1 * features * [self.r2_act.regular_repr])
        direct_sum_type = self.inc.out_type + self.up2.out_type
        self.up3 = R2Up(self.inc.out_type, direct_sum_type, out_type)

        # self.gpool = e2nn.GroupPooling(out_type)

    @property
    def out_type(self):
        return self.up3.out_type

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.in_type)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.gpool(x)

        # x = x.tensor
        return x
