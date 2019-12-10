from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np


class Head(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError()


def DensityMap():
    def build_density_map(features):
        return DensityMapModule(features)

    return build_density_map


class DensityMapModule(Head):

    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1)

    @property
    def out_channels(self):
        return 1

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


def ClassificationMap(nclasses):
    def build(features):
        return ClassificationMapModule(features, nclasses=nclasses)

    return build


class ClassificationMapModule(Head):

    def __init__(self, features, nclasses):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=features, out_channels=nclasses, kernel_size=1)
        self._nclasses = nclasses

    @property
    def out_channels(self):
        return self._nclasses

    def forward(self, x):
        return nn.Softmax2d()(self.conv(x))


class UNet(nn.Module):

    def __init__(self, mappers, in_channels=1, init_features=16, p=.3):
        super(UNet, self).__init__()

        features = init_features

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop1 = nn.Dropout(p=p)

        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop2 = nn.Dropout(p=p)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop3 = nn.Dropout(p=p)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop4 = nn.Dropout(p=p)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.updrop4 = nn.Dropout(p=p)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.updrop3 = nn.Dropout(p=p)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.updrop2 = nn.Dropout(p=p)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.updrop1 = nn.Dropout(p=p)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.mappers = [mapper(features) for mapper in mappers]

        for i, mapper in enumerate(self.mappers):
            setattr(self, 'mapper' + str(i), mapper)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.downdrop1(self.pool1(enc1)))
        enc3 = self.encoder3(self.downdrop2(self.pool2(enc2)))
        enc4 = self.encoder4(self.downdrop3(self.pool3(enc3)))

        bottleneck = self.bottleneck(self.downdrop4(self.pool4(enc4)))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.updrop4(dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.updrop3(dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.updrop2(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.updrop1(dec1)
        dec1 = self.decoder1(dec1)

        return [mapper(dec1) for mapper in self.mappers]

    def mc_predict(self, images, n):

        mc_outputs = [np.zeros((n,) + (images.shape[0],) + (mapper.out_channels,) + images.shape[2:]) for mapper in
                      self.mappers]

        for i in range(n):
            outputs = self.forward(images)
            for j in range(len(mc_outputs)):
                mc_outputs[j][i] = outputs[j].cpu().detach().numpy()

        return [(np.mean(mc_output, 0), np.std(mc_output, 0)) for mc_output in mc_outputs]

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
