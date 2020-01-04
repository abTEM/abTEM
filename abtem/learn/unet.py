from collections import OrderedDict
from abtem.utils import BatchGenerator
import numpy as np
import torch
import torch.nn as nn
from abtem.learn.partialconv2d import PartialConv2d

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
        #x = self.conv(x)
        return self.activation(x)


class UNet(nn.Module):

    def __init__(self, heads, in_channels=1, init_features=16, dropout=.5):
        super().__init__()

        features = init_features

        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop1 = nn.Dropout(p=dropout)

        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop2 = nn.Dropout(p=dropout)

        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop3 = nn.Dropout(p=dropout)

        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downdrop4 = nn.Dropout(p=dropout)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        #self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.updrop4 = nn.Dropout(p=dropout)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.updrop3 = nn.Dropout(p=dropout)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.updrop2 = nn.Dropout(p=dropout)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.updrop1 = nn.Dropout(p=dropout)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.heads = heads

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

        return {key: head(dec1) for key, head in self.heads.items()}

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

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        PartialConv2d(
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
                        PartialConv2d(
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
