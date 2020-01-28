import numpy as np
import torch
import torch.nn.functional as F
from abtem.learn.postprocess import non_maximum_suppresion
from abtem.learn.preprocess import pad_to_size, weighted_normalization
from abtem.learn.scale import find_hexagonal_sampling
from abtem.learn.unet import UNet


def build_unet_model(parameters, device):
    model = UNet(in_channels=parameters['in_channels'],
                 out_channels=parameters['out_channels'],
                 init_features=parameters['init_features'],
                 dropout=0.)
    model.load_state_dict(torch.load(parameters['weights_file'], map_location=device))
    return model


def build_model_from_dict(parameters):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mask_model = build_unet_model(parameters=parameters['mask_model'], device=device)
    density_model = build_unet_model(parameters=parameters['density_model'], device=device)

    if parameters['scale']['crystal_system'] == 'hexagonal':
        scale_model = lambda x: find_hexagonal_sampling(x, a=parameters['scale']['lattice_constant'],
                                                        min_sampling=parameters['scale']['min_sampling'])
    else:
        raise NotImplementedError('')

    def discretization_model(density):
        nms_distance_pixels = int(np.round(parameters['nms']['distance'] / parameters['training_sampling']))

        accepted = non_maximum_suppresion(density, distance=nms_distance_pixels,
                                          threshold=parameters['nms']['threshold'])

        points = np.array(np.where(accepted[0])).T
        # probabilities = probabilities[0, :, points[:, 0], points[:, 1]]
        return points

    model = AtomRecognitionModel(mask_model, density_model, training_sampling=parameters['training_sampling'],
                                 scale_model=scale_model, discretization_model=discretization_model)


class AtomRecognitionModel:

    def __init__(self, mask_model, density_model, training_sampling, scale_model, discretization_model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask_model = mask_model
        self.density_model = density_model
        self.training_sampling = training_sampling
        self.scale_model = scale_model
        self.discretization_model = discretization_model

    def standardize_dims(self, images):
        if len(images.shape) == 2:
            images = images.unsqueeze(0).unsqueeze(0)
        elif len(images.shape) == 3:
            images = torch.unsqueeze(images, 0)
        elif len(images.shape) != 4:
            raise RuntimeError('')
        return images

    def rescale_images(self, images, sampling):
        scale_factor = sampling / self.training_sampling
        images = F.interpolate(images, scale_factor=scale_factor, mode='nearest')
        images = pad_to_size(images, images.shape[2], images.shape[3], n=16)
        return images

    def normalize_images(self, images, mask=None):
        return weighted_normalization(images, mask)

    # def postprocess_images(self, image, original_shape, sampling):
    #     image = rescale(image, self.training_sampling / sampling, multichannel=False, anti_aliasing=False)
    #     shape = image.shape
    #     padding = (shape[0] - original_shape[0], shape[1] - original_shape[1])
    #     image = image[padding[0] // 2: padding[0] // 2 + original_shape[0],
    #             padding[1] // 2: padding[1] // 2 + original_shape[1]]
    #     return image

    def postprocess_points(self, points, shape, original_shape, sampling):
        shape = np.round(np.array(shape) * self.training_sampling / sampling)
        padding = (shape[0] - original_shape[0], shape[1] - original_shape[1])
        points = points * self.training_sampling / sampling
        return points - np.array([padding[0] // 2, padding[1] // 2])

    def forward(self, images):
        images = torch.tensor(images).to(self.device)
        images = self.standardize_dims(images)
        orig_shape = images.shape[-2:]
        sampling = self.scale_model(images)
        images = self.rescale_images(images, sampling)
        images = self.normalize_images(images)
        mask = self.mask_model(images)
        mask = torch.sum(mask[:, :-1], dim=1)[:, None]
        images = self.normalize_images(images, mask)
        density = self.density_model(images)
        density = mask * density
        density = density.detach().cpu().numpy()
        points = self.discretization_model(density)
        points = [self.postprocess_points(p, density.shape[-2:], orig_shape, sampling) for p in points]
        return points
