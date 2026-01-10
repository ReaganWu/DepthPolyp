import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as Aug
from albumentations.core.composition import OneOf

from skimage.draw import disk
from skimage.filters import gaussian


class AddLightSpots(Aug.ImageOnlyTransform):
    """
    Manually augmented light spots for endoscopic scenarios.
    This transform is integrated into Albumentations to enhance data diversity.
    """
    def __init__(self, radius_range=(5, 20), intensity=0.9, num_spots=1, p=0.5, always_apply=False):
        """
        :param radius_range: Range of light spot radius (min_radius, max_radius)
        :param intensity: Brightness of the light spot (higher means brighter, range [0, 1])
        :param num_spots: Number of light spots generated per image
        :param always_apply: Whether to always apply this transform
        :param p: Probability of applying this transform
        """
        super().__init__(p=p, always_apply=always_apply)
        self.radius_range = radius_range
        self.intensity = intensity
        self.num_spots = num_spots

    def apply(self, image, **params):
        """
        Add synthetic light spots to the input image.
        """
        h, w, c = image.shape
        light_layer = np.zeros((h, w), dtype=np.float32)  # Single-channel light intensity map

        for _ in range(self.num_spots):
            # Randomly select the center position and radius of the light spot
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            radius = np.random.randint(self.radius_range[0], self.radius_range[1])

            # Generate a circular light spot
            rr, cc = disk((center_y, center_x), radius, shape=(h, w))
            light_layer[rr, cc] += self.intensity

        # Blur the light layer to simulate realistic illumination
        light_layer = gaussian(light_layer, sigma=np.mean(self.radius_range) / 2)
        light_layer = np.clip(light_layer, 0, 1)  # Prevent overflow

        # Normalize image if necessary
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Overlay the light layer onto each channel
        for i in range(c):
            image[:, :, i] = np.clip(image[:, :, i] + light_layer, 0, 1)

        return (image * 255).astype(np.uint8)
