import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch

# preprocessing steps:
# -convert observation to pil image
# -convert image to grayscale
# -crop image to 84x84 to reduce input size


class Preprocessor:
    def __init__(self, size, to_grayscale=True, num_output_channels=1):
        self.size = size
        self.to_grayscale = to_grayscale
        self.num_output_channels = num_output_channels

        if to_grayscale:
            self.transformer = T.Compose([
                T.Grayscale(num_output_channels=self.num_output_channels),
                T.Resize(size=(size, size), interpolation=1),
            ])
        else:
            self.transformer = T.Compose([
                T.CenterCrop(self.size),
            ])

    @staticmethod
    def to_pil(observation):
        # convert observation array to pil image
        pixel_array = np.array(observation, dtype=np.uint8)
        image = Image.fromarray(pixel_array)
        return image

    def process(self, observation):
        # processes a pil image
        #tensor = self.transformer(self.to_pil(observation))
        state = np.array(observation)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)
