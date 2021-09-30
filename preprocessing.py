import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch

# preprocessing steps:
# -convert observation to pil image
# -convert image to grayscale
# -crop image to 84x84 to reduce input size


class Preprocessor:
    def __init__(self, size, device, to_grayscale=True):
        self.size = size
        self.to_grayscale = to_grayscale
        self.device = device

        # transformer function for pil images
        if to_grayscale:
            self.transformer = T.Compose([
                T.Grayscale(),
                T.Resize(size=(size, size)),
                # T.ToTensor()
            ])
        else:
            self.transformer = T.Compose([
                T.Resize(size, size),
                T.ToTensor()
            ])

    @staticmethod
    def to_pil(observation):
        # convert observation array to pil image
        pixel_array = np.array(observation, dtype=np.uint8)
        image = Image.fromarray(pixel_array)
        return image

    def process(self, observation):
        # processes a pil image
        # Convert to float, rescale, convert to tensor
        observation = observation.transpose(2, 0, 1)
        screen = np.ascontiguousarray(observation, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # screen cropping
        bbox = [34, 0, 160, 160]
        screen = screen[:, bbox[0]:bbox[2] + bbox[0], bbox[1]:bbox[3] + bbox[1]]

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Grayscale()
            , T.Resize((84, 84))
            , T.ToTensor()
        ])

        screen = self.transformer(screen)
        screen = screen.unsqueeze(0)
        return screen.to(self.device)
