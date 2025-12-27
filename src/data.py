import numpy as np
import torch
from torch.utils.data import Dataset


class PixelDataset(Dataset):
    """
    Dataset generating 50x50 images with one active pixel.
    """

    def __init__(self, num_samples: int = 10000):
        self.num_samples = num_samples
        self.image_size = 50

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = np.zeros((self.image_size, self.image_size),
                         dtype=np.float32)

        x = np.random.randint(0, self.image_size)
        y = np.random.randint(0, self.image_size)

        image[y, x] = 255.0

        # Normalize
        image /= 255.0
        target = np.array([x / 49.0, y / 49.0], dtype=np.float32)

        return torch.tensor(image).unsqueeze(0), torch.tensor(target)
