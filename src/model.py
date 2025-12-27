import torch.nn as nn


class CoordinateCNN(nn.Module):
    """
    Lightweight CNN for coordinate regression.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 50 * 50, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
