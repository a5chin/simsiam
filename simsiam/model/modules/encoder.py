import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, dim: int = 2048) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),  # second layer
            nn.BatchNorm1d(dim, affine=False),
        )

    def forward(self, x) -> torch.Tensor:
        return self.layer(x)
