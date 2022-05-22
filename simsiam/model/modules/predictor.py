import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, dim: int = 2048, pred_dim: int = 512) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),  # output layer
        )

    def forward(self, x) -> torch.Tensor:
        return self.layer(x)
