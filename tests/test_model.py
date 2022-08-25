import sys
from pathlib import Path

import torch
from torch import nn

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + "/../")

from simsiam.loss import NegativeCosineSimilarity
from simsiam.model import SimSiam
from simsiam.model.backbone import resnet18


def test_model():
    backbone = resnet18(num_classes=4, pretrained=True)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    simsiam = SimSiam(backbone=backbone, dim=feat_dim)
    assert simsiam is not None

    criterion = NegativeCosineSimilarity()

    images = torch.randn(4, 3, 512, 512)
    out0, out1 = simsiam(images[:2], images[2:])
    assert (out0[0].shape == out0[1].shape) and (
        out1[0].shape == out1[1].shape
    )

    loss = (criterion(*out0).mean() + criterion(*out1).mean()) / 2
    assert loss is not None
