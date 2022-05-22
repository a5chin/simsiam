import torch
from torch import nn

from .backbone import Darknet
from .modules import Encoder, Predictor

darknet = Darknet(depth=53)


class SimSiam(nn.Module):
    """Implementation of SimSiam[0] network
    Recommended loss: :py:class:`lightly.loss.sym_neg_cos_sim_loss.SymNegCosineSimilarityLoss`
    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566
    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        pred_hidden_dim:
            Dimension of the hidden layer of the predicion head. This should
            be `num_ftrs` / 4.
        out_dim:
            Dimension of the output (after the projection head).
    """

    def __init__(
        self,
        backbone: nn.Module = darknet,
        dim: int = 2048,
        pred_dim: int = 512,
    ) -> None:

        super().__init__()

        self.dim = dim
        self.pred_dim = pred_dim

        self.backbone = backbone
        # for parm in self.backbone.parameters():
        #     parm.requires_grad = False

        self.encoder = Encoder(dim=dim)
        self.predictor = Predictor(dim=dim, pred_dim=pred_dim)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor = None, return_features: bool = False):
        """Forward pass through SimSiam.
        Extracts features with the backbone and applies the projection
        head and prediction head to the output space. If both x0 and x1 are not
        None, both will be passed through the backbone, projection, and
        prediction head. If x1 is None, only x0 will be forwarded.
        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.
            return_features:
                Whether or not to return the intermediate features backbone(x).
        Returns:
            The output prediction and projection of x0 and (if x1 is not None)
            the output prediction and projection of x1. If return_features is
            True, the output for each x is a tuple (out, f) where f are the
            features before the projection head.

        Examples:
            >>> # single input, single output
            >>> out = model(x)
            >>>
            >>> # single input with return_features=True
            >>> out, f = model(x, return_features=True)
            >>>
            >>> # two inputs, two outputs
            >>> out0, out1 = model(x0, x1)
            >>>
            >>> # two inputs, two outputs with return_features=True
            >>> (out0, f0), (out1, f1) = model(x0, x1, return_features=True)
        """
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.encoder(f0)
        p0 = self.predictor(z0)

        out0 = (z0, p0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0

        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.encoder(f1)
        p1 = self.predictor(z1)

        out1 = (z1, p1)

        # append features if requested
        if return_features:
            out1 = (out1, f1)

        return out0, out1