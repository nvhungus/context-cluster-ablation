from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .layers import CoordinateAugment


class CoordInput(nn.Module):
    """Prepend normalised xy channels to a backbone whose first conv takes 5 channels.

    The HBCC family appends two coordinate channels before its stem, so a headline
    comparison against a three-channel torchvision baseline confounds the coordinate
    channels with the architecture. This wrapper gives a baseline the same input, so
    the comparison can be made with that difference removed.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.coord = CoordinateAugment(enabled=True)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.coord(x))


def _widen_first_conv(conv: nn.Conv2d, in_chans: int = 5) -> nn.Conv2d:
    """Same convolution, more input channels. Weights are re-initialised by nn.Conv2d."""
    return nn.Conv2d(
        in_chans,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )


def resnet18_cifar(num_classes: int = 10, **_: object) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def mobilenet_v2_cifar(num_classes: int = 10, **_: object) -> nn.Module:
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    model.features[0][0].stride = (1, 1)
    return model


def shufflenet_v2_x1_0_cifar(num_classes: int = 10, **_: object) -> nn.Module:
    model = models.shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
    model.conv1[0].stride = (1, 1)
    model.maxpool = nn.Identity()
    return model


def shufflenet_v2_x1_5_cifar(num_classes: int = 10, use_coord: bool = False, **_: object) -> nn.Module:
    """Wider ShuffleNetV2 (~1.5x channels), used as a compute-matched CNN reference.

    ShuffleNetV2-x1.0 at 32x32 costs ~46M MACs, well below the ~139M of the HBCC-family
    models, so a like-for-like comparison needs a CNN at a similar compute budget.

    ``use_coord`` gives this baseline the two coordinate channels the HBCC family
    receives, widening the first convolution from three input channels to five. It
    defaults to False, so every previously reported run is unaffected: that path
    returns before any of the extra machinery is touched. Turning it on costs
    5*24*9 - 3*24*9 = 432 parameters, taking the 200-class model from 2,683,624 to
    2,684,056 and leaving the parameter match against the proposed design intact.
    """

    model = models.shufflenet_v2_x1_5(weights=None, num_classes=num_classes)
    model.conv1[0].stride = (1, 1)
    model.maxpool = nn.Identity()
    if not use_coord:
        return model
    model.conv1[0] = _widen_first_conv(model.conv1[0], in_chans=5)
    return CoordInput(model)
