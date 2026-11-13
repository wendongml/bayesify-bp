"""Export the four self-contained tutorial models for torchinfo and Netron."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from bayesify_torch.architectures import densenet40, resnet20
from bayesify_torch.inspect import write_architecture_catalog
from bayesify_torch.models import MomentLeNet, MomentMLP


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "assets" / "architectures",
    help="catalog directory (default: pytorch/assets/architectures)",
)
args = parser.parse_args()

torch.manual_seed(41)
catalog = {
    "mnist_mlp": (MomentMLP().float().eval(), torch.zeros(1, 1, 28, 28)),
    "mnist_lenet": (MomentLeNet().float().eval(), torch.zeros(1, 1, 28, 28)),
    "cifar_resnet20": (resnet20().float().eval(), torch.zeros(1, 3, 32, 32)),
    "cifar_densenet40": (densenet40().float().eval(), torch.zeros(1, 3, 32, 32)),
}
print(write_architecture_catalog(catalog, args.output))
