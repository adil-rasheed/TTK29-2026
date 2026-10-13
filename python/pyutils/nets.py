"""Small, shared network definitions used across the course notebooks.

    from pyutils.nets import MLP

`MLP` is a tanh multilayer perceptron with a linear output layer and
Glorot/Xavier initialisation. It is deliberately smooth (no ReLU) so that the
same class serves every module, including those that differentiate the network
with respect to its inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MLP"]


class MLP(nn.Module):
    """Tanh MLP. `layers` e.g. [1, 32, 32, 1] = 1 input, two hidden, 1 output."""

    def __init__(self, layers: list[int]):
        super().__init__()
        mods: list[nn.Module] = []
        for d_in, d_out in zip(layers[:-1], layers[1:]):
            mods.append(nn.Linear(d_in, d_out))
            mods.append(nn.Tanh())
        mods.pop()  # drop the final activation -> linear output
        self.net = nn.Sequential(*mods)
        # Glorot/Xavier init, matching the original code.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
