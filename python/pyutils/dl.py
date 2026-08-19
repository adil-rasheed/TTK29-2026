"""Deep-learning scaffolding for the architecture modules.

The rest of the course runs in ``float64`` on the CPU, full batch. That is the
right choice there: PINN residuals differentiate the network twice, and
second derivatives in ``float32`` lose most of their significant digits. It is
the wrong choice here. Convolutional stacks, attention and diffusion samplers
are several times slower in double precision, gain nothing from it, and need
mini-batching to train at all.

So this module owns the *other* set of conventions, and states them explicitly
rather than letting a notebook inherit whichever default was set last:

    from pyutils.dl import set_dl_defaults, get_device, train_supervised
    set_dl_defaults(seed=0)          # float32, seeded, deterministic

``get_device()`` returns the CPU unless you ask otherwise. Apple's MPS and CUDA
backends do not reproduce each other bit for bit, and every notebook in this
course ends in ``assert`` statements that quote numbers in the slides. A
reproducible three-minute CPU run is worth more here than a fast one nobody
else can reproduce; pass ``prefer_accelerator=True`` when you want the speed.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn

__all__ = [
    "set_dl_defaults",
    "get_device",
    "count_params",
    "train_supervised",
    "TrainHistory",
]


# --------------------------------------------------------------------------- #
#  conventions
# --------------------------------------------------------------------------- #
def set_dl_defaults(seed: int = 0, dtype: torch.dtype = torch.float32) -> None:
    """Pin dtype, seed every RNG, and ask torch for deterministic kernels."""
    torch.set_default_dtype(dtype)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(prefer_accelerator: bool = False) -> torch.device:
    """CPU by default; an accelerator only when explicitly requested."""
    if prefer_accelerator:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Parameter count -- the number these modules compare architectures by."""
    ps = model.parameters()
    return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)


# --------------------------------------------------------------------------- #
#  training
# --------------------------------------------------------------------------- #
class TrainHistory(dict):
    """``{'train': [...], 'val': [...], 'epochs': int, 'best_val': float}``."""

    @property
    def best_epoch(self) -> int:
        return int(np.argmin(self["val"])) if self["val"] else -1


def train_supervised(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_fraction: float = 0.2,
    loss_fn: Callable | None = None,
    patience: int | None = None,
    device: torch.device | None = None,
    shuffle_seed: int = 0,
    log_every: int = 0,
) -> TrainHistory:
    """Mini-batch training with a held-out split and optional early stopping.

    Deliberately small and readable: the architecture modules are about what
    the *model* does, so the loop stays something a reader can hold in their
    head. Returns per-epoch train and validation loss.

    The split is taken from the *end* of the array, so pass shuffled data for
    i.i.d. problems -- and do NOT shuffle for time series, where the last
    slice is exactly the right validation set (see the forecasting chapter).
    """
    device = device or get_device()
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    loss_fn = loss_fn or nn.MSELoss()

    n = x.shape[0]
    n_val = int(round(val_fraction * n))
    n_train = n - n_val
    xt, yt = x[:n_train], y[:n_train]
    xv, yv = x[n_train:], y[n_train:]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(shuffle_seed)
    hist = TrainHistory(train=[], val=[], epochs=0, best_val=math.inf)
    best_state, bad_epochs = None, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        running = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            running += loss.item() * idx.numel()
        tr = running / max(n_train, 1)

        model.eval()
        with torch.no_grad():
            va = loss_fn(model(xv), yv).item() if n_val else float("nan")

        hist["train"].append(tr)
        hist["val"].append(va)
        hist["epochs"] = epoch + 1
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  train {tr:.4e}  val {va:.4e}")

        if n_val and va < hist["best_val"] - 1e-12:
            hist["best_val"] = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                if log_every:
                    print(f"early stop at epoch {epoch} (patience {patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
