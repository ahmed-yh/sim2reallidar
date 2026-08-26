"""Shared train/validate loop + early stopping.

Written fresh for this repo, not copied from the existing CNN baseline
elsewhere in this project (per earlier project decision: that codebase is
not used as a design reference here) -- same standard early-stopping idea,
independent implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses import chamfer_distance, segmentation_loss


def compute_losses(model: nn.Module, xyz: torch.Tensor, cls: torch.Tensor,
                    class_weights: torch.Tensor, lambda_class: float,
                    chamfer_target_subsample: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass -> (total_loss, recon_loss, class_loss), the latter
    two kept separate so training curves can be watched independently (a
    class loss that stalls while recon keeps improving, or vice versa, is a
    real diagnostic signal, not something to bury in a single combined
    number)."""
    out = model(xyz)
    recon_loss = chamfer_distance(out["recon_points"], xyz, target_subsample=chamfer_target_subsample)
    class_loss = segmentation_loss(out["class_logits"], cls, class_weights)
    total = recon_loss + lambda_class * class_loss
    return total, recon_loss, class_loss


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                     class_weights: torch.Tensor, lambda_class: float,
                     chamfer_target_subsample: int | None, device: torch.device) -> tuple[float, float]:
    model.train()
    recon_total, class_total, n = 0.0, 0.0, 0
    for xyz, cls in loader:
        xyz, cls = xyz.to(device, non_blocking=True), cls.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        total, recon_loss, class_loss = compute_losses(
            model, xyz, cls, class_weights, lambda_class, chamfer_target_subsample
        )
        total.backward()
        optimizer.step()
        bs = xyz.size(0)
        recon_total += recon_loss.item() * bs
        class_total += class_loss.item() * bs
        n += bs
    return recon_total / max(n, 1), class_total / max(n, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, class_weights: torch.Tensor,
             lambda_class: float, chamfer_target_subsample: int | None,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    recon_total, class_total, n = 0.0, 0.0, 0
    for xyz, cls in loader:
        xyz, cls = xyz.to(device, non_blocking=True), cls.to(device, non_blocking=True)
        _, recon_loss, class_loss = compute_losses(
            model, xyz, cls, class_weights, lambda_class, chamfer_target_subsample
        )
        bs = xyz.size(0)
        recon_total += recon_loss.item() * bs
        class_total += class_loss.item() * bs
        n += bs
    return recon_total / max(n, 1), class_total / max(n, 1)


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 1e-4
    best_val_loss: float = math.inf
    counter: int = 0
    best_state: dict | None = field(default=None, repr=False)

    def step(self, val_loss: float, model: nn.Module | None = None) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        optimizer: torch.optim.Optimizer, class_weights: torch.Tensor, lambda_class: float,
        chamfer_target_subsample: int | None, device: torch.device, max_epochs: int,
        patience: int, log_fn: Callable[[int, float, float, float, float], None] | None = None
        ) -> EarlyStopping:
    es = EarlyStopping(patience=patience)
    for epoch in range(1, max_epochs + 1):
        train_recon, train_class = train_one_epoch(
            model, train_loader, optimizer, class_weights, lambda_class, chamfer_target_subsample, device
        )
        val_recon, val_class = validate(
            model, val_loader, class_weights, lambda_class, chamfer_target_subsample, device
        )
        val_total = val_recon + lambda_class * val_class
        if log_fn is not None:
            log_fn(epoch, train_recon, train_class, val_recon, val_class)
        if es.step(val_total, model):
            break
    es.restore_best(model)
    return es
