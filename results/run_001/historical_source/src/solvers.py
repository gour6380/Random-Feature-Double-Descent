"""SVD-based minimum-norm and ridge regression, sharing each decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from .config import SolverConfig


@dataclass(frozen=True)
class Decomposition:
    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor


def _require_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or min(value.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix.")
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError(f"{name} must use CPU float64.")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite values in {name}.")


def decompose(phi: torch.Tensor) -> Decomposition:
    _require_matrix(phi, "design matrix")
    u, s, vh = torch.linalg.svd(phi, full_matrices=False)
    if not all(torch.isfinite(value).all() for value in (u, s, vh)):
        raise FloatingPointError("The SVD returned nonfinite values.")
    return Decomposition(u, s, vh)


def minimum_norm_filter(s: torch.Tensor, rcond: float) -> torch.Tensor:
    """Strict cutoff; masked indexing avoids taking the reciprocal of zero."""
    retained = s > rcond * s[0]
    result = torch.zeros_like(s)
    result[retained] = s[retained].reciprocal()
    return result


def solve_coefficients(
    decomposition: Decomposition,
    y: torch.Tensor,
    n_samples: int,
    config: SolverConfig,
) -> dict[str, torch.Tensor]:
    config.validate()
    _require_matrix(y, "target matrix")
    if n_samples != len(y) or len(y) != decomposition.u.shape[0]:
        raise ValueError("Training sample count does not match the SVD and targets.")
    projected = decomposition.u.T @ y
    s = decomposition.s
    filters = {
        "minimum_norm": minimum_norm_filter(s, config.rcond),
        "ridge": s / (s.square() + n_samples * config.ridge_lambda),
        **{
            f"cutoff_{cutoff:g}": minimum_norm_filter(s, cutoff)
            for cutoff in config.sensitivity_rconds
        },
    }
    coefficients = {
        name: decomposition.vh.T @ (filter_[:, None] * projected)
        for name, filter_ in filters.items()
    }
    if not all(torch.isfinite(value).all() for value in coefficients.values()):
        raise FloatingPointError("Regression coefficients are nonfinite.")
    return coefficients


def numerical_summary(
    phi: torch.Tensor,
    y: torch.Tensor,
    beta: torch.Tensor,
    decomposition: Decomposition,
    config: SolverConfig,
    *,
    rcond: float | None = None,
) -> list[dict[str, float | int | bool]]:
    if y.ndim == 1:
        y = y[:, None]
    if beta.ndim == 1:
        beta = beta[:, None]
    if phi.shape[0] != y.shape[0] or phi.shape[1] != beta.shape[0] or y.shape[1] != beta.shape[1]:
        raise ValueError("Design, target and coefficient dimensions disagree.")
    s = decomposition.s
    cutoff = config.rcond if rcond is None else rcond
    rank = int((s > cutoff * s[0]).sum().item())
    # Full rectangular matrix condition number; do not replace tiny singular values.
    condition_number = float((s[0] / s[-1]).item()) if s[-1] > 0 else float("inf")
    residual = phi @ beta - y
    residual_norms = torch.linalg.vector_norm(residual, dim=0)
    relative = residual_norms / torch.linalg.vector_norm(y, dim=0).clamp_min(1.0)
    coefficient_norms = torch.linalg.vector_norm(beta, dim=0)
    return [
        {
            "numerical_rank": rank,
            "condition_number": condition_number,
            "coefficient_norm": float(coefficient_norms[column].item()),
            "residual_l2": float(residual_norms[column].item()),
            "relative_residual": float(relative[column].item()),
            "interpolates": bool(relative[column] <= config.interpolation_tolerance),
            "singular_max": float(s[0].item()),
            "singular_min": float(s[-1].item()),
        }
        for column in range(y.shape[1])
    ]


def metrics(
    pred: torch.Tensor, clean: torch.Tensor, observed: torch.Tensor
) -> dict[str, float | int]:
    """Sample-weighted metrics for one predictor, including incomplete chunks."""
    if pred.ndim != 1 or clean.shape != pred.shape or observed.shape != pred.shape or not len(pred):
        raise ValueError("Metrics require equally sized, nonempty vectors.")
    if not all(torch.isfinite(value).all() for value in (pred, clean, observed)):
        raise FloatingPointError("Cannot measure nonfinite predictions or targets.")
    signal_sse = float((pred - clean).square().sum().item())
    observed_sse = float((pred - observed).square().sum().item())
    if not all(isfinite(value) for value in (signal_sse, observed_sse)):
        raise FloatingPointError("Squared errors overflowed.")
    return {
        "signal_mse": signal_sse / len(pred),
        "observed_mse": observed_sse / len(pred),
        "signal_sse": signal_sse,
        "observed_sse": observed_sse,
        "count": len(pred),
    }
