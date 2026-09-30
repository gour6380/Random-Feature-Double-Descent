"""Standalone Fourier-feature predictors saved as metadata plus tensor state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .features import FeatureBank, make_features
from .io import atomic_torch_save

if TYPE_CHECKING:
    from .runs import Run


@dataclass(frozen=True)
class Regressor:
    bank: FeatureBank
    count: int
    beta: torch.Tensor
    metadata: dict[str, Any]

    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if x.ndim != 2 or x.shape[1] != self.bank.frequencies.shape[1]:
            raise ValueError("Inputs must have shape [samples, input_dim].")
        if x.dtype != torch.float64 or x.device.type != "cpu":
            raise ValueError("Predictions require CPU float64 inputs.")
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite input to regressor.")
        result = torch.empty(len(x), dtype=torch.float64, device="cpu")
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            result[start:stop] = make_features(x[start:stop], self.bank, self.count) @ self.beta
        if not torch.isfinite(result).all():
            raise FloatingPointError("Regressor produced nonfinite predictions.")
        return result


def save_regressor(
    path: Path | str,
    bank: FeatureBank,
    count: int,
    beta: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    if beta.shape != (count,) or beta.dtype != torch.float64 or beta.device.type != "cpu":
        raise ValueError("Checkpoint coefficients must be a CPU float64 vector of length count.")
    if not torch.isfinite(beta).all() or not 1 <= count <= len(bank.phases):
        raise ValueError("Invalid predictor coefficients/feature count.")
    # Store only this predictor's prefix; no external feature-bank dependency remains.
    prefix = FeatureBank(
        bank.frequencies[:count].clone(),
        bank.phases[:count].clone(),
        bank.seed,
        bank.phase_seed,
        bank.lengthscale,
    )
    atomic_torch_save(
        Path(path),
        {
            "format_version": 1,
            "bank": prefix.payload(),
            "count": count,
            "beta": beta.detach().clone(),
            "metadata": metadata,
            "input_encoding": "raw independent normal coordinates; no normalization/intercept",
            "dtype": "float64",
            "backend": "cpu",
        },
    )


def load_regressor(
    path_or_run: Path | str | Run,
    *,
    seed: int | None = None,
    count: int | None = None,
    noise_std: float = 0.3,
    solver: str = "minimum_norm",
) -> Regressor:
    if isinstance(path_or_run, (str, Path)):
        path = Path(path_or_run)
    else:
        run = path_or_run
        run.verify_identity()
        if seed not in run.config.features.seeds or count not in run.config.features.counts:
            raise ValueError("Specify a registered seed and feature count.")
        if noise_std not in run.config.data.noise_stds or solver not in ("minimum_norm", "ridge"):
            raise ValueError("Unknown noise condition or primary solver.")
        from .fit import verified_status

        directory = run.path / "trials" / f"seed_{seed}" / f"features_{count:04d}"
        status = verified_status(run, seed, count)
        if status is None or status.get("status") != "complete":
            raise ValueError("Requested predictor has not completed successfully.")
        noise_index = run.config.data.noise_stds.index(noise_std)
        path = directory / "models" / f"{solver}_noise_{noise_index}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format_version") != 1
        or payload.get("dtype") != "float64"
        or payload.get("backend") != "cpu"
    ):
        raise ValueError("Unsupported predictor format/precision/backend.")
    bank = FeatureBank.from_payload(payload["bank"])
    beta = payload["beta"]
    if (
        beta.shape != (payload["count"],)
        or beta.dtype != torch.float64
        or not torch.isfinite(beta).all()
    ):
        raise ValueError("Invalid coefficient vector in predictor checkpoint.")
    if payload["count"] != len(bank.phases):
        raise ValueError("Predictor feature count and saved bank disagree.")
    return Regressor(bank, payload["count"], beta, payload["metadata"])
