"""Nested random Fourier features using independent, local CPU generators."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .runs import Run


@dataclass(frozen=True)
class FeatureBank:
    frequencies: torch.Tensor
    phases: torch.Tensor
    seed: int
    phase_seed: int
    lengthscale: float

    def payload(self) -> dict[str, Any]:
        return {
            "frequencies": self.frequencies,
            "phases": self.phases,
            "seed": self.seed,
            "phase_seed": self.phase_seed,
            "lengthscale": self.lengthscale,
            "formula": "sqrt(2/p) * cos(x @ frequencies[:p].T + phases[:p])",
        }

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> FeatureBank:
        bank = cls(
            **{
                key: value[key]
                for key in (
                    "frequencies",
                    "phases",
                    "seed",
                    "phase_seed",
                    "lengthscale",
                )
            }
        )
        if bank.frequencies.ndim != 2 or bank.phases.shape != (len(bank.frequencies),):
            raise ValueError("Invalid feature-bank shapes.")
        for tensor in (bank.frequencies, bank.phases):
            if tensor.dtype != torch.float64 or tensor.device.type != "cpu":
                raise ValueError("Feature banks must contain CPU float64 tensors.")
            if not torch.isfinite(tensor).all():
                raise ValueError("Feature bank contains nonfinite values.")
        return bank


def generate_bank(config: ExperimentConfig, seed: int) -> FeatureBank:
    config.validate()
    if seed not in config.features.seeds:
        raise ValueError("Feature seed is not registered.")
    phase_seed = seed + config.features.phase_seed_offset
    generator_w = torch.Generator(device="cpu").manual_seed(seed)
    generator_b = torch.Generator(device="cpu").manual_seed(phase_seed)
    width = max(config.features.counts)
    frequencies = (
        torch.randn(
            (width, config.data.input_dim),
            generator=generator_w,
            dtype=torch.float64,
            device="cpu",
        )
        / config.features.lengthscale
    )
    phases = (
        2
        * pi
        * torch.rand(
            width,
            generator=generator_b,
            dtype=torch.float64,
            device="cpu",
        )
    )
    return FeatureBank(frequencies, phases, seed, phase_seed, config.features.lengthscale)


def get_bank(run: Run, seed: int) -> FeatureBank:
    if seed not in run.config.features.seeds:
        raise ValueError("Feature seed is not registered.")
    run.data_identity("development")
    path = run.path / "data" / "feature_banks" / f"seed_{seed}.pt"
    return FeatureBank.from_payload(torch.load(path, map_location="cpu", weights_only=True))


def make_features(x: torch.Tensor, bank: FeatureBank, count: int) -> torch.Tensor:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(bank.phases):
        raise ValueError("count must be a positive integer within the feature bank.")
    if x.ndim != 2 or x.shape[1] != bank.frequencies.shape[1]:
        raise ValueError("Input shape does not match the feature bank.")
    if x.device.type != "cpu" or x.dtype != torch.float64:
        raise ValueError("Inputs must use CPU float64; no implicit backend/precision conversion.")
    return torch.cos(x @ bank.frequencies[:count].T + bank.phases[:count]) * sqrt(2.0 / count)
