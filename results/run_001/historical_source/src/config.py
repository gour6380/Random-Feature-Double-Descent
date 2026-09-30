"""Editable, validated protocol settings. Importing does not execute an experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite, sqrt
from numbers import Real
from typing import Any


def _integer(value: object, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _number(value: object, name: str, *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number >= {minimum}.")
    if not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}.")


def _seeds(values: tuple[int, ...], name: str) -> None:
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique.")
    for value in values:
        _integer(value, name, minimum=0)
        if value >= 2**63:
            raise ValueError(f"{name} must be below 2**63.")


@dataclass(frozen=True)
class DataConfig:
    input_dim: int = 8
    train_size: int = 256
    validation_size: int = 1_024
    test_size: int = 8_192
    input_seeds: tuple[int, ...] = (1001, 1002, 1003)
    noise_seeds: tuple[int, ...] = (2001, 2002, 2003)
    noise_stds: tuple[float, ...] = (0.0, 0.3)

    def validate(self) -> None:
        _integer(self.input_dim, "input_dim", minimum=3)
        for name in ("train_size", "validation_size", "test_size"):
            _integer(getattr(self, name), name)
        _seeds(self.input_seeds, "input_seeds")
        _seeds(self.noise_seeds, "noise_seeds")
        if len(self.input_seeds) != 3 or len(self.noise_seeds) != 3:
            raise ValueError("Exactly three seeds are required for train/validation/test.")
        if set(self.input_seeds) & set(self.noise_seeds):
            raise ValueError("Input and noise streams require distinct seeds.")
        if not self.noise_stds or len(set(self.noise_stds)) != len(self.noise_stds):
            raise ValueError("noise_stds must be nonempty and unique.")
        for sigma in self.noise_stds:
            _number(sigma, "noise standard deviation")


@dataclass(frozen=True)
class FeatureConfig:
    counts: tuple[int, ...] = (
        16,
        32,
        64,
        96,
        128,
        192,
        224,
        240,
        248,
        255,
        256,
        257,
        264,
        272,
        288,
        320,
        384,
        512,
        768,
        1024,
        2048,
    )
    seeds: tuple[int, ...] = (2026, 2027, 2028, 2029, 2030)
    phase_seed_offset: int = 10_000
    lengthscale: float = sqrt(8.0)

    def validate(self) -> None:
        if not self.counts or tuple(sorted(set(self.counts))) != self.counts:
            raise ValueError("Feature counts must be nonempty, increasing and unique.")
        for count in self.counts:
            _integer(count, "feature count")
        _seeds(self.seeds, "feature seeds")
        _integer(self.phase_seed_offset, "phase_seed_offset")
        phase_seeds = tuple(seed + self.phase_seed_offset for seed in self.seeds)
        _seeds(phase_seeds, "phase seeds")
        if set(phase_seeds) & set(self.seeds):
            raise ValueError("Frequency and phase streams must have distinct seeds.")
        _number(self.lengthscale, "lengthscale")
        if self.lengthscale == 0:
            raise ValueError("lengthscale must be positive.")


@dataclass(frozen=True)
class SolverConfig:
    rcond: float = 1e-12
    ridge_lambda: float = 1e-4
    interpolation_tolerance: float = 1e-8
    sensitivity_rconds: tuple[float, ...] = (1e-10, 1e-14)

    def validate(self) -> None:
        for name in ("rcond", "ridge_lambda", "interpolation_tolerance"):
            _number(getattr(self, name), name)
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.rcond >= 1:
            raise ValueError("rcond must be below one.")
        if len(set(self.sensitivity_rconds)) != len(self.sensitivity_rconds):
            raise ValueError("Sensitivity cutoffs must be unique.")
        for value in self.sensitivity_rconds:
            _number(value, "sensitivity cutoff")
            if not 0 < value < 1 or value == self.rcond:
                raise ValueError("Sensitivity cutoffs must be in (0,1), excluding rcond.")


@dataclass(frozen=True)
class ExecutionConfig:
    threads: int = 4
    eval_batch_size: int = 1_024
    budget_seconds: float = 1_800.0
    preflight_widths: tuple[int, ...] = (64, 256, 2048)

    def validate(self) -> None:
        _integer(self.threads, "threads")
        _integer(self.eval_batch_size, "eval_batch_size")
        _number(self.budget_seconds, "budget_seconds")
        if self.budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive.")
        if not self.preflight_widths or len(set(self.preflight_widths)) != len(
            self.preflight_widths
        ):
            raise ValueError("preflight_widths must be nonempty and unique.")
        for count in self.preflight_widths:
            _integer(count, "preflight width")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    protocol_version: str = "1.0"

    def validate(self) -> None:
        self.data.validate()
        self.features.validate()
        self.solver.validate()
        self.execution.validate()
        if not set(self.execution.preflight_widths).issubset(self.features.counts):
            raise ValueError("Preflight widths must belong to the registered feature grid.")
        if max(self.execution.preflight_widths) != max(self.features.counts):
            raise ValueError("Preflight must measure the largest registered feature count.")
        feature_streams = set(self.features.seeds) | {
            seed + self.features.phase_seed_offset for seed in self.features.seeds
        }
        data_streams = set(self.data.input_seeds) | set(self.data.noise_seeds)
        if feature_streams & data_streams:
            raise ValueError("Feature and data generation require independent seed streams.")
        if not isinstance(self.protocol_version, str) or not self.protocol_version:
            raise ValueError("protocol_version must be a nonempty string.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentConfig:
        payload = dict(value)
        for section, constructor, tuple_keys in (
            ("data", DataConfig, ("input_seeds", "noise_seeds", "noise_stds")),
            ("features", FeatureConfig, ("counts", "seeds")),
            ("solver", SolverConfig, ("sensitivity_rconds",)),
            ("execution", ExecutionConfig, ("preflight_widths",)),
        ):
            arguments = dict(payload.get(section, {}))
            for key in tuple_keys:
                if key in arguments:
                    arguments[key] = tuple(arguments[key])
            payload[section] = constructor(**arguments)
        config = cls(**payload)
        config.validate()
        return config
