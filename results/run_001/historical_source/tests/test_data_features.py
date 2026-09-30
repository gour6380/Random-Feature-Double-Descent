"""Authored contract tests; implementation handoff does not execute these tests."""

from dataclasses import replace
from math import exp, sqrt

import numpy as np
import pytest
import torch

from src.config import DataConfig, ExecutionConfig, ExperimentConfig, FeatureConfig
from src.data import (
    Partition,
    assert_disjoint,
    generate_partition,
    partition_digest,
    teacher,
)
from src.features import FeatureBank, generate_bank, make_features


def small_config():
    return ExperimentConfig(
        data=DataConfig(train_size=7, validation_size=5, test_size=9),
        features=FeatureConfig(counts=(2, 4, 8), seeds=(2026, 2027)),
        execution=ExecutionConfig(preflight_widths=(2, 4, 8)),
    )


def test_locked_defaults_and_json_roundtrip():
    config = ExperimentConfig()
    config.validate()
    assert len(config.features.counts) * len(config.features.seeds) == 105
    assert 105 * len(config.data.noise_stds) * 2 == 420
    assert config.features.counts[9:12] == (255, 256, 257)
    assert config.solver.ridge_lambda * config.data.train_size == 0.0256
    assert config.execution.budget_seconds == 1800
    assert ExperimentConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "field,value", [("train_size", True), ("input_dim", 2), ("test_size", 1.5)]
)
def test_invalid_data_config(field, value):
    with pytest.raises(ValueError):
        replace(DataConfig(), **{field: value}).validate()


def test_preflight_must_cover_grid_and_independent_streams():
    with pytest.raises(ValueError, match="largest"):
        replace(
            ExperimentConfig(), execution=ExecutionConfig(preflight_widths=(64, 256))
        ).validate()
    with pytest.raises(ValueError, match="distinct"):
        replace(DataConfig(), noise_seeds=(1001, 2002, 2003)).validate()
    with pytest.raises(ValueError):
        replace(FeatureConfig(), lengthscale=float("nan")).validate()


def test_generation_is_deterministic_paired_and_rng_isolated():
    config = small_config()
    state = torch.random.get_rng_state().clone()
    train = generate_partition(config.data, "train")
    duplicate = generate_partition(config.data, "train")
    bank = generate_bank(config, 2026)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert train.x.dtype == torch.float64 and train.x.device.type == "cpu"
    assert partition_digest(train) == partition_digest(duplicate)
    torch.testing.assert_close(train.labels[:, 0], train.clean, rtol=0, atol=0)
    torch.testing.assert_close(train.labels[:, 1], train.clean + 0.3 * train.noise, rtol=0, atol=0)
    assert bank.phase_seed == bank.seed + 10000


def test_partitions_are_independent_and_overlap_is_rejected():
    config = small_config().data
    partitions = [generate_partition(config, name) for name in ("train", "validation", "test")]
    assert_disjoint(*partitions)
    assert [len(part) for part in partitions] == [7, 5, 9]
    train = partitions[0]
    copied = Partition(
        "fake",
        train.x,
        train.clean,
        train.noise,
        train.labels,
        tuple(f"fake:{i}" for i in range(len(train))),
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(train, copied)


def test_teacher_unit_population_variance_by_gaussian_quadrature():
    # Quadrature checks the analytic distribution, not a noisy generated estimate.
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    coordinates = torch.from_numpy(nodes * sqrt(2.0))
    points = torch.cartesian_prod(coordinates, coordinates, coordinates)
    probabilities = torch.from_numpy(weights / sqrt(np.pi))
    joint = torch.cartesian_prod(probabilities, probabilities, probabilities).prod(dim=1)
    values = teacher(points)
    assert abs(float((joint * values).sum())) < 1e-12
    assert float((joint * values.square()).sum()) == pytest.approx(1.0, abs=1e-12)
    unit = torch.tensor([[np.pi / 2, 0, 0]], dtype=torch.float64)
    expected = 1 / sqrt((1 - exp(-2)) / 2 * (1 + 0.5**2 + 0.25**2))
    assert teacher(unit).item() == pytest.approx(expected)


def test_nested_features_have_count_specific_scaling():
    config = small_config()
    x = generate_partition(config.data, "train").x
    bank = generate_bank(config, 2026)
    small = make_features(x, bank, 2)
    large = make_features(x, bank, 8)
    torch.testing.assert_close(small, large[:, :2] * sqrt(8 / 2), rtol=1e-14, atol=1e-14)
    again = generate_bank(config, 2026)
    torch.testing.assert_close(bank.frequencies, again.frequencies, rtol=0, atol=0)
    assert not torch.equal(bank.frequencies, generate_bank(config, 2027).frequencies)
    restored = FeatureBank.from_payload(bank.payload())
    torch.testing.assert_close(make_features(x, restored, 8), large, rtol=0, atol=0)


def test_feature_encoding_rejects_silent_precision_changes():
    config = small_config()
    bank = generate_bank(config, 2026)
    with pytest.raises(ValueError, match="float64"):
        make_features(torch.zeros(2, 8, dtype=torch.float32), bank, 4)
    with pytest.raises(ValueError):
        make_features(torch.zeros(2, 8, dtype=torch.float64), bank, 9)
