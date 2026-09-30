"""Authored aggregation tests; never executed as part of implementation preparation."""

from math import sqrt

import pandas as pd
import pytest

from src.analysis import _summarize, _unique_frame


def test_failed_conditions_remain_in_counts_and_do_not_enter_means():
    frame = pd.DataFrame(
        [
            {"n_features": 16, "status": "complete", "mse": 1.0},
            {"n_features": 16, "status": "complete", "mse": 3.0},
            {"n_features": 16, "status": "failed_numerical", "mse": 999.0},
        ]
    )
    result = _summarize(frame, ["n_features"], ["mse"]).iloc[0]
    assert result.registered_count == 3
    assert result.complete_count == 2
    assert result.failed_count == 1
    assert result.mse_count == 2
    assert result.mse_mean == 2.0
    assert result.mse_std == pytest.approx(sqrt(2.0))


def test_one_seed_has_no_sample_standard_deviation():
    frame = pd.DataFrame([{"n_features": 16, "status": "complete", "mse": 0.0}])
    result = _summarize(frame, ["n_features"], ["mse"]).iloc[0]
    assert result.mse_mean == 0.0
    assert pd.isna(result.mse_std)
    assert result.mse_count == 1


def test_all_failed_group_is_retained_without_a_fake_zero():
    frame = pd.DataFrame([{"n_features": 16, "status": "failed_numerical", "mse": None}])
    result = _summarize(frame, ["n_features"], ["mse"]).iloc[0]
    assert result.registered_count == result.failed_count == 1
    assert result.mse_count == 0
    assert pd.isna(result.mse_mean)


def test_duplicate_measurement_identities_are_rejected():
    record = dict(seed=2026, n_features=16, noise_std=0.3, solver="minimum_norm")
    with pytest.raises(ValueError, match="Duplicate"):
        _unique_frame([record, record.copy()], "measurements")
