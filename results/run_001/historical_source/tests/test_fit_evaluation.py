"""Authored CPU integration checks; not executed during implementation handoff."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src import fit
from src.checkpoints import load_regressor
from src.config import DataConfig, ExecutionConfig, ExperimentConfig, FeatureConfig
from src.data import prepare_data, prepare_test_data
from src.evaluate import (
    _evaluate_unit,
    evaluate_checkpoints,
    freeze_evaluation,
    verify_frozen,
)
from src.io import read_json, sha256_file
from src.runs import BudgetExceeded, create_run


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    """Real persistence with tiny scientific dimensions and no recursive test gate."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "fixture.py").write_text("# fixture provenance\n")
    (tmp_path / "requirements.txt").write_text("# fixture dependency identity\n")
    config = ExperimentConfig(
        data=DataConfig(train_size=8, validation_size=9, test_size=11),
        features=FeatureConfig(counts=(2, 4, 8), seeds=(2026, 2027)),
        execution=ExecutionConfig(
            threads=1, eval_batch_size=4, budget_seconds=100_000, preflight_widths=(2, 4, 8)
        ),
    )
    monkeypatch.setattr("src.runtime.require_runtime_validation", lambda run: None)
    monkeypatch.setattr("src.tracking.refresh_tracking", lambda run: None)
    return create_run(tmp_path, config)


def _complete(run):
    data = prepare_data(run)
    fit.run_preflight(run, data)
    fit.fit_sweep(run, data)
    return data


def test_default_cartesian_counts_are_registered():
    class View:
        config = ExperimentConfig()

    assert len(fit.expected_units(View())) == 105
    primary = [
        row
        for row in fit.base_rows(View(), 2026, 256, "complete")
        if row["solver"] in fit.PRIMARY_SOLVERS
    ]
    assert len(primary) * len(fit.expected_units(View())) == 420


def test_one_svd_supports_both_noise_conditions_and_all_filters(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)
    original = fit.decompose
    calls = []

    def counted(phi):
        calls.append(phi.shape)
        return original(phi)

    monkeypatch.setattr(fit, "decompose", counted)
    fit._fit_unit(tiny_run, data, 2026, 4)
    assert calls == [torch.Size([8, 4])]
    path = tiny_run.path / "trials" / fit.unit_id(2026, 4)
    assert len(read_json(path / "metrics.json")) == 4
    assert len(read_json(path / "sensitivity_metrics.json")) == 4
    assert len(list((path / "models").glob("*.pt"))) == 8
    model = load_regressor(tiny_run, seed=2026, count=4, noise_std=0.3)
    prediction = model.predict(data.validation.x, batch_size=4)
    expected = next(
        row
        for row in read_json(path / "metrics.json")
        if row["solver"] == "minimum_norm" and row["noise_std"] == 0.3
    )
    torch.testing.assert_close(
        torch.mean((prediction - data.validation.clean).square()),
        torch.tensor(expected["validation_signal_mse"], dtype=torch.float64),
    )


def test_preflight_models_reused_without_extra_decompositions(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)
    original = fit.decompose
    calls = []

    def counted(phi):
        calls.append(phi.shape)
        return original(phi)

    monkeypatch.setattr(fit, "decompose", counted)
    projection = fit.run_preflight(tiny_run, data)
    assert projection["allowed"]
    assert len(calls) == 3
    fit.fit_sweep(tiny_run, data)
    assert len(calls) == 6
    assert all(row["status"] == "complete" for row in fit.inspect_fits(tiny_run))


def test_completed_unit_not_overwritten_or_refitted(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)
    fit._fit_unit(tiny_run, data, 2026, 4)
    path = tiny_run.path / "trials" / fit.unit_id(2026, 4) / "manifest.json"
    original_hash = sha256_file(path)

    def forbidden(phi):
        pytest.fail("A verified complete unit must never be refitted.")

    monkeypatch.setattr(fit, "decompose", forbidden)
    fit._fit_unit(tiny_run, data, 2026, 4)
    assert sha256_file(path) == original_hash


def test_modified_committed_predictor_is_rejected(tiny_run):
    data = prepare_data(tiny_run)
    fit._fit_unit(tiny_run, data, 2026, 4)
    model = tiny_run.path / "trials" / fit.unit_id(2026, 4) / "models" / "ridge_noise_0.pt"
    with model.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        fit._fit_unit(tiny_run, data, 2026, 4)


def test_atomic_publication_failure_leaves_no_complete_unit(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)
    original_publish = fit.publish_directory

    def fail_publication(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(fit, "publish_directory", fail_publication)
    with pytest.raises(OSError, match="publication"):
        fit._fit_unit(tiny_run, data, 2026, 4)
    directory = tiny_run.path / "trials" / fit.unit_id(2026, 4)
    assert not directory.exists()
    assert not list(directory.parent.glob(".features_*"))
    monkeypatch.setattr(fit, "publish_directory", original_publish)
    assert fit._fit_unit(tiny_run, data, 2026, 4)["status"] == "complete"


def test_numerical_failure_is_preserved_and_not_retried(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)

    def nonfinite(phi):
        raise FloatingPointError("nonfinite test SVD")

    monkeypatch.setattr(fit, "decompose", nonfinite)
    first = fit._fit_unit(tiny_run, data, 2026, 4)
    assert first["status"] == "failed_numerical"
    assert fit._fit_unit(tiny_run, data, 2026, 4) == first
    path = tiny_run.path / "trials" / fit.unit_id(2026, 4)
    assert all(row["status"] == "failed_numerical" for row in read_json(path / "metrics.json"))
    assert not (path / "models").exists()


def test_memory_or_infrastructure_failure_stops_without_numerical_record(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)

    def failed(phi):
        raise RuntimeError("simulated allocation failure")

    monkeypatch.setattr(fit, "decompose", failed)
    with pytest.raises(RuntimeError, match="allocation"):
        fit._fit_unit(tiny_run, data, 2026, 4)
    assert not (tiny_run.path / "trials" / fit.unit_id(2026, 4)).exists()


def test_incomplete_sweep_cannot_freeze_or_generate_test(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)
    fit._fit_unit(tiny_run, data, 2026, 4)
    with pytest.raises(RuntimeError, match="unfinished"):
        freeze_evaluation(tiny_run)

    def forbidden(*args):
        pytest.fail("Test generation cannot precede a completed frozen sweep.")

    monkeypatch.setattr("src.data.generate_partition", forbidden)
    with pytest.raises(RuntimeError, match="Freeze"):
        prepare_test_data(tiny_run)
    assert not (tiny_run.path / "data" / "test.pt").exists()


def test_freeze_verifies_all_artifacts_and_is_idempotent(tiny_run):
    _complete(tiny_run)
    frozen = freeze_evaluation(tiny_run)
    assert frozen["complete_units"] == 6
    assert frozen["failed_units"] == 0
    assert freeze_evaluation(tiny_run) == frozen
    path = tiny_run.path / "trials" / fit.unit_id(2027, 8) / "models" / "ridge_noise_1.pt"
    path.unlink()
    with pytest.raises(ValueError, match="checksum"):
        verify_frozen(tiny_run)


def test_evaluation_counts_incomplete_batch_and_reuses_outputs(tiny_run, monkeypatch):
    _complete(tiny_run)
    freeze_evaluation(tiny_run)
    test = prepare_test_data(tiny_run)
    evaluated = evaluate_checkpoints(tiny_run, test)
    assert len(evaluated) == 6
    path = tiny_run.path / "evaluation" / fit.unit_id(2026, 4)
    predictions = torch.load(path / "predictions.pt", weights_only=True)
    rows = read_json(path / "metrics.json") + read_json(path / "sensitivity_metrics.json")
    for index, row in enumerate(rows):
        expected = torch.mean((predictions["predictions"][:, index] - test.clean).square())
        assert row["test_count"] == 11
        assert row["test_signal_mse"] == pytest.approx(float(expected))
    digest = sha256_file(path / "manifest.json")

    def forbidden(*args):
        pytest.fail("Completed evaluations must not be repeated.")

    monkeypatch.setattr("src.evaluate._predict_unit", forbidden)
    _evaluate_unit(
        tiny_run, test, 2026, 4, sha256_file(tiny_run.path / "evaluation" / "frozen.json")
    )
    assert sha256_file(path / "manifest.json") == digest
    baselines = read_json(tiny_run.path / "evaluation" / "baselines.json")
    teacher = next(
        row for row in baselines if row["predictor"] == "teacher" and row["noise_std"] == 0.3
    )
    assert teacher["signal_mse"] == 0
    assert teacher["theoretical_observed_mse"] == pytest.approx(0.09)


def test_wrong_partition_is_rejected_before_predictions(tiny_run):
    data = _complete(tiny_run)
    freeze_evaluation(tiny_run)
    prepare_test_data(tiny_run)
    with pytest.raises(ValueError, match="held-out"):
        evaluate_checkpoints(tiny_run, data.validation)


def test_failed_registered_units_remain_in_test_results(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)

    def nonfinite(phi):
        raise FloatingPointError("numerical failure")

    monkeypatch.setattr(fit, "decompose", nonfinite)
    for seed, count in fit.expected_units(tiny_run):
        fit._fit_unit(tiny_run, data, seed, count)
    frozen = freeze_evaluation(tiny_run)
    assert frozen["failed_units"] == 6
    test = prepare_test_data(tiny_run)
    result = evaluate_checkpoints(tiny_run, test)
    assert len(result) == 6
    assert all(row["status"] == "failed_numerical" for row in result)
    metrics = read_json(tiny_run.path / "evaluation" / fit.unit_id(2026, 4) / "metrics.json")
    assert len(metrics) == 4
    assert all("test_signal_mse" not in row for row in metrics)


def test_preflight_blocks_projected_over_budget_work(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)
    fit.run_preflight(tiny_run, data)
    original_status = fit.verified_status

    def slow_status(run, seed, count, **kwargs):
        status = dict(original_status(run, seed, count, **kwargs))
        status.update(total_seconds=1e6, prediction_seconds=1e6)
        return status

    monkeypatch.setattr(fit, "verified_status", slow_status)
    with pytest.raises(fit.PreflightBlocked, match="preflight"):
        fit.fit_sweep(tiny_run, data)
    assert not (tiny_run.path / "trials" / fit.unit_id(2027, 2)).exists()


def test_budget_stop_preserves_commit_and_blocks_freeze(tiny_run, monkeypatch):
    data = prepare_data(tiny_run)
    fit._fit_unit(tiny_run, data, 2026, 2)

    def exhausted():
        raise BudgetExceeded("simulated budget boundary")

    monkeypatch.setattr(tiny_run, "check_budget", exhausted)
    with pytest.raises(BudgetExceeded):
        fit._fit_unit(tiny_run, data, 2026, 4)
    assert fit.verified_status(tiny_run, 2026, 2)["status"] == "complete"
    assert not (tiny_run.path / "trials" / fit.unit_id(2026, 4)).exists()


def test_wrong_development_tensors_rejected(tiny_run):
    data = prepare_data(tiny_run)
    changed = replace(data, train=replace(data.train, labels=data.train.labels + 1))
    with pytest.raises(ValueError, match="registered"):
        fit.run_preflight(tiny_run, changed)


def test_unregistered_development_files_cannot_enter_preflight(tiny_run):
    data = prepare_data(tiny_run)
    (tiny_run.path / "data" / "development_manifest.json").unlink()
    with pytest.raises(ValueError, match="prepared"):
        fit.run_preflight(tiny_run, data)


def test_nonfinite_condition_is_explicit_json_safe():
    result = fit._json_summary({"condition_number": float("inf"), "coefficient_norm": 2.0})
    assert result["condition_number"] is None
    assert result["condition_number_is_infinite"]
    with pytest.raises(fit.NumericalFailure):
        fit._json_summary({"coefficient_norm": float("nan")})
