"""Frozen all-model evaluation; test data cannot select a capacity or solver."""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from tqdm.auto import tqdm

from .checkpoints import load_regressor
from .data import Partition, load_partition, partition_digest
from .features import make_features
from .fit import (
    NumericalFailure,
    _commit_rows,
    _finite_tensor,
    base_rows,
    expected_units,
    unit_id,
    verified_status,
)
from .io import (
    atomic_json,
    atomic_torch_save,
    canonical_hash,
    publish_directory,
    read_json,
    sha256_file,
)
from .solvers import metrics


def _selection(run: Any) -> dict[str, Any]:
    units = {}
    for seed, count in expected_units(run):
        path = run.path / "trials" / unit_id(seed, count)
        if not path.exists():
            raise RuntimeError(
                f"Cannot freeze an unfinished sweep: missing {unit_id(seed, count)}. "
                "Budget-stopped or unattempted units must be completed first."
            )
        status = verified_status(run, seed, count)
        manifest = read_json(path / "manifest.json")
        units[unit_id(seed, count)] = {
            "status": status["status"],
            "manifest_sha256": sha256_file(path / "manifest.json"),
            "files": manifest["files"],
        }
    return {
        "schema_version": 1,
        "policy": "all_registered_models_no_selection",
        "config_hash": canonical_hash(run.config.to_dict()),
        "development_identity": run.data_identity("development"),
        "units": units,
        "complete_units": sum(value["status"] == "complete" for value in units.values()),
        "failed_units": sum(value["status"] == "failed_numerical" for value in units.values()),
    }


def freeze_evaluation(run: Any) -> dict[str, Any]:
    """Freeze every registered terminal unit before test generation or observation."""
    with run.writer_lock(), run.work("freeze_evaluation"):
        run.verify_identity()
        run.verify_data("development")
        path = run.path / "evaluation" / "frozen.json"
        if path.exists():
            return verify_frozen(run)
        if (run.path / "data" / "test.pt").exists():
            raise RuntimeError(
                "Test data already exists without a frozen selection; start a fresh run."
            )
        selection = _selection(run)
        atomic_json(path, selection)
        return selection


def verify_frozen(run: Any) -> dict[str, Any]:
    """Verify current artifacts against frozen identities without evaluating anything."""
    with run.writer_lock(), run.work("verify_frozen"):
        run.verify_identity()
        run.verify_data("development")
        path = run.path / "evaluation" / "frozen.json"
        if not path.exists():
            raise RuntimeError(
                "Freeze every fitted-model identity before generating or evaluating test data."
            )
        frozen = read_json(path)
        if canonical_hash(frozen) != canonical_hash(_selection(run)):
            raise ValueError(
                "Frozen selection no longer matches the registered models, data or protocol."
            )
        return frozen


def _validate_test(run: Any, test: Partition) -> None:
    run.data_identity("test")
    if set(read_json(run.path / "data" / "test_manifest.json")["files"]) != {"data/test.pt"}:
        raise ValueError("Test registry does not cover exactly the registered held-out data.")
    saved = load_partition(run.path / "data" / "test.pt")
    if test.name != "test" or partition_digest(test) != partition_digest(saved):
        raise ValueError("Passed test partition differs from the registered held-out partition.")
    if len(test.x) != run.config.data.test_size:
        raise ValueError("Test sample count differs from the registered protocol.")
    payload = torch.load(run.path / "data" / "test.pt", map_location="cpu", weights_only=True)
    if payload["generation"]["frozen_identity"] != sha256_file(
        run.path / "evaluation" / "frozen.json"
    ):
        raise ValueError("The test partition was generated for a different frozen selection.")


def _baseline_rows(run: Any, test: Partition) -> list[dict[str, Any]]:
    train = load_partition(run.path / "data" / "train.pt")
    result = []
    for index, sigma in enumerate(run.config.data.noise_stds):
        for name, predicted, signal_reference in (
            (
                "training_mean",
                torch.full_like(test.clean, float(train.labels[:, index].mean())),
                None,
            ),
            ("zero", torch.zeros_like(test.clean), 1.0),
            ("teacher", test.clean, 0.0),
        ):
            measured = metrics(predicted, test.clean, test.labels[:, index])
            result.append(
                {
                    "noise_std": sigma,
                    "predictor": name,
                    **measured,
                    "theoretical_signal_mse": signal_reference,
                    "theoretical_observed_mse": None
                    if signal_reference is None
                    else signal_reference + sigma**2,
                    "reference_scope": "population_expectation; finite-sample measurements are separate",
                }
            )
    return result


def _predict_unit(
    run: Any, test: Partition, seed: int, count: int, rows: list[dict[str, Any]]
) -> torch.Tensor:
    """One feature chunk shared across all paired predictors and cutoff diagnostics."""
    source = run.path / "trials" / unit_id(seed, count) / "models"
    regressors = [
        load_regressor(source / f"{row['solver']}_noise_{row['noise_index']}.pt") for row in rows
    ]
    reference = regressors[0]
    for regressor, row in zip(regressors, rows, strict=True):
        if (
            regressor.count != count
            or regressor.metadata["seed"] != seed
            or regressor.metadata["solver"] != row["solver"]
            or regressor.metadata["noise_index"] != row["noise_index"]
            or not torch.equal(regressor.bank.frequencies, reference.bank.frequencies)
            or not torch.equal(regressor.bank.phases, reference.bank.phases)
        ):
            raise ValueError("Frozen predictor metadata or paired feature banks disagree.")
    coefficients = torch.stack([regressor.beta for regressor in regressors], dim=1)
    chunks = []
    for start in range(0, len(test.x), run.config.execution.eval_batch_size):
        chunk = test.x[start : start + run.config.execution.eval_batch_size]
        phi = make_features(chunk, reference.bank, count)
        chunks.append(phi @ coefficients)
    predictions = torch.cat(chunks)
    _finite_tensor(predictions, "test predictions")
    return predictions


def _evaluate_unit(
    run: Any, test: Partition, seed: int, count: int, frozen_hash: str
) -> dict[str, Any]:
    destination = run.path / "evaluation" / unit_id(seed, count)
    with run.work(f"evaluate/{unit_id(seed, count)}"):
        if destination.exists():
            status = verified_status(run, seed, count, root="evaluation")
            if status.get("frozen_sha256") != frozen_hash:
                raise ValueError(
                    "Previously evaluated unit belongs to a different frozen selection."
                )
            if status.get("test_identity") != run.data_identity("test"):
                raise ValueError("Previously evaluated unit belongs to different test data.")
            return status
        fit_status = verified_status(run, seed, count)
        started = time.perf_counter()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            rows = base_rows(run, seed, count, fit_status["status"])
            failure_reason = fit_status.get("failure_reason")
            status = fit_status["status"]
            prediction_seconds = 0.0
            if status == "complete":
                try:
                    prediction_start = time.perf_counter()
                    predictions = _predict_unit(run, test, seed, count, rows)
                    prediction_seconds = time.perf_counter() - prediction_start
                    for column, row in enumerate(rows):
                        measured = metrics(
                            predictions[:, column], test.clean, test.labels[:, row["noise_index"]]
                        )
                        for key, value in measured.items():
                            if isinstance(value, float) and not math.isfinite(value):
                                raise NumericalFailure(f"Nonfinite test {key}.")
                            row[f"test_{key}"] = value
                    atomic_torch_save(
                        staging / "predictions.pt",
                        {
                            "sequence_ids": list(test.ids),
                            "columns": [
                                {"solver": row["solver"], "noise_std": row["noise_std"]}
                                for row in rows
                            ],
                            "predictions": predictions,
                            "clean_targets": test.clean,
                            "observed_targets": test.labels,
                        },
                    )
                except (NumericalFailure, FloatingPointError) as error:
                    status, failure_reason = "failed_numerical", f"{type(error).__name__}: {error}"
                    run.logger.exception("Numerical test failure: %s", unit_id(seed, count))
                    rows = base_rows(run, seed, count, status)
            if status != "complete":
                for row in rows:
                    row["failure_reason"] = failure_reason
            elapsed = time.perf_counter() - started
            for row in rows:
                row.update(evaluation_seconds=elapsed, test_prediction_seconds=prediction_seconds)
            _commit_rows(staging, rows)
            unit_status = {
                "seed": seed,
                "n_features": count,
                "status": status,
                "failure_reason": failure_reason,
                "total_seconds": elapsed,
                "prediction_seconds": prediction_seconds,
                "cpu_rss_bytes": psutil.Process().memory_info().rss,
                "frozen_sha256": frozen_hash,
                "test_identity": run.data_identity("test"),
                "config_hash": canonical_hash(run.config.to_dict()),
                "development_identity": run.data_identity("development"),
            }
            atomic_json(staging / "status.json", unit_status)
            publish_directory(staging, destination)
            return unit_status
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def evaluate_checkpoints(run: Any, test_data: Partition) -> list[dict[str, Any]]:
    """Evaluate every frozen model, preserving failures and resumable completed units."""
    from .tracking import refresh_tracking

    with run.writer_lock():
        with run.work("evaluation_setup"):
            verify_frozen(run)
            _validate_test(run, test_data)
            torch.set_num_threads(run.config.execution.threads)
            frozen_hash = sha256_file(run.path / "evaluation" / "frozen.json")
        baseline_path = run.path / "evaluation" / "baselines.json"
        with run.work("test_baselines"):
            baseline_rows = _baseline_rows(run, test_data)
            if baseline_path.exists():
                if canonical_hash(read_json(baseline_path)) != canonical_hash(baseline_rows):
                    raise ValueError("Previously saved test baselines were altered.")
            else:
                atomic_json(baseline_path, baseline_rows)
        statuses = []
        try:
            for seed, count in tqdm(expected_units(run), desc="Frozen model evaluation"):
                statuses.append(_evaluate_unit(run, test_data, seed, count, frozen_hash))
        finally:
            refresh_tracking(run)
        return statuses
