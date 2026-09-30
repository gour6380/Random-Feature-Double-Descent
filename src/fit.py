"""One SVD per capacity/seed, atomically committed and explicitly budgeted."""

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

from .checkpoints import save_regressor
from .data import DevelopmentData, load_partition, partition_digest
from .features import get_bank, make_features
from .io import (
    atomic_json,
    atomic_torch_save,
    canonical_hash,
    publish_directory,
    read_json,
    verify_directory,
)
from .solvers import decompose, metrics, numerical_summary, solve_coefficients

PRIMARY_SOLVERS = ("minimum_norm", "ridge")
TERMINAL_STATUSES = ("complete", "failed_numerical")


class NumericalFailure(RuntimeError):
    """Nonfinite scientific output, recorded without changing the protocol."""


class PreflightBlocked(RuntimeError):
    """The conservative projection does not fit the current explicit budget."""


def unit_id(seed: int, count: int) -> str:
    return f"seed_{seed}/features_{count:04d}"


def expected_units(run: Any) -> list[tuple[int, int]]:
    return [
        (seed, count) for seed in run.config.features.seeds for count in run.config.features.counts
    ]


def solver_names(run: Any) -> tuple[str, ...]:
    return PRIMARY_SOLVERS + tuple(
        f"cutoff_{value:g}" for value in run.config.solver.sensitivity_rconds
    )


def base_rows(run: Any, seed: int, count: int, status: str) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "n_features": count,
            "ratio": count / run.config.data.train_size,
            "noise_std": sigma,
            "noise_index": index,
            "solver": solver,
            "status": status,
        }
        for solver in solver_names(run)
        for index, sigma in enumerate(run.config.data.noise_stds)
    ]


def verified_status(run: Any, seed: int, count: int, *, root: str = "trials") -> dict[str, Any]:
    """Do not silently trust a status file or skip altered committed work."""
    path = run.path / root / unit_id(seed, count)
    verify_directory(path)
    status = read_json(path / "status.json")
    if (status.get("seed"), status.get("n_features")) != (seed, count):
        raise ValueError(f"Wrong identity in {path}.")
    if status.get("status") not in TERMINAL_STATUSES:
        raise ValueError(f"Nonterminal committed unit: {path}.")
    if status.get("config_hash") != canonical_hash(run.config.to_dict()):
        raise ValueError(f"Committed unit belongs to a different protocol: {path}.")
    if status.get("development_identity") != run.data_identity("development"):
        raise ValueError(f"Committed unit belongs to different development data: {path}.")
    rows = read_json(path / "metrics.json") + read_json(path / "sensitivity_metrics.json")
    required = {
        (name, index)
        for name in solver_names(run)
        for index in range(len(run.config.data.noise_stds))
    }
    actual = {(row.get("solver"), row.get("noise_index")) for row in rows}
    if len(rows) != len(required) or actual != required:
        raise ValueError(f"Committed unit has missing or duplicate predictor records: {path}.")
    if any(
        row.get("seed") != seed
        or row.get("n_features") != count
        or row.get("status") != status["status"]
        or row.get("noise_std") != run.config.data.noise_stds[row["noise_index"]]
        for row in rows
    ):
        raise ValueError(f"Committed predictor metadata disagrees with unit status: {path}.")
    if status["status"] == "complete":
        if root == "trials":
            expected_files = {f"{name}_noise_{index}.pt" for name, index in required}
            if (
                not (path / "models").is_dir()
                or {file.name for file in (path / "models").iterdir()} != expected_files
            ):
                raise ValueError(f"Complete unit lacks its registered predictors: {path}.")
            if not (path / "singular_values.pt").is_file():
                raise ValueError(f"Complete unit lacks its saved singular spectrum: {path}.")
        elif not (path / "predictions.pt").is_file():
            raise ValueError(f"Complete evaluation lacks its predictions: {path}.")
    return status


def inspect_fits(run: Any) -> list[dict[str, Any]]:
    """Read-only completion view, including every not-yet-attempted unit."""
    result = []
    for seed, count in expected_units(run):
        path = run.path / "trials" / unit_id(seed, count)
        result.append(
            verified_status(run, seed, count)
            if path.exists()
            else {"seed": seed, "n_features": count, "status": "pending"}
        )
    return result


def _validate_development(run: Any, data: DevelopmentData) -> None:
    run.data_identity("development")
    required = {"data/train.pt", "data/validation.pt", "data/development.json"}
    required.update(f"data/feature_banks/seed_{seed}.pt" for seed in run.config.features.seeds)
    if set(read_json(run.path / "data" / "development_manifest.json")["files"]) != required:
        raise ValueError(
            "Development registry does not cover exactly the registered data and feature banks."
        )
    for name in ("train", "validation"):
        supplied = getattr(data, name)
        persisted = load_partition(run.path / "data" / f"{name}.pt")
        if partition_digest(supplied) != partition_digest(persisted):
            raise ValueError(f"Passed {name} data differs from the registered partition.")


def _finite_tensor(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise NumericalFailure(f"Nonfinite {label}.")


def _json_summary(summary: dict[str, Any]) -> dict[str, Any]:
    output = dict(summary)
    condition = output.get("condition_number")
    output["condition_number_is_infinite"] = condition is not None and math.isinf(condition)
    if output["condition_number_is_infinite"]:
        output["condition_number"] = None
    for name, value in output.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise NumericalFailure(f"Nonfinite numerical diagnostic {name}.")
    return output


def _development_predictions(
    data: DevelopmentData, bank: Any, count: int, all_beta: torch.Tensor, batch_size: int
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(data.validation.x), batch_size):
        phi = make_features(data.validation.x[start : start + batch_size], bank, count)
        chunks.append(phi @ all_beta)
    result = torch.cat(chunks)
    _finite_tensor(result, "validation predictions")
    return result


def _commit_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_json(path / "metrics.json", [row for row in rows if row["solver"] in PRIMARY_SOLVERS])
    atomic_json(
        path / "sensitivity_metrics.json",
        [row for row in rows if row["solver"] not in PRIMARY_SOLVERS],
    )


def _fit_unit(run: Any, data: DevelopmentData, seed: int, count: int) -> dict[str, Any]:
    destination = run.path / "trials" / unit_id(seed, count)
    with run.work(f"fit/{unit_id(seed, count)}"):
        if destination.exists():
            return verified_status(run, seed, count)
        started = time.perf_counter()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            rows = base_rows(run, seed, count, "complete")
            timings = {"feature_seconds": 0.0, "svd_seconds": 0.0, "prediction_seconds": 0.0}
            svd_count = 0
            try:
                feature_start = time.perf_counter()
                bank = get_bank(run, seed)
                phi = make_features(data.train.x, bank, count)
                _finite_tensor(phi, "training feature matrix")
                timings["feature_seconds"] = time.perf_counter() - feature_start
                svd_start = time.perf_counter()
                svd_count = 1
                decomposition = decompose(phi)
                timings["svd_seconds"] = time.perf_counter() - svd_start
                _finite_tensor(decomposition.s, "singular spectrum")
                coefficients = solve_coefficients(
                    decomposition, data.train.labels, len(data.train.x), run.config.solver
                )
                names = solver_names(run)
                all_beta = torch.cat([coefficients[name] for name in names], dim=1)
                _finite_tensor(all_beta, "coefficients")
                prediction_start = time.perf_counter()
                training_prediction = phi @ all_beta
                _finite_tensor(training_prediction, "training predictions")
                validation_prediction = _development_predictions(
                    data, bank, count, all_beta, run.config.execution.eval_batch_size
                )
                timings["prediction_seconds"] = time.perf_counter() - prediction_start
                summaries: dict[str, list[dict[str, Any]]] = {}
                for name in names:
                    cutoff = (
                        float(name.removeprefix("cutoff_"))
                        if name.startswith("cutoff_")
                        else run.config.solver.rcond
                    )
                    summaries[name] = numerical_summary(
                        phi,
                        data.train.labels,
                        coefficients[name],
                        decomposition,
                        run.config.solver,
                        rcond=cutoff,
                    )
                n_noise = len(run.config.data.noise_stds)
                for row in rows:
                    name, index = row["solver"], row["noise_index"]
                    column = names.index(name) * n_noise + index
                    row.update(_json_summary(summaries[name][index]))
                    for partition, predictions in (
                        (data.train, training_prediction),
                        (data.validation, validation_prediction),
                    ):
                        measured = metrics(
                            predictions[:, column], partition.clean, partition.labels[:, index]
                        )
                        for key, value in measured.items():
                            if isinstance(value, float) and not math.isfinite(value):
                                raise NumericalFailure(f"Nonfinite {partition.name} {key}.")
                            row[f"{partition.name}_{key}"] = value
                    metadata = {
                        "seed": seed,
                        "n_features": count,
                        "noise_std": row["noise_std"],
                        "noise_index": index,
                        "solver": name,
                        "rcond": float(name.removeprefix("cutoff_"))
                        if name.startswith("cutoff_")
                        else run.config.solver.rcond,
                        "ridge_lambda": run.config.solver.ridge_lambda if name == "ridge" else None,
                        "n_training_samples": len(data.train.x),
                        "training_label_mean": float(data.train.labels[:, index].mean()),
                        "role": "primary" if name in PRIMARY_SOLVERS else "cutoff_sensitivity_only",
                    }
                    model_path = staging / "models" / f"{name}_noise_{index}.pt"
                    model_path.parent.mkdir(exist_ok=True)
                    save_regressor(model_path, bank, count, coefficients[name][:, index], metadata)
                atomic_torch_save(staging / "singular_values.pt", decomposition.s)
                status = "complete"
                reason = None
            except (NumericalFailure, FloatingPointError, torch.linalg.LinAlgError) as error:
                status, reason = "failed_numerical", f"{type(error).__name__}: {error}"
                run.logger.exception("Numerical failure: %s", unit_id(seed, count))
                # Partial models never masquerade as complete scientific output.
                if (staging / "models").exists():
                    shutil.rmtree(staging / "models")
                rows = base_rows(run, seed, count, status)
                for row in rows:
                    row["failure_reason"] = reason
            elapsed = time.perf_counter() - started
            rss = psutil.Process().memory_info().rss
            for row in rows:
                row.update(timings, total_seconds=elapsed, cpu_rss_bytes=rss)
            _commit_rows(staging, rows)
            unit_status = {
                "seed": seed,
                "n_features": count,
                "status": status,
                "failure_reason": reason,
                **timings,
                "total_seconds": elapsed,
                "cpu_rss_bytes": rss,
                "svd_count": svd_count,
                "config_hash": canonical_hash(run.config.to_dict()),
                "development_identity": run.data_identity("development"),
            }
            atomic_json(staging / "status.json", unit_status)
            publish_directory(staging, destination)
            run.logger.info("Committed %s: %s", unit_id(seed, count), status)
            return unit_status
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _projection(run: Any) -> dict[str, Any]:
    """Conservative measured buckets; prediction remains an estimate, not a promise."""
    widths = sorted(run.config.execution.preflight_widths)
    if widths[-1] != max(run.config.features.counts):
        raise ValueError("The largest registered width must be included in preflight.")
    samples = [verified_status(run, run.config.features.seeds[0], count) for count in widths]
    budget = run.budget_status()
    if any(sample["status"] != "complete" for sample in samples):
        return {
            "allowed": False,
            "reason": "A numerical preflight failure prevents a reliable timing projection.",
            "budget": budget,
            "samples": samples,
        }
    fit_remaining = 0.0
    evaluation_remaining = 0.0
    for seed, count in expected_units(run):
        bucket = next(index for index, width in enumerate(widths) if width >= count)
        upper_fit = max(sample["total_seconds"] for sample in samples[: bucket + 1])
        upper_prediction = max(sample["prediction_seconds"] for sample in samples[: bucket + 1])
        if not (run.path / "trials" / unit_id(seed, count)).exists():
            fit_remaining += upper_fit
        if not (run.path / "evaluation" / unit_id(seed, count)).exists():
            evaluation_remaining += max(
                upper_fit,
                upper_prediction * run.config.data.test_size / run.config.data.validation_size,
            )
    reserve = 120.0  # Test preparation, aggregation, eight figures and report; measured later.
    projected_remaining = 2.0 * (fit_remaining + evaluation_remaining) + reserve
    projected_total = budget["used_seconds"] + projected_remaining
    return {
        "allowed": projected_remaining <= budget["remaining_seconds"],
        "reason": "Timing is approximate; upper-width buckets, a 2x multiplier and a 120s preparation/report reserve are used.",
        "fit_seconds_before_safety_factor": fit_remaining,
        "evaluation_seconds_before_safety_factor": evaluation_remaining,
        "reserve_seconds": reserve,
        "safety_multiplier": 2.0,
        "projected_remaining_seconds": projected_remaining,
        "projected_total_seconds": projected_total,
        "budget": budget,
        "samples": samples,
    }


def run_preflight(run: Any, data: DevelopmentData) -> dict[str, Any]:
    """Measure three registered fit units and retain their models for the sweep."""
    from .runtime import require_runtime_validation
    from .tracking import refresh_tracking

    with run.writer_lock():
        with run.work("preflight_setup"):
            run.verify_identity()
            require_runtime_validation(run)
            _validate_development(run, data)
            torch.set_num_threads(run.config.execution.threads)
        try:
            for count in tqdm(
                run.config.execution.preflight_widths, desc="Preflight decompositions"
            ):
                _fit_unit(run, data, run.config.features.seeds[0], count)
        finally:
            refresh_tracking(run)
        with run.work("preflight_projection"):
            projection = _projection(run)
            atomic_json(run.path / "preflight.json", projection)
        return projection


def fit_sweep(run: Any, data: DevelopmentData) -> list[dict[str, Any]]:
    """Sequential complete-seed sweep, reusing verified preflight/resume units."""
    from .runtime import require_runtime_validation
    from .tracking import refresh_tracking

    with run.writer_lock():
        with run.work("sweep_setup"):
            run.verify_identity()
            require_runtime_validation(run)
            _validate_development(run, data)
            torch.set_num_threads(run.config.execution.threads)
            if not (run.path / "preflight.json").exists():
                raise PreflightBlocked("Run the timing preflight before starting the sweep.")
        with run.work("sweep_projection"):
            projection = _projection(run)
            atomic_json(run.path / "preflight.json", projection)
            if not projection["allowed"]:
                raise PreflightBlocked(
                    f"Sweep blocked by the measured preflight: {projection['reason']} "
                    "Keep the current artifacts; any budget extension must be explicit and recorded."
                )
        try:
            for seed, count in tqdm(expected_units(run), desc="Registered decompositions"):
                _fit_unit(run, data, seed, count)
        finally:
            refresh_tracking(run)
        with run.work("verify_sweep_completion"):
            return inspect_fits(run)
