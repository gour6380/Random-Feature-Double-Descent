"""Saved-artifact aggregation and separate, reproducible analysis figures."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .io import atomic_json, sha256_file

if TYPE_CHECKING:
    from .runs import Run

KEYS = ["seed", "n_features", "noise_std", "solver"]
METRICS = [
    "train_signal_mse",
    "train_observed_mse",
    "validation_signal_mse",
    "validation_observed_mse",
    "test_signal_mse",
    "test_observed_mse",
    "coefficient_norm",
    "relative_residual",
    "numerical_rank",
    "condition_number",
]
COLORS = {"minimum_norm": "#2369a1", "ridge": "#dc7437"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


@contextmanager
def _destination(path: Path):
    """Atomic individual-file publication; a report lists only complete files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".pending-", suffix=path.suffix, dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    with _destination(path) as temporary:
        frame.to_csv(temporary, index=False)


def _write_text(text: str, path: Path) -> None:
    with _destination(path) as temporary:
        temporary.write_text(text)


def _protected(function):
    @wraps(function)
    def wrapped(run: Run, *args, **kwargs):
        from .evaluate import verify_frozen

        with run.writer_lock(), run.work(function.__name__):
            run.verify_identity()
            verify_frozen(run)
            return function(run, *args, **kwargs)

    return wrapped


def _records(root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"seed_*/features_*/{filename}")):
        value = _read(path)
        if not isinstance(value, list):
            raise ValueError(f"Expected a list of measurements in {path}.")
        rows.extend(value)
    return rows


def _unique_frame(records: list[dict[str, Any]], name: str) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    if frame.empty or not set(KEYS).issubset(frame):
        raise RuntimeError(f"{name} is missing. Complete the registered workflow first.")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate identities in {name}.")
    return frame


def _summarize(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for identity, group in frame.groupby(groups, dropna=False, sort=True):
        values = identity if isinstance(identity, tuple) else (identity,)
        row = dict(zip(groups, values, strict=True))
        row.update(
            registered_count=len(group),
            complete_count=int(group["status"].eq("complete").sum()),
            failed_count=int(group["status"].ne("complete").sum()),
        )
        for metric in metrics:
            if metric not in group:
                continue
            measurements = pd.to_numeric(group[metric], errors="coerce")
            usable = measurements[group["status"].eq("complete") & measurements.notna()]
            row[f"{metric}_count"] = len(usable)
            row[f"{metric}_mean"] = usable.mean() if len(usable) else np.nan
            row[f"{metric}_std"] = usable.std(ddof=1) if len(usable) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


@_protected
def aggregate_results(run: Run) -> dict[str, pd.DataFrame]:
    """Join every registered predictor; never drop failed conditions from the grid."""
    from .fit import expected_units, verified_status

    frozen_hash = sha256_file(run.path / "evaluation" / "frozen.json")
    test_identity = run.data_identity("test")
    for seed, count in expected_units(run):
        verified_status(run, seed, count)
        status = verified_status(run, seed, count, root="evaluation")
        if (
            status.get("frozen_sha256") != frozen_hash
            or status.get("test_identity") != test_identity
        ):
            raise ValueError("Evaluation unit differs from the frozen models or held-out data.")
    expected = pd.DataFrame(
        [
            dict(seed=seed, n_features=count, noise_std=noise, solver=solver)
            for seed in run.config.features.seeds
            for count in run.config.features.counts
            for noise in run.config.data.noise_stds
            for solver in ("minimum_norm", "ridge")
        ]
    )
    fits = _unique_frame(_records(run.path / "trials", "metrics.json"), "fit metrics")
    evaluated = _unique_frame(_records(run.path / "evaluation", "metrics.json"), "test metrics")
    expected_ids = set(map(tuple, expected[KEYS].to_numpy()))
    for frame, name in ((fits, "fit"), (evaluated, "test")):
        if set(map(tuple, frame[KEYS].to_numpy())) != expected_ids:
            raise RuntimeError(f"The {name} identity grid is incomplete or unexpected.")
    combined = expected.merge(fits, on=KEYS, validate="one_to_one")
    test_columns = KEYS + [column for column in evaluated if column.startswith("test_")]
    test_columns.extend(column for column in ("status", "failure_reason") if column in evaluated)
    combined = combined.merge(
        evaluated[test_columns], on=KEYS, validate="one_to_one", suffixes=("", "_test")
    )
    if "status_test" in combined:
        combined["fit_status"] = combined["status"]
        failed_test = combined["status_test"].ne("complete")
        combined.loc[failed_test, "status"] = combined.loc[failed_test, "status_test"]
    for metric in METRICS + ["train_count", "validation_count", "test_count", "interpolates"]:
        if metric not in combined:
            combined[metric] = np.nan
    combined["ratio"] = combined["n_features"] / run.config.data.train_size
    if "condition_number_is_infinite" in combined:
        combined.loc[
            combined["condition_number_is_infinite"].fillna(False).astype(bool),
            "condition_number",
        ] = np.inf
    summary = _summarize(combined, ["n_features", "ratio", "noise_std", "solver"], METRICS)
    clean_noise = min(run.config.data.noise_stds)
    paired_rows = []
    for noise in run.config.data.noise_stds:
        if noise == clean_noise:
            continue
        left = combined[combined.noise_std.eq(clean_noise)]
        right = combined[combined.noise_std.eq(noise)]
        pairs = right.merge(
            left,
            on=["seed", "n_features", "solver"],
            suffixes=("_noisy", "_reference"),
            validate="one_to_one",
        )
        for record in pairs.to_dict("records"):
            complete = record["status_noisy"] == record["status_reference"] == "complete"
            paired_rows.append(
                dict(
                    seed=record["seed"],
                    n_features=record["n_features"],
                    ratio=record["ratio_noisy"],
                    noise_std=noise,
                    reference_noise_std=clean_noise,
                    solver=record["solver"],
                    status="complete" if complete else "failed_pair",
                    test_signal_mse_difference=(
                        record["test_signal_mse_noisy"] - record["test_signal_mse_reference"]
                        if complete
                        else np.nan
                    ),
                )
            )
    paired = pd.DataFrame(paired_rows)
    pair_summary = (
        _summarize(
            paired,
            ["n_features", "ratio", "noise_std", "solver"],
            ["test_signal_mse_difference"],
        )
        if not paired.empty
        else pd.DataFrame()
    )
    sensitivity = _unique_frame(
        _records(run.path / "trials", "sensitivity_metrics.json"), "sensitivity fit metrics"
    )
    sensitivity_test = _unique_frame(
        _records(run.path / "evaluation", "sensitivity_metrics.json"), "sensitivity test metrics"
    )
    test_columns = KEYS + [column for column in sensitivity_test if column.startswith("test_")]
    test_columns.extend(
        column for column in ("status", "failure_reason") if column in sensitivity_test
    )
    sensitivity = sensitivity.merge(
        sensitivity_test[test_columns], on=KEYS, validate="one_to_one", suffixes=("", "_test")
    )
    sensitivity["fit_status"] = sensitivity["status"]
    if "status_test" in sensitivity:
        failed_test = sensitivity["status_test"].ne("complete")
        sensitivity.loc[failed_test, "status"] = sensitivity.loc[failed_test, "status_test"]
    for metric in METRICS:
        if metric not in sensitivity:
            sensitivity[metric] = np.nan
    units = []
    for path in sorted((run.path / "trials").glob("seed_*/features_*/status.json")):
        units.append(_read(path))
    frames = dict(
        primary=combined,
        aggregate=summary,
        paired=paired,
        paired_aggregate=pair_summary,
        sensitivity=sensitivity,
        units=pd.DataFrame(units),
    )
    for name, frame in frames.items():
        _write_csv(frame, run.path / "tables" / f"{name}.csv")
    manifest = {
        "registered_predictors": len(expected),
        "complete_predictors": int(combined.status.eq("complete").sum()),
        "failed_predictors": int(combined.status.ne("complete").sum()),
        "feature_seeds": list(run.config.features.seeds),
        "dispersion": "sample standard deviation across feature seeds, conditional on fixed data",
        "infinity_encoding": "inf in CSV; condition_number_is_infinite in source JSON",
    }
    _write_text(json.dumps(manifest, indent=2) + "\n", run.path / "tables" / "summary.json")
    atomic_json(
        run.path / "tables" / "table_hashes.json",
        {path.name: sha256_file(path) for path in (run.path / "tables").glob("*.csv")},
    )
    return frames


def inspect_development(run: Run) -> pd.DataFrame:
    """Read saved development rows before the frozen test boundary."""
    run.verify_identity()
    from .fit import inspect_fits

    inspect_fits(run)
    return pd.DataFrame(_records(run.path / "trials", "metrics.json"))


def _table(run: Run, name: str) -> pd.DataFrame:
    path = run.path / "tables" / f"{name}.csv"
    if not path.exists():
        raise RuntimeError("Execute aggregate_results(run) before generating analysis figures.")
    hashes = _read(run.path / "tables" / "table_hashes.json")
    if hashes.get(path.name) != sha256_file(path):
        raise ValueError(
            "Derived table changed; regenerate aggregate_results from committed artifacts."
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _seed_curve(ax, frame: pd.DataFrame, metric: str, label: str, color: str) -> None:
    if frame.empty or metric not in frame:
        ax.text(0.5, 0.5, "No registered measurements", transform=ax.transAxes, ha="center")
        return
    usable = frame.copy()
    usable.loc[usable.status.ne("complete"), metric] = np.nan
    for _, group in usable.groupby("seed"):
        ordered = group.sort_values("ratio")
        ax.plot(ordered.ratio, ordered[metric], color=color, alpha=0.22, linewidth=0.8)
    grouped = usable.groupby("ratio")[metric]
    mean, std, count = grouped.mean(), grouped.std(ddof=1), grouped.count()
    ax.plot(mean.index, mean.values, color=color, marker="o", markersize=3, label=label)
    ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.13)
    if len(count):
        for x, y, n in zip(mean.index, mean, count, strict=True):
            if np.isfinite(y):
                ax.annotate(
                    str(n),
                    (x, y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=5,
                    color=color,
                    ha="center",
                )
        ax.text(0.02, 0.02, "Numerals: contributing seeds", transform=ax.transAxes, fontsize=7)
    failures = int(frame.status.ne("complete").sum())
    if failures:
        ax.text(
            0.02,
            0.96,
            f"{failures} failed predictor rows; gaps retained",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
    ax.axvline(1, color="#777777", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Features / training examples (p/n)")
    ax.grid(alpha=0.18)


def _finish(run: Run, figure, name: str, title: str):
    figure.suptitle(title, fontsize=14)
    figure.text(
        0.5,
        0.006,
        "Curves: thin individual seeds · thick mean · band sample SD (not a confidence interval)",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    for extension in ("png", "svg"):
        with _destination(run.path / "figures" / f"{name}.{extension}") as temporary:
            figure.savefig(temporary, dpi=170, bbox_inches="tight")
    atomic_json(
        run.path / "figures" / f"{name}.json",
        {
            "title": title,
            "table_hashes": _read(run.path / "tables" / "table_hashes.json"),
            "formats": ["png", "svg"],
            "dispersion": "sample SD across feature seeds, conditional on fixed data",
        },
    )
    plt.close(figure)
    return figure


@_protected
def plot_error_curves(run: Run):
    frame = _table(run, "primary")
    noises = tuple(run.config.data.noise_stds)
    figure, axes = plt.subplots(len(noises), 2, figsize=(12, 4 * len(noises)), squeeze=False)
    for row, noise in enumerate(noises):
        for column, metric in enumerate(("train_observed_mse", "test_signal_mse")):
            ax = axes[row, column]
            for solver, color in COLORS.items():
                subset = frame[frame.noise_std.eq(noise) & frame.solver.eq(solver)]
                _seed_curve(ax, subset, metric, solver, color)
            if column == 1:
                ax.axhline(1, linestyle="--", color="#555555", label="Zero predictor: E[MSE]=1")
            ax.set_yscale("symlog", linthresh=1e-12)
            ax.set_ylabel("MSE (symmetric log; zeros retained)")
            ax.set_title(
                f"σ={noise:g} · {'training observed labels' if column == 0 else 'test clean signal'}"
            )
            ax.legend(fontsize=8)
    return _finish(
        run, figure, "01_error_curves", "Capacity, training fit and clean-target test error"
    )


@_protected
def plot_noise_differences(run: Run):
    frame = _table(run, "paired")
    figure, ax = plt.subplots(figsize=(10, 5))
    if not frame.empty:
        for (noise, solver), group in frame.groupby(["noise_std", "solver"]):
            _seed_curve(
                ax,
                group,
                "test_signal_mse_difference",
                f"σ={noise:g} minus reference · {solver}",
                COLORS[solver],
            )
        ax.legend(fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_ylabel("Paired clean-test MSE difference (symmetric log)")
    return _finish(run, figure, "02_noise_differences", "Paired effects of observation noise")


@_protected
def plot_solver_comparison(run: Run):
    frame = _table(run, "primary")
    noises = tuple(run.config.data.noise_stds)
    figure, axes = plt.subplots(1, len(noises), figsize=(6 * len(noises), 5), squeeze=False)
    for ax, noise in zip(axes[0], noises, strict=True):
        subset = frame[frame.noise_std.eq(noise)]
        for solver, color in COLORS.items():
            _seed_curve(ax, subset[subset.solver.eq(solver)], "test_signal_mse", solver, color)
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_ylabel("Clean-test MSE (symmetric log)")
        ax.set_title(f"σ={noise:g}; fixed ridge λ={run.config.solver.ridge_lambda:g}")
        ax.legend(fontsize=8)
    return _finish(
        run, figure, "03_solver_comparison", "Minimum norm and one fixed ridge comparison"
    )


@_protected
def plot_rank_and_interpolation(run: Run):
    frame = _table(run, "primary")
    minimum = frame[frame.solver.eq("minimum_norm")].copy()
    reference = minimum[minimum.noise_std.eq(min(run.config.data.noise_stds))]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    _seed_curve(axes[0], reference, "numerical_rank", "Numerical rank", COLORS["minimum_norm"])
    axes[0].axhline(run.config.data.train_size, color="#555555", linestyle="--", label="n")
    axes[0].set_ylabel("Retained singular values")
    axes[0].legend(fontsize=8)
    for index, (noise, group) in enumerate(minimum.groupby("noise_std")):
        group = group.copy()
        group["interpolation_fraction"] = group["interpolates"].map(
            lambda value: (
                float(value)
                if isinstance(value, (bool, np.bool_, int, float))
                else {"True": 1.0, "False": 0.0}.get(str(value), np.nan)
            )
        )
        _seed_curve(axes[1], group, "interpolation_fraction", f"σ={noise:g}", f"C{index}")
    axes[1].set_ylabel("Fraction meeting measured residual threshold")
    axes[1].set_ylim(-0.55, 1.55)
    axes[1].legend(fontsize=8)
    return _finish(
        run,
        figure,
        "04_rank_and_interpolation",
        "Square matrices and numerical fitting are different measurements",
    )


@_protected
def plot_conditioning_and_coefficients(run: Run):
    frame = _table(run, "primary")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    reference = frame[
        frame.solver.eq("minimum_norm") & frame.noise_std.eq(min(run.config.data.noise_stds))
    ].copy()
    infinite = np.isinf(reference.condition_number)
    reference.loc[infinite, "condition_number"] = np.nan
    _seed_curve(
        axes[0], reference, "condition_number", "Finite condition number", COLORS["minimum_norm"]
    )
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("Condition number (symmetric log)")
    for ratio in sorted(reference.loc[infinite, "ratio"].unique()):
        axes[0].text(ratio, 0.88, "∞", transform=axes[0].get_xaxis_transform(), ha="center")
    axes[0].set_title(f"Infinite values marked ∞ ({int(infinite.sum())} rows); retained in tables")
    for (noise, solver), group in frame.groupby(["noise_std", "solver"]):
        _seed_curve(axes[1], group, "coefficient_norm", f"σ={noise:g} · {solver}", COLORS[solver])
    axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[1].set_ylabel("Coefficient L2 norm (symmetric log)")
    axes[1].legend(fontsize=8)
    return _finish(
        run,
        figure,
        "05_conditioning_and_coefficients",
        "Numerical conditioning and coefficient size",
    )


@_protected
def plot_singular_spectra(run: Run):
    n = run.config.data.train_size
    counts = sorted(run.config.features.counts, key=lambda count: abs(count - n))[:3]
    figure, axes = plt.subplots(1, len(counts), figsize=(5 * len(counts), 5), squeeze=False)
    for ax, count in zip(axes[0], sorted(counts), strict=True):
        spectra = []
        for seed in run.config.features.seeds:
            path = (
                run.path
                / "trials"
                / f"seed_{seed}"
                / f"features_{count:04d}"
                / "singular_values.pt"
            )
            if not path.exists():
                continue
            singular = torch.load(path, map_location="cpu", weights_only=True).numpy()
            spectra.append(singular)
            ax.plot(np.arange(1, len(singular) + 1), singular, alpha=0.22, color="C0")
        if spectra:
            values = np.stack(spectra)
            x = np.arange(1, values.shape[1] + 1)
            mean = values.mean(axis=0)
            ax.plot(x, mean, color="C0")
            if len(values) > 1:
                std = values.std(axis=0, ddof=1)
                ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.15)
        ax.set_yscale("symlog", linthresh=1e-14)
        ax.set_title(f"p={count}; {len(spectra)}/{len(run.config.features.seeds)} spectra")
        ax.set_xlabel("Singular-value index (descending)")
        ax.set_ylabel("Singular value (symmetric log; zeros retained)")
        ax.grid(alpha=0.18)
    return _finish(
        run, figure, "06_singular_spectra", "Singular spectra nearest the square-matrix boundary"
    )


@_protected
def plot_cutoff_sensitivity(run: Run):
    primary = _table(run, "primary")
    sensitivity = _table(run, "sensitivity")
    primary = primary[primary.solver.eq("minimum_norm")].copy()
    primary["solver"] = f"primary cutoff {run.config.solver.rcond:g}"
    frame = pd.concat([primary, sensitivity], ignore_index=True)
    if "fit_status" in frame:
        frame["status"] = frame["fit_status"].fillna(frame["status"])
    frame["ratio"] = frame.n_features / run.config.data.train_size
    noises = tuple(run.config.data.noise_stds)
    figure, axes = plt.subplots(1, len(noises), figsize=(6 * len(noises), 5), squeeze=False)
    for ax, noise in zip(axes[0], noises, strict=True):
        for index, (solver, group) in enumerate(frame[frame.noise_std.eq(noise)].groupby("solver")):
            _seed_curve(ax, group, "validation_signal_mse", solver, f"C{index}")
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_ylabel("Validation clean-signal MSE (symmetric log)")
        ax.set_title(f"σ={noise:g}; diagnostics only")
        ax.legend(fontsize=8)
    return _finish(
        run,
        figure,
        "07_cutoff_sensitivity",
        "Pseudoinverse cutoff sensitivity; primary cutoff remains fixed",
    )


@_protected
def plot_runtime_and_status(run: Run):
    units = _table(run, "units")
    frame = _table(run, "primary")
    units["ratio"] = units.n_features / run.config.data.train_size
    units["fit_status"] = units["status"]
    units["status"] = "complete"  # A failed fit still has a measured elapsed duration.
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    _seed_curve(
        axes[0],
        units,
        "total_seconds",
        "Fit-unit active duration (including failed fits)",
        COLORS["minimum_norm"],
    )
    axes[0].set_ylabel("Seconds (one SVD shared across conditions)")
    status = frame.groupby(["n_features", "status"]).size().unstack(fill_value=0)
    status.plot.bar(stacked=True, ax=axes[1], colormap="Set2")
    axes[1].set_ylabel("Registered primary predictors")
    axes[1].set_xlabel("Feature count")
    axes[1].tick_params(axis="x", labelsize=7)
    return _finish(
        run, figure, "08_runtime_and_status", "Measured fit costs and explicit completion outcomes"
    )
