"""Export/redraw the completed study; never import or execute the experiment.

From the checkout: ``.venv/bin/python -B tools/build_results.py``.
The default verifies and redraws the already published CSV measurements.
If the package is absent, export verifies the recorded local run before copying evidence.
This is a presentation utility, not an experiment command-line interface.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import zipfile
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reviews" / "plot-cache"))

import matplotlib  # noqa: E402 — set the project-local font cache before import.

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — select the noninteractive backend first.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#2463A3"
ORANGE = "#B65023"
TEAL = "#087F80"
INK = "#183047"
GRAY = "#657581"
COLORS = {"minimum_norm": BLUE, "ridge": ORANGE}
LABELS = {"minimum_norm": "Minimum norm", "ridge": "Fixed ridge"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_files(base: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        path = (base / name).resolve()
        path.relative_to(base.resolve())
        if sha256(path) != digest:
            raise ValueError(f"Saved artifact hash mismatch: {name}")


def _tensor_spec(storage, offset, shape, stride, requires_grad, hooks):
    if requires_grad or hooks:
        raise ValueError("Expected a saved diagnostic vector without autograd hooks")
    return storage, offset, shape, stride


class _SavedVectorReader(pickle.Unpickler):
    """Read only the tensor descriptor; never import a pickle-specified module."""

    def find_class(self, module, name):
        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"): _tensor_spec,
            ("torch", "DoubleStorage"): "float64-storage",
            ("collections", "OrderedDict"): OrderedDict,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported saved-vector descriptor: {module}.{name}")
        return allowed[(module, name)]

    def persistent_load(self, identity):
        kind, dtype, key, location, size = identity
        if kind != "storage" or dtype != "float64-storage" or location != "cpu":
            raise ValueError("Expected a CPU float64 diagnostic vector")
        return str(key), int(size)


def saved_singular_values(path: Path, expected_size: int) -> np.ndarray:
    """Decode an already saved, hash-verified vector, without torch or model loading."""
    with zipfile.ZipFile(path) as archive:
        descriptor = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        prefix = descriptor.rsplit("/", 1)[0]
        storage, offset, shape, stride = _SavedVectorReader(
            io.BytesIO(archive.read(descriptor))
        ).load()
        key, size = storage
        if offset != 0 or shape != (expected_size,) or stride != (1,) or size != expected_size:
            raise ValueError("Unexpected singular-vector shape or layout")
        byteorder = archive.read(f"{prefix}/byteorder").decode()
        if byteorder not in {"little", "big"}:
            raise ValueError("Unknown saved tensor byte order")
        values = np.frombuffer(
            archive.read(f"{prefix}/data/{key}"), dtype="<f8" if byteorder == "little" else ">f8"
        )
        if len(values) != size or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Expected a finite positive saved singular spectrum")
        return values.copy()


def export_saved_run(root: Path, destination: Path) -> None:
    """Copy saved evidence; preserve every file in the private local run."""
    run = root / "runs" / "run_001"
    identity = read_json(run / "config" / "identity.json")
    config = read_json(run / "config" / "experiment.json")
    if canonical_hash(config) != identity["config_hash"]:
        raise ValueError("Saved configuration no longer matches its registered identity")
    environment = read_json(run / "environment" / "environment.json")
    if canonical_hash(environment) != identity["environment_hash"]:
        raise ValueError("Saved environment identity mismatch")
    verify_files(run / "source_snapshot", identity["source"])
    verify_files(run, read_json(run / "report" / "artifact_hashes.json")["files"])
    scopes = {
        scope: read_json(run / "data" / f"{scope}_manifest.json")
        for scope in ("development", "test")
    }
    for manifest in scopes.values():
        verify_files(run, manifest["files"])
    frozen = read_json(run / "evaluation" / "frozen.json")
    frozen_digest = sha256(run / "evaluation" / "frozen.json")
    if frozen_digest != "b67ed19803cb398d4c9a9c41e1a3bae0460b6aca0556069857bcf11d5230b42c":
        raise ValueError(
            "This presentation is for the recorded expanded run; a new run needs a new report"
        )
    expected = {
        f"seed_{seed}/features_{count:04d}"
        for seed in config["features"]["seeds"]
        for count in config["features"]["counts"]
    }
    if set(frozen["units"]) != expected or frozen["failed_units"]:
        raise ValueError("This public report requires the completed registered grid")
    if frozen["config_hash"] != identity["config_hash"] or frozen[
        "development_identity"
    ] != canonical_hash(scopes["development"]):
        raise ValueError("Frozen configuration or development identity mismatch")
    unit_evidence = {}
    for key, entry in frozen["units"].items():
        trial = run / "trials" / key
        verify_files(trial, entry["files"])
        if sha256(trial / "manifest.json") != entry["manifest_sha256"]:
            raise ValueError(f"Frozen manifest changed: {key}")
        evaluated = run / "evaluation" / key
        manifest = read_json(evaluated / "manifest.json")
        verify_files(evaluated, manifest["files"])
        status = read_json(evaluated / "status.json")
        if (
            status["status"] != "complete"
            or status["frozen_sha256"] != frozen_digest
            or status["test_identity"] != canonical_hash(scopes["test"])
        ):
            raise ValueError(f"Evaluation identity mismatch: {key}")
        unit_evidence[key] = {
            "fit_manifest_sha256": entry["manifest_sha256"],
            "evaluation_manifest_sha256": sha256(evaluated / "manifest.json"),
            "model_sha256": {
                name: digest
                for name, digest in entry["files"].items()
                if name.startswith("models/")
            },
            "predictions_sha256": manifest["files"]["predictions.pt"],
            "singular_values_sha256": entry["files"]["singular_values.pt"],
        }
    destination.mkdir(parents=True, exist_ok=True)
    for subdirectory in ("tables", "provenance", "figures", "historical_source"):
        (destination / subdirectory).mkdir(exist_ok=True)
    for source in sorted((run / "tables").iterdir()):
        if source.suffix in {".csv", ".json"}:
            shutil.copyfile(source, destination / "tables" / source.name)
    shutil.copyfile(run / "config" / "experiment.json", destination / "config.json")
    shutil.copyfile(
        run / "evaluation" / "baselines.json", destination / "tables" / "baselines.json"
    )
    for relative in identity["source"]:
        target = destination / "historical_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / "source_snapshot" / relative, target)
    sanitized_environment = dict(environment)
    sanitized_environment["executable"] = ".venv/bin/python"
    write_json(destination / "provenance" / "environment.json", sanitized_environment)
    receipt = read_json(run / "logs" / "runtime_checks.json")
    if (
        not receipt["passed"]
        or receipt["source_hash"] != canonical_hash(identity["source"])
        or receipt["environment_hash"] != identity["environment_hash"]
    ):
        raise ValueError("Saved runtime correctness receipt differs from the completed run")
    write_json(destination / "provenance" / "runtime_checks.json", receipt)
    output = re.sub(r"\x1b\[[0-9;]*m", "", (run / "logs" / "runtime_checks.txt").read_text())
    (destination / "provenance" / "runtime_checks.txt").write_text(output)
    write_json(destination / "provenance" / "budget.json", read_json(run / "logs" / "budget.json"))
    preflight = read_json(run / "preflight.json")
    write_json(
        destination / "provenance" / "preflight.json",
        {key: value for key, value in preflight.items() if key != "budget"},
    )
    write_json(destination / "provenance" / "unit_artifact_hashes.json", unit_evidence)
    write_json(
        destination / "provenance" / "identity.json",
        {
            **identity,
            "source_hash": canonical_hash(identity["source"]),
            "frozen_sha256": frozen_digest,
            "frozen_policy": frozen["policy"],
            "data_manifests": scopes,
            "original_generated_report_sha256": sha256(run / "report" / "technical_report.md"),
            "source_scope": "historical_source matches the measured run; current source may include later portability changes",
            "environment_redaction": "The executable path is relative in the public copy; environment_hash identifies the original unredacted record.",
            "excluded_artifacts": "Generated arrays, feature banks, models, per-example predictions and TensorBoard events remain local. Their identities are retained.",
        },
    )
    n = config["data"]["train_size"]
    nearby = sorted(sorted(config["features"]["counts"], key=lambda p: abs(p - n))[:3])
    with (destination / "tables" / "singular_spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "n_features", "singular_index", "singular_value"])
        for count in nearby:
            for seed in config["features"]["seeds"]:
                values = saved_singular_values(
                    run
                    / "trials"
                    / f"seed_{seed}"
                    / f"features_{count:04d}"
                    / "singular_values.pt",
                    min(n, count),
                )
                writer.writerows(
                    (seed, count, index, repr(float(value)))
                    for index, value in enumerate(values, 1)
                )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#C8D0D6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 13,
            "text.color": INK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "grid.color": "#D9E0E5",
            "grid.alpha": 0.6,
            "grid.linewidth": 0.65,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "svg.fonttype": "none",
            "svg.hashsalt": "rff-expanded-run-001",
        }
    )


def finish(figure, destination: Path, name: str, title: str, subtitle: str, footer: str) -> None:
    figure.suptitle(title, x=0.065, y=0.98, ha="left", fontsize=18, fontweight="bold")
    figure.text(0.065, 0.915, subtitle, fontsize=10.5, color=GRAY)
    figure.text(0.065, 0.016, footer, fontsize=8.5, color=GRAY)
    figure.tight_layout(rect=(0.025, 0.052, 0.995, 0.88), h_pad=2.6, w_pad=3.0)
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        figure.savefig(destination / "figures" / f"{name}.{extension}", dpi=180, **kwargs)
    plt.close(figure)


def capacity_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(
        [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8], ["1/16", "1/8", "1/4", "1/2", "1", "2", "4", "8"]
    )
    ax.set_xlim(0.058, 8.7)
    ax.axvline(1, color=GRAY, linestyle=(0, (2, 3)), linewidth=1, zorder=1)
    ax.set_xlabel("Feature / sample ratio  p/n")
    ax.grid(axis="y", which="major")


def seed_curve(ax, frame: pd.DataFrame, metric: str, color: str, label: str, linestyle="-") -> None:
    for _, group in frame.groupby("seed"):
        group = group.sort_values("n_features")
        ax.plot(group.ratio, group[metric], color=color, alpha=0.19, linewidth=0.85, zorder=2)
    average = frame.groupby("ratio", sort=True)[metric].mean()
    ax.plot(
        average.index,
        average,
        color=color,
        linewidth=2.4,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )


def positive_log(ax, values, *, floor: float | None = None) -> None:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Nonnegative finite plotting values required; inspect failed conditions")
    if (values == 0).any():
        threshold = floor if floor is not None else min(values[values > 0], default=1e-12) / 10
        ax.set_yscale("symlog", linthresh=threshold)
        ax.set_ylim(0, max(values) * 2)
    else:
        ax.set_yscale("log")
        ax.set_ylim(min(values) / 2, max(values) * 2)


def redraw_figures(destination: Path) -> None:
    configure_style()
    primary = pd.read_csv(destination / "tables" / "primary.csv")
    paired = pd.read_csv(destination / "tables" / "paired.csv")
    sensitivity = pd.read_csv(destination / "tables" / "sensitivity.csv")
    spectra = pd.read_csv(destination / "tables" / "singular_spectra.csv")
    units = pd.read_csv(destination / "tables" / "units.csv")
    config = read_json(destination / "config.json")
    n = config["data"]["train_size"]
    seeds = len(config["features"]["seeds"])
    units["ratio"] = units.n_features / n
    footer = f"Thin lines: {seeds} feature seeds · thick lines: arithmetic mean · sample SD in tables · one fixed dataset; no confidence interval"
    zero = next(
        x["signal_mse"]
        for x in read_json(destination / "tables" / "baselines.json")
        if x["predictor"] == "zero"
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9))
    for row, noise in enumerate(config["data"]["noise_stds"]):
        group = primary[primary.noise_std.eq(noise)]
        for column, metric in enumerate(("train_observed_mse", "test_signal_mse")):
            ax = axes[row, column]
            for solver, color in COLORS.items():
                seed_curve(ax, group[group.solver.eq(solver)], metric, color, LABELS[solver])
            capacity_axis(ax)
            positive_log(ax, group[metric])
            ax.set_title(
                f"{'Training labels' if column == 0 else 'Held-out clean signal'} · σ = {noise:g}"
            )
            ax.set_ylabel("Mean squared error · log scale")
            if column == 1:
                ax.axhline(
                    zero, color=GRAY, linestyle="--", linewidth=1, label="Zero predictor (measured)"
                )
            ax.legend(frameon=False, fontsize=8.5, loc="best")
    finish(
        fig,
        destination,
        "01_error_curves",
        "Near-zero training error can coexist with extreme test error",
        "The square design lies at p/n = 1; all extreme seed outcomes remain visible.",
        footer,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    for ax, solver in zip(axes, COLORS, strict=True):
        group = paired[paired.solver.eq(solver)]
        seed_curve(ax, group, "test_signal_mse_difference", COLORS[solver], LABELS[solver])
        capacity_axis(ax)
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.set_yscale("symlog", linthresh=0.001)
        differences = group.test_signal_mse_difference
        ax.set_ylim(min(0, differences.min() * 2), max(0.001, differences.max() * 2))
        ax.set_ylabel("Noisy minus clean test MSE · symmetric log")
        ax.set_title(LABELS[solver])
        ax.legend(frameon=False)
    finish(
        fig,
        destination,
        "02_noise_differences",
        "Noise changes the cost of interpolation",
        "Paired differences use the same dataset and feature bank. Negative values, where present, are retained.",
        footer,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    for ax, noise in zip(axes, config["data"]["noise_stds"], strict=True):
        group = primary[primary.noise_std.eq(noise)]
        for solver, color in COLORS.items():
            seed_curve(ax, group[group.solver.eq(solver)], "test_signal_mse", color, LABELS[solver])
        capacity_axis(ax)
        positive_log(ax, group.test_signal_mse)
        ax.axhline(zero, color=GRAY, linestyle="--", linewidth=1, label="Zero predictor (measured)")
        ax.set_ylabel("Clean-target test MSE · log scale")
        ax.set_title(f"{'Noise-free labels' if noise == 0 else 'Noisy labels'} · σ = {noise:g}")
        ax.legend(frameon=False, fontsize=9)
    finish(
        fig,
        destination,
        "03_solver_comparison",
        "A second descent does not guarantee useful predictions",
        "At p/n = 8 with noise: minimum norm 5.004; ridge 0.0301; zero predictor 0.987.",
        footer,
    )

    minimum = primary[primary.solver.eq("minimum_norm")]
    reference = minimum[minimum.noise_std.eq(0)]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    seed_curve(axes[0], reference, "numerical_rank", BLUE, "Measured rank")
    capacity_axis(axes[0])
    axes[0].set_ylabel("Retained singular values")
    axes[0].set_ylim(0, n * 1.06)
    axes[0].axhline(n, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_title("Rank saturates at the sample count")
    for noise, color, style in [(0, BLUE, "-"), (0.3, ORANGE, "--")]:
        group = minimum[minimum.noise_std.eq(noise)].copy()
        group["fit_fraction"] = group.interpolates.astype(float)
        seed_curve(axes[1], group, "fit_fraction", color, f"σ = {noise:g}", style)
    capacity_axis(axes[1])
    axes[1].set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    axes[1].set_ylim(-0.035, 1.06)
    axes[1].set_ylabel("Seeds meeting the residual threshold")
    axes[1].set_title("All seeds first interpolate at p = 1,024")
    axes[1].legend(frameon=False)
    finish(
        fig,
        destination,
        "04_rank_and_interpolation",
        "The fitting transition is measured, not assumed",
        "Interpolation: ||Φβ − y||₂ / max(||y||₂, 1) ≤ 10⁻⁸; primary singular-value cutoff 10⁻¹².",
        footer,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    seed_curve(axes[0], reference, "condition_number", BLUE, "Design condition number")
    capacity_axis(axes[0])
    positive_log(axes[0], reference.condition_number)
    axes[0].set_ylabel("Largest / smallest singular value · log scale")
    axes[0].set_title("Weak directions near the square design")
    for solver, color in COLORS.items():
        for noise, style in [(0, "--"), (0.3, "-")]:
            group = primary[primary.solver.eq(solver) & primary.noise_std.eq(noise)]
            seed_curve(
                axes[1], group, "coefficient_norm", color, f"{LABELS[solver]}, σ = {noise:g}", style
            )
    capacity_axis(axes[1])
    positive_log(axes[1], primary.coefficient_norm)
    axes[1].set_ylabel("Coefficient L2 norm · log scale")
    axes[1].set_title("Ridge restrains coefficient growth")
    axes[1].legend(frameon=False, fontsize=8.5)
    finish(
        fig,
        destination,
        "05_conditioning_and_coefficients",
        "Small singular values expose a noise-amplification mechanism",
        "Conditioning alone does not determine test error; target alignment and prediction geometry also matter.",
        footer,
    )

    counts = sorted(spectra.n_features.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.5))
    limits = [spectra.singular_value.min() / 3, spectra.singular_value.max() * 2]
    for ax, count in zip(axes, counts, strict=True):
        group = spectra[spectra.n_features.eq(count)]
        for _, seed in group.groupby("seed"):
            ax.plot(seed.singular_index, seed.singular_value, color=BLUE, alpha=0.19, linewidth=0.9)
        mean = group.groupby("singular_index").singular_value.mean()
        ax.plot(mean.index, mean, color=BLUE, linewidth=2.2)
        ax.set_yscale("log")
        ax.set_ylim(*limits)
        ax.set_xlabel("Singular-value index (descending)")
        ax.set_ylabel("Singular value · log scale")
        ax.set_title(f"p = {count:,}")
        ax.grid(axis="y")
    finish(
        fig,
        destination,
        "06_singular_spectra",
        "The smallest singular directions are the fragile ones",
        "Saved spectra for all seeds immediately below, at and above n = 1,024; identical y-axis limits.",
        footer,
    )

    columns = ["seed", "n_features", "noise_std", "test_signal_mse", "numerical_rank"]
    compare = sensitivity.merge(
        minimum[columns],
        on=columns[:3],
        suffixes=("_diagnostic", "_primary"),
        validate="many_to_one",
    )
    compare["mse_difference"] = compare.test_signal_mse_diagnostic - compare.test_signal_mse_primary
    compare["rank_difference"] = compare.numerical_rank_diagnostic - compare.numerical_rank_primary
    compare["ratio"] = compare.n_features / n
    compare.to_csv(destination / "tables" / "cutoff_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    for ax, noise in zip(axes, config["data"]["noise_stds"], strict=True):
        for solver, color, style in [("cutoff_1e-10", TEAL, "-"), ("cutoff_1e-14", ORANGE, "--")]:
            group = compare[compare.noise_std.eq(noise) & compare.solver.eq(solver)]
            seed_curve(
                ax, group, "mse_difference", color, solver.replace("cutoff_", "Cutoff "), style
            )
        capacity_axis(ax)
        ax.set_ylim(-1e-12, 1e-12)
        ax.set_yticks([-1e-12, 0, 1e-12])
        ax.set_ylabel("Diagnostic minus primary test MSE")
        ax.set_title(f"σ = {noise:g}: all saved differences are zero")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    finish(
        fig,
        destination,
        "07_cutoff_sensitivity",
        "These two cutoff checks leave the saved result unchanged",
        "All 840 diagnostic comparisons: identical rank and test MSE at cutoffs 10⁻¹⁰ / 10⁻¹⁴ versus 10⁻¹².",
        footer,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.5))
    seed_curve(axes[0], units, "total_seconds", BLUE, "Fit-unit duration")
    capacity_axis(axes[0])
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Seconds per shared-SVD fit unit")
    axes[0].set_title("Measured fitting cost")
    complete = primary.groupby("n_features").status.apply(lambda s: (s == "complete").sum())
    axes[1].bar(np.arange(len(complete)), complete, color=TEAL, width=0.72)
    axes[1].set_xticks(
        np.arange(len(complete)),
        [f"{x:,}" for x in complete.index],
        rotation=65,
        ha="right",
        fontsize=8,
    )
    axes[1].set_ylim(0, complete.max() * 1.12)
    axes[1].set_yticks([0, 10, 20, 30, 40])
    axes[1].set_ylabel("Complete primary predictors")
    axes[1].set_xlabel("Feature count")
    axes[1].set_title("40 / 40 complete at every capacity")
    axes[1].grid(axis="y")
    axes[1].set_axisbelow(True)
    finish(
        fig,
        destination,
        "08_runtime_and_status",
        "The full registered sweep completed within its budget",
        "210 SVD units · 840 primary predictors · zero failures · active-time ledger 124.65 s / 1,800 s.",
        footer,
    )
    write_json(
        destination / "figures" / "figure_metadata.json",
        {
            "origin": "redrawn from the public saved-measurement tables, without fitting or evaluation",
            "dispersion": "individual feature-seed lines and arithmetic mean; exact sample SD is retained in aggregate.csv",
            "count": seeds,
            "uncertainty_scope": "conditional on one fixed dataset/noise realization; not a confidence interval",
            "mse_axes": "positive logarithmic ranges; no mean-minus-SD ribbon or suppressed extreme seed",
            "signed_axes": "paired differences use symmetric log; cutoff differences retain exact zeros",
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "presentation_script_sha256": sha256(Path(__file__)),
        },
    )


def summarize_public_tables(destination: Path) -> None:
    """Compute descriptive summaries from CSV rows, never from model inference."""
    frame = pd.read_csv(destination / "tables" / "primary.csv")
    minimum = frame[frame.solver.eq("minimum_norm")]
    config = read_json(destination / "config.json")
    n = config["data"]["train_size"]
    peak = minimum[minimum.n_features.eq(n) & minimum.noise_std.eq(0.3)]
    noisy_values = peak.test_signal_mse.sort_values()
    peak[
        ["seed", "test_signal_mse", "condition_number", "coefficient_norm", "relative_residual"]
    ].to_csv(destination / "tables" / "noisy_boundary_seeds.csv", index=False)
    interpolation = []
    for (seed, noise), group in minimum.groupby(["seed", "noise_std"]):
        fitted = group[group.interpolates].sort_values("n_features")
        interpolation.append(
            {
                "seed": int(seed),
                "noise_std": float(noise),
                "first_sampled_fitting_capacity": int(fitted.n_features.min()),
                "fitting_capacities": ";".join(str(int(p)) for p in fitted.n_features),
                "maximum_test_error_capacity": int(
                    group.loc[group.test_signal_mse.idxmax(), "n_features"]
                ),
            }
        )
    pd.DataFrame(interpolation).to_csv(destination / "tables" / "interpolation.csv", index=False)
    grouped = frame.groupby(["n_features", "noise_std", "solver"]).test_signal_mse
    summary = grouped.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    landmarks = summary[summary.n_features.isin([64, 1024, 8192])]
    landmarks.to_csv(destination / "tables" / "landmarks.csv", index=False)
    compare = pd.read_csv(destination / "tables" / "cutoff_comparison.csv")
    claims = {
        "noisy_boundary_mean": float(noisy_values.mean()),
        "noisy_boundary_median": float(noisy_values.median()),
        "noisy_boundary_sample_sd": float(noisy_values.std(ddof=1)),
        "noisy_boundary_top_two_share": float(noisy_values.iloc[-2:].sum() / noisy_values.sum()),
        "maximum_condition_number": float(frame.condition_number.max()),
        "minimum_relative_singular_value": float((frame.singular_min / frame.singular_max).min()),
        "maximum_interpolating_relative_residual": float(
            minimum[minimum.interpolates].relative_residual.max()
        ),
        "cutoff_maximum_absolute_test_mse_difference": float(compare.mse_difference.abs().max()),
        "cutoff_maximum_absolute_rank_difference": int(compare.rank_difference.abs().max()),
        "all_measured_ranks_equal_minimum_of_rows_and_columns": bool(
            (frame.numerical_rank == np.minimum(n, frame.n_features)).all()
        ),
        "interpretation": "Descriptive saved-run summaries; post-run landmarks and medians do not select or refit predictors.",
    }
    write_json(destination / "tables" / "descriptive_claims.json", claims)


def write_manifest(destination: Path) -> None:
    write_json(
        destination / "manifest.json",
        {
            "scope": "Every public result artifact except this manifest; copied historical evidence and derived presentation are distinct.",
            "files": {
                path.relative_to(destination).as_posix(): sha256(path)
                for path in sorted(destination.rglob("*"))
                if path.is_file() and path != destination / "manifest.json"
            },
        },
    )


def build_results(root: Path = ROOT) -> Path:
    destination = root / "results" / "run_001"
    if (destination / "tables" / "primary.csv").exists():
        verify_files(destination, read_json(destination / "manifest.json")["files"])
    elif (root / "runs" / "run_001").exists():
        export_saved_run(root, destination)
    else:
        raise FileNotFoundError("Neither local run_001 nor its public measurement package exists")
    redraw_figures(destination)
    summarize_public_tables(destination)
    write_manifest(destination)
    return destination


if __name__ == "__main__":
    print(build_results())
