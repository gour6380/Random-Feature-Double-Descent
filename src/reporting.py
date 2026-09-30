"""Evidence-led Markdown reporting from committed experiment artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .analysis import _protected, _read, _table, _write_text
from .io import atomic_json, sha256_file

if TYPE_CHECKING:
    from .runs import Run

FIGURES = (
    ("01_error_curves", "Training fit and clean-target test error"),
    ("02_noise_differences", "Paired effects of observation noise"),
    ("03_solver_comparison", "Minimum norm and fixed ridge"),
    ("04_rank_and_interpolation", "Numerical rank and measured interpolation"),
    ("05_conditioning_and_coefficients", "Conditioning and coefficient size"),
    ("06_singular_spectra", "Singular spectra near p/n = 1"),
    ("07_cutoff_sensitivity", "Numerical-cutoff diagnostics"),
    ("08_runtime_and_status", "Runtime and completion outcomes"),
)


def _number(value) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, float)):
        if math.isinf(value):
            return "∞" if value > 0 else "−∞"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No measurements available."
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(_number(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


@_protected
def generate_report(run: Run) -> Path:
    """Require all eight saved figures and complete identity-grid evaluation."""
    primary = _table(run, "primary")
    summary = _table(run, "aggregate")
    manifest = _read(run.path / "tables" / "summary.json")
    seed_count = len(run.config.features.seeds)
    expected = seed_count * len(run.config.features.counts) * len(run.config.data.noise_stds) * 2
    if len(primary) != expected or primary["test_count"].isna().any():
        failed = primary.status.ne("complete")
        if len(primary) != expected or primary.loc[~failed, "test_count"].isna().any():
            raise RuntimeError("Complete frozen evaluation and aggregate_results before reporting.")
    for name, _ in FIGURES:
        metadata = run.path / "figures" / f"{name}.json"
        if not metadata.exists() or _read(metadata)["table_hashes"] != _read(
            run.path / "tables" / "table_hashes.json"
        ):
            raise RuntimeError(f"Regenerate the {name} figure from current aggregate tables.")
        for extension in ("png", "svg"):
            if not (run.path / "figures" / f"{name}.{extension}").is_file():
                raise RuntimeError(f"Generate the missing {name} figure before reporting.")
    selected_counts = sorted(
        {
            min(run.config.features.counts),
            max(run.config.features.counts),
            min(run.config.features.counts, key=lambda p: abs(p - run.config.data.train_size)),
        }
    )
    landmarks = summary[summary.n_features.isin(selected_counts)][
        [
            "n_features",
            "noise_std",
            "solver",
            "test_signal_mse_mean",
            "test_signal_mse_std",
            "test_signal_mse_count",
            "failed_count",
        ]
    ].rename(
        columns={
            "n_features": "Features",
            "noise_std": "Noise σ",
            "solver": "Solver",
            "test_signal_mse_mean": "Clean-test MSE mean",
            "test_signal_mse_std": "Sample SD",
            "test_signal_mse_count": "Contributing seeds",
            "failed_count": "Failed seeds",
        }
    )
    fitting_rows = []
    for (seed, noise), group in primary[primary.solver.eq("minimum_norm")].groupby(
        ["seed", "noise_std"]
    ):
        fitted = group[group.status.eq("complete") & group.interpolates.eq(True)]
        fitting_rows.append(
            {
                "Seed": seed,
                "Noise σ": noise,
                "First sampled fitting capacity": fitted.n_features.min()
                if len(fitted)
                else "none",
                "Sampled fitting capacities": ", ".join(
                    str(int(p)) for p in sorted(fitted.n_features)
                )
                or "none",
            }
        )
    baselines = pd.DataFrame(_read(run.path / "evaluation" / "baselines.json"))
    failed = primary[primary.status.ne("complete")]
    budget = run.budget_status()
    units = _table(run, "units")
    observed_seconds = pd.to_numeric(units.total_seconds, errors="coerce").sum()
    figure_markdown = "\n\n".join(
        f"### {title}\n\n![{title}](../figures/{name}.png)\n\n[Vector version](../figures/{name}.svg)"
        for name, title in FIGURES
    )
    failure_text = (
        "Every registered primary predictor has a complete numerical result."
        if failed.empty
        else f"{len(failed)} of {expected} registered predictors failed numerically. "
        "These rows remain in the primary table; means include only measured successful "
        "values and expose their contributing counts. Results with fewer contributing seeds "
        f"must not be presented as complete {seed_count}-seed evidence."
    )
    text = f"""# Random-Feature Double Descent — {run.path.name}

## Evidence and protocol

This report was generated from saved measurements and immutable fitted-model identities.
It describes one fixed synthetic dataset/noise realization and a registered random-feature
capacity sweep. It is a paper-inspired adaptation, not an exact reproduction.

The run contains **{manifest["complete_predictors"]} complete primary predictors** and
**{manifest["failed_predictors"]} failed primary predictors**, from **{expected} registered
predictors**. {failure_text}

The model uses fixed Gaussian random Fourier features and fits only the linear output
coefficients. Inputs have {run.config.data.input_dim} independent standard-normal coordinates;
partition sizes are {run.config.data.train_size} / {run.config.data.validation_size} /
{run.config.data.test_size}. Noise standard deviations are {run.config.data.noise_stds}.
Feature seeds are {run.config.features.seeds}; capacities are {run.config.features.counts}.
The normalized sinusoidal teacher has population mean zero and variance one.

CPU float64 and {run.config.execution.threads} compute threads are registered. Minimum-norm
fitting discards singular values at or below relative cutoff {run.config.solver.rcond:g}.
The ridge objective is mean squared training residual plus
{run.config.solver.ridge_lambda:g} times coefficient squared norm. Each SVD supports both
noise conditions and solvers. Numerical interpolation uses relative residual threshold
{run.config.solver.interpolation_tolerance:g}. There are no epochs or optimizer updates.

All fitted-model identities were frozen before test generation. No test-based selection is
performed. See [resolved configuration](../config/experiment.json),
[frozen identities](../evaluation/frozen.json), the repository's `docs/protocol.md`, and
`docs/expanded-protocol.md` for the larger follow-up.

## Observations

At three predeclared landmarks—the smallest capacity, the capacity nearest the square-matrix
boundary and the largest capacity—the measured clean-test errors are:

{_markdown_table(landmarks)}

These landmarks are descriptive, not selected optima. The complete capacity grid is retained
in [primary measurements](../tables/primary.csv) and [aggregate measurements](../tables/aggregate.csv).
Sample SD is computed with denominator `count - 1`; it is absent when fewer than two seeds
contribute. Each table explicitly reports contributing and failed counts.

Measured minimum-norm fitting across the sampled grid is:

{_markdown_table(pd.DataFrame(fitting_rows))}

A square design at `p/n=1` is a geometric reference. The table above uses actual residuals;
it does not assume interpolation merely from parameter count. Reaching a fitting threshold
at one sampled capacity does not replace measurements at later capacities.

Finite-test-set baselines and their separately labelled population expectations are:

{_markdown_table(baselines)}

The training-mean predictor is fitted from training labels only. Zero prediction has
population clean-signal MSE 1; the exact teacher has clean-signal MSE 0. For noise σ=0.3,
the population irreducible observed-target MSE is 0.09. The finite test measurements may differ.

{len(units)} seed/capacity units have saved status records. Their summed recorded unit durations
are {observed_seconds:.3f} seconds. The active-time ledger reads {budget["used_seconds"]:.3f}
seconds used of {budget["limit_seconds"]:.3f} seconds at report generation; it excludes
notebook idle time and environment installation. This report's remaining write time is charged
after the cell completes. [Budget ledger](../logs/budget.json) is authoritative if a
later operation or extension changes it. These timings describe this run only.

## Figures

{figure_markdown}

## Interpretation and limitations

Relate any nonmonotonic error pattern to the measured interpolation, singular spectrum,
conditioning and coefficient-size figures before attributing it to an interpolation
transition. No automatic “double descent proven” classification is applied. If the sweep
never crosses measured fitting, interpolation-related behavior remains inconclusive. If no
peak appears, that absence is a result limited to this registered protocol.

Ridge changes the estimator through one fixed penalty; it was not tuned using test outcomes.
Cutoff sensitivities are labelled diagnostics, and the primary cutoff remains unchanged.
Large errors, numerical failures and exact zero values are retained. Infinite condition
numbers are explicitly marked in figures and retained in tables. Symmetric-log axes preserve
zeros and negative paired differences instead of dropping them for a logarithmic axis.

Feature-seed variation is conditional on one dataset and noise realization. Sample SD across
the {seed_count} registered feature seeds is not a confidence interval over all datasets.
The teacher, Fourier bandwidth, sample
size and capacity grid constrain the scope of every conclusion. Nested feature prefixes
improve pairing but make adjacent capacity measurements correlated. This study provides no
evidence about learned neural feature representations or CNN training behavior.

The implementation uses a direct SVD rather than iterative optimization. Operationally,
numerical conditioning and rank deserve separate inspection from predictive error: a small
training residual alone does not establish stable coefficients or low test risk.

## Reproduction and evidence links

Use native Python 3.13.11, the repository's hashed requirements and the registered notebook
kernel. Open `notebooks/random_feature_double_descent.ipynb` from the checkout. Match
this run's resolved configuration, execute correctness checks, then preflight, fitting, freezing, test
generation, evaluation, aggregation and each separate figure cell. The saved source snapshot
and environment identify this run. Changed source/configuration requires a fresh run.

- [Primary rows](../tables/primary.csv), [aggregate rows](../tables/aggregate.csv)
- [Paired noise differences](../tables/paired.csv), [paired aggregates](../tables/paired_aggregate.csv)
- [Cutoff diagnostics](../tables/sensitivity.csv), [unit status and duration](../tables/units.csv)
- [Summary counts](../tables/summary.json)
- Per-example predictions and sample identities are stored in each committed evaluation unit.
- [Belkin et al.](https://arxiv.org/abs/1812.11118)
- [Rahimi and Recht](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
- [Hastie et al.](https://arxiv.org/abs/1903.08560)
"""
    destination = run.path / "report" / "technical_report.md"
    _write_text(text, destination)
    paths = sorted((run.path / "tables").glob("*")) + sorted((run.path / "figures").glob("*"))
    paths += [destination]
    atomic_json(
        run.path / "report" / "artifact_hashes.json",
        {
            "files": {
                path.relative_to(run.path).as_posix(): sha256_file(path)
                for path in paths
                if path.is_file()
            },
            "scope": "derived tables, figures and report; fitted/data identities are frozen separately",
        },
    )
    return destination
