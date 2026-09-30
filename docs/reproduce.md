# Run and reproduce the study

## Read the saved results or run a new experiment

The [notebook](../notebooks/random_feature_double_descent.ipynb) contains the completed protocol 1.1 and its original outputs. The [public report](../results/run_001/technical_report.md) and [curated evidence](../results/run_001/README.md) can be read without installing Python. Models and raw arrays remain in local ignored run directories, so opening the notebook alone does not restore its variables.

To reproduce the workflow, start a fresh run from the top of the notebook. Its 1,024/4,096/32,768 partition sizes, ten feature seeds and 21 capacities are explicit. Inspect the [registered protocol](expanded-protocol.md) before changing them.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Open a terminal in the repository folder. Native CPython 3.13 is recommended; the measured study used 3.13.11. The application does not require a specific Python patch version or environment name to create a fresh run.

macOS/Linux:

```bash
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python --torch-backend cpu --require-hashes requirements.txt
.venv/bin/python -m jupyterlab notebooks/random_feature_double_descent.ipynb
```

Windows PowerShell:

```powershell
uv venv --python 3.13 .venv
uv pip sync --python .venv\Scripts\python.exe --torch-backend cpu --require-hashes requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab notebooks\random_feature_double_descent.ipynb
```

Reuse an existing environment if preferred: replace the executable path with that environment's Python. The current hashed lock was resolved across platforms using CPU PyTorch wheels and retains the recorded macOS package versions. This avoids downloading the CUDA dependency stack for a CPU study. Package availability still depends on a supported OS/architecture; this preparation has not executed the experiment on Windows, Linux or Intel Macs.

### Select a kernel

Choose the installed environment's Python kernel in Jupyter or VS Code. The name saved in the notebook is a convenience, not an application requirement. If the name is unavailable, select a different Python kernel backed by the installed environment.

If using a Jupyter server from another environment, register a kernel once with the study environment's Python:

```bash
python -m ipykernel install --user --name rff-study --display-name "Random-feature study"
```

Both names are editable. Do not register a kernel if your editor already lets you select the interpreter directly.

## Notebook sequence

1. Import modules and discover the checkout root.
2. Inspect the separate data, feature, solver and execution configuration cells.
3. Leave `RUN_NAME = None` and create a fresh numbered run.
4. Inspect the environment and prepare only training/validation data.
5. Execute the explicit correctness-check cell, then the timing preflight.
6. Fit/resume all registered units and inspect completion.
7. Freeze fitted-model identities before generating test arrays.
8. Evaluate every frozen predictor; aggregate the results.
9. Generate each figure and the automatic report in separate cells.
10. Load a predictor and demonstrate prediction. TensorBoard is optional.

The preflight performs real registered fits at 256, 1,024 and 8,192 features for the first seed; the sweep reuses them. A projected total beyond the active-time budget blocks progress. A numerical failure remains visible; an incomplete sweep cannot advance to test evaluation. A missing peak is not a reason to change the sweep after seeing results.

The synthetic data need no external download. Test arrays use a distinct seed and are created only after freezing. CPU float64 is a scientific choice: moving the solver to MPS or changing precision would define a different numerical experiment.

## Recovery and time budget

Record the actual allocated run name. Set `RUN_NAME` to it for explicit resume, with the same source and configuration. Every completed unit is verified before reuse; interrupted units repeat. Do not rerun the fresh-run cell unintentionally, because it allocates another directory.

Fresh runs bind the active machine's own environment. The current compatibility policy ignores the interpreter's installation path and unrelated package versions. It protects Python version, platform/architecture, numerical libraries, backend/build, precision and thread settings. Full environment information remains saved for provenance. Moving to another scientific environment is a fresh experiment, not an automatic continuation of another person's run.

Native file locks exclude concurrent writers on supported platforms. Do not delete a lock file to bypass an active process. Completed historical runs retain their original source snapshots and strict identities; the new portability source does not rewrite or bypass those identities. You can inspect the exported historical measurements without resuming anything.

The active-time allowance is 1,800 seconds. Notebook idle time and installation are excluded. The ledger is refreshed during active work, and checks occur between units. An in-progress operation may finish beyond the limit. No exact wall-clock limit is promised. Any intentional extension uses `extend_budget(run, extra_seconds=..., reason=...)` and records the decision separately.

## Troubleshooting

| Symptom | Action |
|---|---|
| Saved kernel name is missing | Select the environment's Python kernel, or register it using any name above. |
| `No module named torch` or another dependency | Install into the interpreter selected by the notebook; a separate terminal's environment may differ. |
| Checkout root not found | Launch Jupyter inside this project or its `notebooks/` directory. Root discovery uses file markers, not a prescribed folder name. |
| Source/configuration mismatch on resume | Use the matching original source/configuration or start a fresh run. Never overwrite stored hashes. |
| Environment mismatch on resume | Restore the recorded numerical environment or start fresh; renaming an environment alone is permitted for new-policy runs. |
| Preflight or budget stops | Inspect the timing receipt; explicitly approve and record a budget extension if wanted. No automatic sweep reduction occurs. |
| Very different results on another machine | Compare exact configuration, library versions, BLAS/backend and recorded numerical diagnostics; seeds do not ensure bitwise equality across systems. |

## Validation and chart reproduction

The original run recorded 61 passing tests before fitting. The current portability revision separately passes 79 regression tests on macOS. Hosted CI and complete Windows/Linux execution remain unobserved. Current tests can be run with `python -m pytest -q tests` using the selected environment.

`python tools/check_repository.py` checks source syntax, notebook source, local documentation links and public result checksums without executing the notebook. Saved outputs are allowed. Ruff provides separate lint/format checks.

The public figures can be regenerated from the included measurements with:

```bash
python -B tools/build_results.py
```

This utility verifies and redraws the bundled public tables, even if a different local `run_001` exists. Only when the public package is absent does it verify and export the recorded original local run. It does not train models, rerun test evaluation or change the notebook. See the [public result package](../results/run_001/README.md) for the exact included artifacts and historical source identity.
