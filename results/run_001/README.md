# Completed expanded run: public evidence

**840 primary predictors · 210 SVD units · 10 feature seeds · zero failures**

Read the [technical report](technical_report.md). The central result is a sharp interpolation peak followed by decreasing noisy minimum-norm error that still exceeds the measured zero-predictor baseline at the largest capacity. Fixed ridge changes that comparison; noise-free labels show a different endpoint tradeoff.

![Primary solver comparison with individual seeds and their means.](figures/03_solver_comparison.png)

| Artifact | Purpose |
| --- | --- |
| [config.json](config.json) | Exact expanded protocol executed in this run |
| [primary.csv](tables/primary.csv) | 840 original train/validation/test result rows |
| [aggregate.csv](tables/aggregate.csv) | Original means, sample SDs and contributing counts |
| [paired.csv](tables/paired.csv) | Matched noisy-versus-clean differences |
| [sensitivity.csv](tables/sensitivity.csv) | 840 original cutoff-diagnostic rows |
| [units.csv](tables/units.csv) | 210 fit-unit durations and statuses |
| [baselines.json](tables/baselines.json) | Measured baselines and separate population expectations |
| [singular_spectra.csv](tables/singular_spectra.csv) | Saved spectra at 1,023 / 1,024 / 1,025 features |
| [landmarks.csv](tables/landmarks.csv) | Derived descriptive endpoint/boundary summaries |
| [noisy_boundary_seeds.csv](tables/noisy_boundary_seeds.csv) | Every seed's noisy square-design result |
| [interpolation.csv](tables/interpolation.csv) | First fitting capacities and per-seed peak locations |
| [cutoff_comparison.csv](tables/cutoff_comparison.csv) | Matched diagnostic-minus-primary errors and ranks |
| [descriptive_claims.json](tables/descriptive_claims.json) | Machine-readable calculations used in the report |
| [Figure metadata](figures/figure_metadata.json) | Display policy and plotting-library versions |
| [Identity record](provenance/identity.json) | Configuration, frozen evaluation, data and historical source hashes |
| [Artifact identities](provenance/unit_artifact_hashes.json) | Model, prediction and manifest hashes without large tensors |
| [Historical source](historical_source/README.md) | Exact code and original lock that generated the run |
| [Historical test output](provenance/runtime_checks.txt) | 61 passing tests from the recorded execution |
| [Budget ledger](provenance/budget.json) | 124.649 seconds active time; no extension |
| [manifest.json](manifest.json) | SHA-256 for every bundled file except the manifest itself |

The original run's CSVs, configuration and historical source are copied exactly. The human-written report and eight PNG/SVG figure pairs are a later presentation of the saved evidence. Individual seeds and means are shown; sample SD stays in the tables rather than forming negative error ribbons. All extremes are retained.

Generated arrays, raw model checkpoints, per-example prediction tensors and TensorBoard events remain in the ignored local run. The package includes their identities, not their bytes. It is compact evidence for review; a fresh notebook execution is needed to reconstruct the complete experiment from a public checkout.

Run `.venv/bin/python -B tools/build_results.py` from the repository root to redraw. It verifies and uses this public package even if a different local `run_001` exists. Only when the package is absent does it verify and export the recorded original local run. No data generation, fitting, evaluation, notebook execution or test invocation occurs. The utility accepts no experiment arguments. Local font cache goes under ignored `reviews/plot-cache/`.

The original environment contains 123 packages. Its public record redacts only the executable's machine-specific prefix; the stored original environment hash therefore refers to the local unredacted record. The main repository may contain later portability changes; the historical snapshot and its test receipt remain distinct.
