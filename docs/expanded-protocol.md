# Registered random-feature study — protocol 1.1

**Completed:** 210 SVD units, 840 primary predictors, no recorded failures. Read the [technical report](../results/run_001/technical_report.md) for measured outcomes and limitations. Execute the [notebook](../notebooks/random_feature_double_descent.ipynb) to create a new run; its retained outputs document the original execution.

## Question

How do prediction error, numerical conditioning and coefficient size change as random-feature capacity crosses the training-sample count, and how do noise and regularization affect that relationship?

This is a paper-inspired adaptation using fixed random Fourier features. Fitting solves only their linear output coefficients. A missing double-descent shape is a valid outcome; no test-based model selection or sweep expansion is permitted to obtain a preferred curve.

## Fixed data and feature design

| Setting | Value |
|---|---|
| Inputs | Eight independent standard-normal coordinates |
| Training / validation / test examples | 1,024 / 4,096 / 32,768 |
| Input seeds, in partition order | 31001 / 31002 / 31003 |
| Observation-noise seeds | 32001 / 32002 / 32003 |
| Noise standard deviations | 0 and 0.3 |
| Feature seeds | 2026–2035 |
| Frequency / phase generators | Independent local generators; phase seed = feature seed + 10000 |
| Feature bank | 8,192 frequencies/phases per feature seed; nested prefixes |
| Gaussian-kernel lengthscale | √8 |

The clean teacher is `(sin(x1) + 0.5 sin(x2) + 0.25 sin(x3)) / sqrt((1-exp(-2))/2 * 1.3125)`. It has population mean zero and variance one under the registered input distribution. The observed labels are `teacher(x) + sigma * epsilon`, with independent standard-normal noise. Inputs, clean targets and noise realizations are paired across capacities, solvers and feature seeds.

Features are `sqrt(2/p) * cos(x @ W[:p].T + b[:p])`, with each frequency drawn from `N(0, I8/8)` and each phase uniform on `[0, 2π)`. There is no learned embedding, intercept, bandwidth search, normalization or feature standardization.

Registered feature counts:

```text
64, 128, 256, 384, 512, 768, 896, 960, 992,
1023, 1024, 1025, 1056, 1088, 1152, 1280, 1536,
2048, 3072, 4096, 8192
```

Twenty-one counts times ten seeds give **210 reduced SVDs**. Each supports two noise conditions and two primary solvers, giving **840 primary predictors**. Two additional cutoff values provide 840 diagnostic variants using the same decompositions.

## Solvers and measurements

Use CPU float64 and `torch.linalg.svd(..., full_matrices=False)` on the training design matrix. Avoid explicit matrix inversion and normal equations.

- Minimum norm retains singular values strictly above `1e-12 * largest_singular_value` and inverts only those values.
- Ridge minimizes `mean((Phi @ beta - y)**2) + 1e-4 * ||beta||²`. Its filter is `s / (s² + 0.1024)` for 1,024 training examples.
- Sensitivity cutoffs are `1e-10` and `1e-14`; these are diagnostics, not alternative result curves selected after inspection.
- Numerical interpolation requires relative training residual `||Phi @ beta - y|| / max(||y||, 1) <= 1e-8`.

Primary evaluation is test MSE against the clean teacher. Also retain observed-label and training/validation MSE, singular spectrum, numerical rank, condition number, coefficient norm, residuals, durations and CPU RSS observations. A square matrix at `p/n = 1` is a geometric reference; actual residuals determine interpolation.

Generate only development data initially. Freeze fitted-model identities before generating the test partition, then evaluate every registered predictor. There is no test-optimal checkpoint selection. Report individual seeds, arithmetic mean, sample standard deviation and contributing counts, retaining failures and extreme values.

## Execution and budget

Use four CPU compute threads, 1,024-example evaluation chunks and sequential units. The largest training matrix and evaluation chunk are each 64 MiB, excluding other arrays and solver workspace. This size calculation is not a measured process-memory peak.

Preflight measures capacities 256, 1,024 and 8,192 for the first seed and reuses the completed units. The active-time allowance is 30 minutes, with a conservative projection before the main sweep and budget checks between units. No automatic budget extension or experiment reduction is allowed. A running decomposition can finish beyond the deadline; the stop rule is not an exact wall-clock cutoff.

The measured run completed in 124.65 active seconds on an M2 Pro. That observation is specific to the recorded environment and workload. [Reproduction instructions](reproduce.md) describe installation, fresh runs, recovery and portability.

## Interpretation boundaries

Ten feature seeds describe variability conditional on one dataset and noise realization; they are not a confidence interval across datasets. Nested features correlate adjacent capacities. The teacher, bandwidth, fixed ridge penalty and input distribution limit transfer to other problems. This study provides no evidence about learned neural representations or a universal benefit of larger models.

The [original 256-example protocol](protocol.md) is historical. Both sample size and data realization changed in this follow-up, so a cross-run difference is not an isolated causal estimate of sample-size effects. Results from distinct protocols must not be pooled.

The current repository includes portability and presentation changes made after the measured run. Its exact historical source/configuration remains in the public evidence package; no completed scientific run identity was rewritten.
