> Historical protocol 1.0 (256 training examples). The completed 1,024-example study uses [protocol 1.1](expanded-protocol.md) and the current notebook. Original settings below are retained as a reference.

# Registered experiment protocol

## Question and validity boundary

How do prediction error, numerical conditioning and coefficient size change as random-feature capacity crosses the training-sample count—and how do noise and regularization affect that relationship?

This is a paper-inspired adaptation, not an exact reproduction. A negative or inconclusive result is valid. Do not expand the capacity sweep after looking at test results merely to obtain a peak. A model fits only its linear coefficients; its Fourier features remain fixed.

Foundations: [Belkin et al., reconciling classical and modern generalization](https://arxiv.org/abs/1812.11118), [Rahimi and Recht, random features](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf), and [Hastie et al., ridgeless least squares](https://arxiv.org/abs/1903.08560).

## Fixed data realization

Inputs have eight independent standard-normal coordinates. Partition sizes are 256 / 1,024 / 8,192 for training / validation / test. Input seeds are 1001 / 1002 / 1003; independent noise seeds are 2001 / 2002 / 2003.

The target is

\[
f(x)=\frac{\sin(x_1)+0.5\sin(x_2)+0.25\sin(x_3)}{\sqrt{\frac{1-e^{-2}}{2}(1+0.5^2+0.25^2)}}.
\]

Its population mean is zero and variance is one under the specified input distribution. Observations are `y = f(x) + sigma * epsilon`, with `epsilon` standard normal and noise standard deviations 0 and 0.3. Both conditions share inputs, clean targets and one underlying noise vector. Neither finite-sample normalization nor learned preprocessing is applied.

Save input arrays, clean targets, noise, observed labels, stable sample identities, settings and hashes. Verify distinct partitions. Generate development data first; test arrays can only be generated after all registered fitted-model identities are frozen. Validation is for diagnostics, not hyperparameter selection. Test results cannot change fitting or checkpoint selection.

## Feature family and exact comparison

For capacity `p`, use `sqrt(2/p) * cos(x @ W.T + b)`. Each frequency has covariance `I_8 / 8`; phases are uniform over `[0, 2*pi)`. This corresponds to Gaussian-kernel lengthscale `sqrt(8)`. There is no intercept, feature standardization, learned embedding or bandwidth search.

Feature seeds are 2026, 2027, 2028, 2029 and 2030. Each seed generates a fixed 2,048-feature bank. Frequencies use the feature seed; phases use seed + 10000 through a separate local generator. Every capacity uses a nested prefix with its own `sqrt(2/p)` scaling. Data and noise remain fixed across capacities and feature seeds.

Registered capacities:

```text
16, 32, 64, 96, 128, 192, 224, 240, 248,
255, 256, 257, 264, 272, 288, 320, 384,
512, 768, 1024, 2048
```

There are 105 reduced SVDs and 420 primary fitted predictors. Sensitivity coefficients reuse each SVD and are clearly labelled diagnostics.

## SVD and numerical definitions

Use `torch.linalg.svd(Phi, full_matrices=False)` in CPU float64. Minimum-norm fitting uses inverse singular values only where `s > 1e-12 * s_max`. The comparison minimizes `||Phi beta - y||² / n + 1e-4 * ||beta||²`, giving filter `s / (s² + n * 1e-4)`. At the registered `n=256`, the added denominator term is 0.0256. Ridge uses every singular value. Do not form a matrix inverse or solve through normal equations.

Reuse the decomposition and batched right-hand sides for both noise conditions. Record the singular spectrum, numerical rank at the primary cutoff, condition number, coefficient norm and actual residuals. Numerical interpolation means `||Phi beta - y|| / max(||y||, 1) <= 1e-8`. The zero target is therefore handled without division by zero.

Cutoff sensitivities at `1e-10` and `1e-14` reuse the same SVD. Report their results without substituting them for the primary curve. The geometric boundary `p/n=1` is not proof of numerical interpolation: conditioning, rank and measured residuals remain visible.

## Execution and recovery

CPU float64 and four compute threads are intentional. Execution is sequential by feature seed with evaluation chunks of 1,024 examples. No epochs, optimizer, learning rate or GPU-utilization target is involved.

A 30-minute active-execution budget includes data preparation, preflight, fitting, evaluation, aggregation and reporting; notebook idle time and installation are excluded. Preflight fits capacities 64, 256 and 2,048 for the first seed, then conservatively projects total work. Completed preflight fits are reused. A projection over budget blocks the main sweep. A numerical failure remains explicit; infrastructure failures stop execution. The grid and budget never change automatically.

Budget checks happen between work units. A current decomposition can overrun the limit. A recorded explicit budget extension is required to continue; the study must not claim an exact real-time deadline or guaranteed 30-minute completion. An unfinished sweep cannot proceed to final test evaluation.

Fresh runs allocate numbered directories exclusively. A writer lock prevents concurrent mutation. Source, configuration, environment and dataset hashes are bound to the run. Atomically committed seed/capacity units are verified on resume; interrupted units repeat. Log decomposition counts, individual timing boundaries and CPU RSS. A completed predictor contains frequencies, phases, coefficients and construction metadata so it can load without its dataset.

## Measurements and interpretation

The primary metric is test MSE against clean `f(x)`. Secondary metrics include MSE against observed labels, train/validation MSE against both targets, rank, residual, coefficient size, conditioning, timing, memory and failure counts. MSE is aggregated by sample count, including incomplete chunks.

Each noise condition includes a baseline fitted only from its training-label mean. Population references are zero-prediction clean MSE 1, exact-teacher clean MSE 0 and noisy-observation irreducible MSE 0.09. Finite-sample baseline measurements must be distinguished from these expectations.

Show individual seeds, mean, sample standard deviation and contributing counts. Preserve extreme errors and exact zeros; axes may use a labelled symmetric-log scale. Failed conditions stay in tables with status and missing metrics. State contributing counts wherever averages appear.

The five seeds vary feature construction conditional on the fixed data/noise realization; sample SD is not a confidence interval over datasets. A nonmonotonic curve alone is insufficient to establish an interpolation explanation. Relate any observed peak to measured fitting, rank and conditioning. No result from these fixed features establishes how learned CNN representations generalize.

## Required artifacts

Machine-readable primary, sensitivity, paired-difference and aggregate tables; full singular spectra; saved predictions and sample identities; eight separate analysis figures in PNG/SVG; a technical report separating observations, interpretations, limitations and reproduction instructions; provenance and artifact hashes.

During preparation, only source, documentation, authored tests, an output-free notebook, an environment and a registered kernel are produced. Static/dependency checks do not establish numerical correctness. No scientific results exist until owner execution.
