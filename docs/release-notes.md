# Publication preparation — 28 September 2026

The completed protocol 1.1 is now presented through a public results package, a result-specific technical report and a runnable notebook guide. The original notebook's saved outputs and execution counts are retained. The complete local run remains unchanged and excluded from uploads.

## Portability changes

- Replaced the direct Unix-only `fcntl` lock with `filelock` native locks. Nonblocking writer exclusion, nested entry and interruption release remain enforced; silent soft-lock fallback is disabled.
- New runs ignore interpreter location and unrelated installed packages when comparing environments. They still protect scientific configuration, source, data, numerical-library versions, precision, backend and thread settings. Full informational provenance is saved and integrity-checked.
- Runtime-check receipts use the same versioned compatibility policy. Historical identities retain their strict comparison and are never rewritten.
- Resolved a universal, hashed CPU dependency lock with conditional OS packages. macOS versions from the measured run are retained; Windows/Linux use corresponding CPU-only PyTorch wheels.
- Removed the fixed Jupyter kernel-name check from repository validation. Saved outputs are accepted. No startup check requires the original username, project directory, virtual-environment name or Mac.
- Corrected documentation and report-generator references to the existing notebook filename.
- Scoped default pytest discovery to the current `tests/` directory so preserved historical tests are not collected twice.

Fresh runs on other machines capture their own provenance. This does not promise that an existing scientific run can resume across changed numerical environments or that every platform produces bitwise-identical results.

## Results and preservation

The original run's 61-test receipt, source snapshot and 840 primary measurements remain historical evidence. The current source includes portability changes and is separately validated by 79 regression tests on macOS. The experiment was not rerun during publication preparation; charts were redrawn from the saved measurements.

The original notebook remains byte-for-byte unchanged, including its older diagnostic charts and local paths in saved output cells. The README embeds the clearer public figures. The full local run is about 948 MiB; curated CSV/JSON/PNG/SVG artifacts and historical source are included under `results/run_001/`.

There is no duplicate upload checkout or ZIP. Local environments, full runs, caches and private drafts are excluded by `.gitignore`. No Git command or remote publication was performed. GitHub Actions are configured but have not been observed on hosted runners.

[Verification receipt](release_checks.json) · [Technical report](../results/run_001/technical_report.md) · [Run guide](reproduce.md)

## CI checksum correction — 30 September 2026

The initial hosted static check rejected `tables/singular_spectra.csv`: the CSV writer used CRLF record endings, while `.gitattributes` normalizes CSVs to LF. The manifest therefore described the pre-normalization bytes. The public CSV matches the local file exactly after converting those line endings; all 30,710 data rows and values are unchanged.

The derived CSV now uses LF and its manifest entry matches those bytes. CSV, JSON, text and SVG exports explicitly use UTF-8/LF. Historical copied evidence remains untouched. The checker retains strict raw-byte SHA-256 validation and now rejects CRLF in hashed text before publication; it does not mask differences by normalizing before hashing.

Six standard-library regression checks cover correct UTF-8/LF data, CRLF rejection, stale hashes, changed values, binary bytes and escaping paths. They run in CI without scientific dependencies. Local static/lint/format checks and an LF-normalized checkout simulation pass. No notebook, model, figure or experiment was rerun; the original run and notebook remain unchanged. The fix must be published before hosted CI can validate it. [Fix receipt](ci_line_endings_checks.json).
