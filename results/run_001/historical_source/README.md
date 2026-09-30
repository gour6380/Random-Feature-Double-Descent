# Source that generated the measured run

These source files, authored tests and original hashed dependency lock are copied byte-for-byte from the completed run's source snapshot. Their SHA-256 values are listed under `source` in [the identity record](../provenance/identity.json).

This directory records historical provenance. Use the main repository's notebook and current source for new runs. The historical snapshot includes macOS-specific implementation details and the original dependency lock; later portability fixes in the main source intentionally have different identities. They do not change the already recorded experiment or retroactively replace its 61-test receipt.

No package is installed from this directory, and no snapshot module is imported by the public export/redraw utility. The snapshot does not include raw data or models. Its accompanying tests are evidence of what the saved historical runtime check exercised, not an assertion that current-source checks are identical.
