"""Explicit owner-run correctness gate. Importing this module runs no tests."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from .io import atomic_json, canonical_hash, read_json
from .runs import environment_record, source_identity, utc_now


def validate_runtime(run: Any) -> dict[str, Any]:
    """Execute bounded CPU correctness tests only when this notebook action is run."""
    with run.writer_lock(), run.work("runtime correctness checks"):
        run.verify_identity()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=run.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        (run.path / "logs" / "runtime_checks.txt").write_text(
            result.stdout + "\n" + result.stderr, encoding="utf-8"
        )
        receipt = {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "source_hash": canonical_hash(source_identity(run.project_root)),
            "environment_hash": canonical_hash(environment_record()),
            "at": utc_now(),
        }
        atomic_json(run.path / "logs" / "runtime_checks.json", receipt)
        if result.returncode:
            raise RuntimeError(
                "Correctness checks failed; inspect logs/runtime_checks.txt before fitting"
            )
        return receipt


def require_runtime_validation(run: Any) -> None:
    path = run.path / "logs" / "runtime_checks.json"
    if not path.exists():
        raise RuntimeError("Execute the explicit validate_runtime cell before the timing preflight")
    receipt = read_json(path)
    if (
        not receipt.get("passed")
        or receipt["source_hash"] != canonical_hash(source_identity(run.project_root))
        or receipt["environment_hash"] != canonical_hash(environment_record())
    ):
        raise ValueError("A passing correctness receipt for this source/environment is required")
