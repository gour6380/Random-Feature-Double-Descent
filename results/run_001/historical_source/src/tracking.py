"""Rebuild human/machine tracking only from verified committed units."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .io import read_json, verify_directory


def _atomic_text(path: Path, content: str) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def committed_metrics(run: Any, scope: str = "trials") -> list[dict[str, Any]]:
    if scope not in {"trials", "evaluation"}:
        raise ValueError("Unknown tracking scope")
    rows = []
    for path in sorted((run.path / scope).glob("seed_*/features_*")):
        verify_directory(path)
        rows.extend(read_json(path / "metrics.json"))
    return rows


def refresh_tracking(run: Any) -> None:
    """Reconciliation is idempotent; interrupted or duplicate events disappear."""
    from torch.utils.tensorboard import SummaryWriter

    with run.writer_lock(), run.work("reconcile_tracking", allow_exhausted=True):
        run.verify_identity()
        rows = committed_metrics(run)
        columns = sorted({key for row in rows for key in row})
        output = io.StringIO()
        if columns:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        _atomic_text(run.path / "logs" / "metrics.csv", output.getvalue())
        _atomic_text(
            run.path / "logs" / "metrics.jsonl",
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        )
        temporary = run.path / f".tensorboard-{uuid.uuid4().hex}"
        backup = run.path / f".old-tensorboard-{uuid.uuid4().hex}"
        temporary.mkdir()
        published = False
        try:
            with SummaryWriter(str(temporary)) as writer:
                for row in rows:
                    group = f"seed_{row['seed']}/noise_{row['noise_std']}/{row['solver']}"
                    step = run.config.features.counts.index(row["n_features"])
                    writer.add_scalar(f"{group}/n_features", row["n_features"], step)
                    for key, value in row.items():
                        if (
                            key not in {"seed", "n_features", "noise_std"}
                            and isinstance(value, (int, float))
                            and math.isfinite(value)
                        ):
                            writer.add_scalar(f"{group}/{key}", value, step)
            destination = run.path / "tensorboard"
            if destination.exists():
                os.rename(destination, backup)
            try:
                os.rename(temporary, destination)
                published = True
            except OSError:
                if backup.exists() and not destination.exists():
                    os.rename(backup, destination)
                raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
            if published and backup.exists():
                shutil.rmtree(backup)
