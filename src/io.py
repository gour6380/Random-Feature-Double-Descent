"""Atomic local artifacts; JSON never hides nonfinite values."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _files(path: Path) -> dict[str, str]:
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"Symlink is not an immutable artifact: {item}")
        if item.is_file() and item != path / "manifest.json":
            result[item.relative_to(path).as_posix()] = sha256_file(item)
    return result


def publish_directory(staging: Path, destination: Path) -> dict[str, Any]:
    """Called under the run writer lock; existing commits are never replaced."""
    staging, destination = Path(staging), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Committed artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"files": _files(staging)}
    atomic_json(staging / "manifest.json", manifest)
    # Every writer uses the same exclusive run lock, including recovery.
    os.rename(staging, destination)
    fsync_directory(destination.parent)
    return manifest


def verify_directory(path: Path) -> dict[str, Any]:
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if not manifest.get("files") or manifest["files"] != _files(path):
        raise ValueError(f"Committed artifact checksum mismatch: {path}")
    return manifest
