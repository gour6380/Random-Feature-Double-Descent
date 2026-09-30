"""Numbered runs, provenance, exclusive writers and persistent active-time budgets."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from filelock import FileLock, Timeout

from .config import ExperimentConfig
from .io import atomic_json, canonical_hash, read_json, sha256_file, verify_directory

ENVIRONMENT_POLICY = "portable-v1"
NUMERICAL_PACKAGES = ("numpy", "pandas", "torch")


class BudgetExceeded(RuntimeError):
    """Stop at a work boundary, retaining committed work."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def source_identity(root: Path) -> dict[str, str]:
    files = (
        sorted((root / "src").glob("*.py"))
        + sorted((root / "tests").glob("*.py"))
        + [root / "requirements.txt"]
    )
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def environment_record() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": dict(
            sorted(
                (d.metadata["Name"].lower(), d.version)
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            )
        ),
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": "float64",
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.get_default_device()),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "torch_build": torch.__config__.show(),
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def environment_fingerprint(record: dict[str, Any], policy: str = ENVIRONMENT_POLICY) -> str:
    """Compare scientific runtime settings independently of installation location.

    The complete environment remains saved as informational provenance. New runs
    tolerate interpreter relocation and changes to unrelated packages, while Python,
    numerical libraries, platform/build, backend, precision and thread settings stay
    protected. Legacy runs/receipts retain their original full-record comparison.
    """
    if policy == "strict-v1":
        return canonical_hash(record)
    if policy != ENVIRONMENT_POLICY:
        raise ValueError(f"Unknown environment compatibility policy: {policy}")
    relevant = {
        key: value for key, value in record.items() if key not in {"executable", "packages"}
    }
    packages = {
        re.sub(r"[-_.]+", "-", name.lower()): version
        for name, version in record.get("packages", {}).items()
    }
    relevant["packages"] = {name: packages.get(name) for name in NUMERICAL_PACKAGES}
    return canonical_hash(relevant)


def configure_runtime(config: ExperimentConfig) -> None:
    import torch

    config.validate()
    torch.set_num_threads(config.execution.threads)


@dataclass
class Run:
    project_root: Path
    path: Path
    config: ExperimentConfig
    _mutex: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _lock_depth: int = field(default=0, repr=False)
    _work_depth: int = field(default=0, repr=False)

    @property
    def logger(self) -> logging.Logger:
        logger = logging.getLogger(f"random_features.{self.path}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(self.path / "logs" / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        with self._mutex:
            if self._lock_depth:
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                return
            lock = FileLock(
                self.path / ".writer.lock",
                timeout=0,
                fallback_to_soft=False,
                preserve_lock_file=True,
            )
            try:
                lock.acquire()
            except Timeout as exc:
                raise RuntimeError("Another process is writing this run") from exc
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
                lock.release()

    def verify_identity(self) -> None:
        self.config.validate()
        identity = read_json(self.path / "config" / "identity.json")
        if canonical_hash(self.config.to_dict()) != identity["config_hash"]:
            raise ValueError("Configuration changed; create a fresh run")
        if (
            canonical_hash(read_json(self.path / "config" / "experiment.json"))
            != identity["config_hash"]
        ):
            raise ValueError("Stored configuration changed")
        if source_identity(self.project_root) != identity["source"]:
            raise ValueError("Source/dependency lock changed; create a fresh run")
        policy = identity.get("environment_policy", "strict-v1")
        stored_environment = read_json(self.path / "environment" / "environment.json")
        stored_hash = identity.get("environment_record_hash")
        if stored_hash is not None and canonical_hash(stored_environment) != stored_hash:
            raise ValueError("Stored environment provenance changed")
        if environment_fingerprint(environment_record(), policy) != identity["environment_hash"]:
            raise ValueError(
                "Environment/backend/thread settings changed; use the recorded environment"
            )
        for scope in ("development", "test"):
            self.verify_data(scope)

    def _registry(self, scope: str) -> Path:
        if scope not in {"development", "test"}:
            raise ValueError("Unknown data scope")
        return self.path / "data" / f"{scope}_manifest.json"

    def register_data_files(self, paths: list[Path], scope: str = "development") -> None:
        registry_path = self._registry(scope)
        with self.writer_lock():
            registry = read_json(registry_path) if registry_path.exists() else {"files": {}}
            self.verify_data(scope)
            for path in paths:
                path = Path(path)
                if not path.is_absolute():
                    path = self.path / path
                relative = path.resolve().relative_to(self.path.resolve()).as_posix()
                digest = sha256_file(path)
                if relative in registry["files"] and registry["files"][relative] != digest:
                    raise ValueError(f"Data artifact was modified: {relative}")
                if (
                    scope == "development"
                    and (self.path / "evaluation" / "frozen.json").exists()
                    and relative not in registry["files"]
                ):
                    raise ValueError("Development identity is frozen")
                registry["files"][relative] = digest
            atomic_json(registry_path, registry)

    def verify_data(self, scope: str = "development") -> None:
        registry_path = self._registry(scope)
        if not registry_path.exists():
            return
        registry = read_json(registry_path)
        for relative, digest in registry["files"].items():
            path = (self.path / relative).resolve()
            path.relative_to(self.path.resolve())
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Data identity mismatch: {relative}")

    def data_identity(self, scope: str = "development") -> str:
        self.verify_data(scope)
        registry = self._registry(scope)
        if not registry.exists():
            raise ValueError(f"{scope} data has not been prepared")
        return canonical_hash(read_json(registry))

    def budget_status(self) -> dict[str, Any]:
        record = read_json(self.path / "logs" / "budget.json")
        record["remaining_seconds"] = max(0.0, record["limit_seconds"] - record["used_seconds"])
        return record

    def check_budget(self) -> None:
        if self.budget_status()["remaining_seconds"] <= 0:
            raise BudgetExceeded(
                "Active-time budget exhausted; completed work is safe. An explicit recorded budget extension is required to continue."
            )

    @contextmanager
    def work(self, label: str, *, allow_exhausted: bool = False) -> Iterator[None]:
        """Count active execution, excluding notebook idle time; nested work counts once.

        Heartbeats preserve elapsed work after hard termination to roughly one-second
        resolution. A work unit may overrun the budget before its next boundary.
        """
        with self.writer_lock():
            if self._work_depth:
                self._work_depth += 1
                try:
                    yield
                finally:
                    self._work_depth -= 1
                return
            if not allow_exhausted:
                self.check_budget()
            record = self.budget_status()
            record.pop("remaining_seconds", None)
            base = record["used_seconds"]
            start = time.monotonic()
            stop = threading.Event()
            errors: list[Exception] = []
            record["active"] = {"label": label, "pid": os.getpid(), "started_at": utc_now()}
            atomic_json(self.path / "logs" / "budget.json", record)

            def heartbeat() -> None:
                while not stop.wait(1.0):
                    try:
                        update = dict(record, used_seconds=base + time.monotonic() - start)
                        atomic_json(self.path / "logs" / "budget.json", update)
                    except Exception as exc:
                        errors.append(exc)
                        return

            thread = threading.Thread(target=heartbeat, daemon=True)
            self._work_depth = 1
            started = False
            try:
                thread.start()
                started = True
                self.logger.info("START %s", label)
                yield
            except BaseException:
                self.logger.exception("INTERRUPTED/FAILED %s", label)
                raise
            finally:
                stop.set()
                if started:
                    thread.join()
                try:
                    elapsed = time.monotonic() - start
                    record.update(used_seconds=base + elapsed, active=None)
                    atomic_json(self.path / "logs" / "budget.json", record)
                    self.logger.info("END %s %.3fs", label, elapsed)
                finally:
                    self._work_depth = 0
            if errors:
                raise OSError("Budget heartbeat could not be persisted") from errors[0]


def create_run(project_root: Path, config: ExperimentConfig) -> Run:
    root = Path(project_root).resolve()
    configure_runtime(config)
    sources = source_identity(root)
    environment = environment_record()
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    number = 1
    while True:
        path = runs / f"run_{number:03d}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            number += 1
    for name in (
        "config",
        "environment",
        "source_snapshot",
        "data",
        "features",
        "logs",
        "tensorboard",
        "trials",
        "evaluation",
        "tables",
        "figures",
        "report",
    ):
        (path / name).mkdir()
    for relative in sources:
        destination = path / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    atomic_json(path / "config" / "experiment.json", config.to_dict())
    atomic_json(path / "environment" / "environment.json", environment)
    atomic_json(
        path / "config" / "identity.json",
        {
            "created_at": utc_now(),
            "config_hash": canonical_hash(config.to_dict()),
            "source": sources,
            "environment_policy": ENVIRONMENT_POLICY,
            "environment_hash": environment_fingerprint(environment),
            "environment_record_hash": canonical_hash(environment),
        },
    )
    atomic_json(
        path / "logs" / "budget.json",
        {
            "used_seconds": 0.0,
            "limit_seconds": config.execution.budget_seconds,
            "extensions": [],
            "active": None,
        },
    )
    return Run(root, path, config)


def resume_run(project_root: Path, run_name: str, config: ExperimentConfig | None = None) -> Run:
    if not re.fullmatch(r"run_\d{3,}", run_name):
        raise ValueError("Use an explicit numbered run name")
    root = Path(project_root).resolve()
    path = root / "runs" / run_name
    stored = ExperimentConfig.from_dict(read_json(path / "config" / "experiment.json"))
    selected = stored if config is None else config
    configure_runtime(selected)
    run = Run(root, path, selected)
    with run.writer_lock():
        run.verify_identity()
        budget = run.budget_status()
        if budget.get("active"):
            # The process lock is free: the previous writer terminated. Do not count
            # downtime as compute. Retain its persisted one-second heartbeat.
            budget.setdefault("interrupted_operations", []).append(budget["active"])
            budget.update(active=None, heartbeat_recovery_precision_seconds=1.0)
            budget.pop("remaining_seconds", None)
            atomic_json(path / "logs" / "budget.json", budget)
        for manifest in sorted((path / "trials").glob("seed_*/features_*/manifest.json")):
            verify_directory(manifest.parent)
        from .tracking import refresh_tracking

        refresh_tracking(run)
    return run


def extend_budget(run: Run, extra_seconds: float, reason: str) -> dict[str, Any]:
    import math

    if (
        isinstance(extra_seconds, bool)
        or not math.isfinite(extra_seconds)
        or extra_seconds <= 0
        or not reason.strip()
    ):
        raise ValueError("A positive finite extension and a reason are required")
    with run.writer_lock():
        run.verify_identity()
        if run._work_depth:
            raise RuntimeError("Extend the budget between operations")
        record = run.budget_status()
        record.pop("remaining_seconds", None)
        record["limit_seconds"] += extra_seconds
        record["extensions"].append({"seconds": extra_seconds, "reason": reason, "at": utc_now()})
        atomic_json(run.path / "logs" / "budget.json", record)
        return record


def inspect_environment(run: Run) -> dict[str, Any]:
    run.verify_identity()
    return {
        "environment": read_json(run.path / "environment" / "environment.json"),
        "budget": run.budget_status(),
    }


def summarize_run(run: Run) -> list[dict[str, Any]]:
    result = []
    for seed in run.config.features.seeds:
        for count in run.config.features.counts:
            path = run.path / "trials" / f"seed_{seed}" / f"features_{count:04d}"
            if path.exists():
                verify_directory(path)
                status = read_json(path / "status.json")
            else:
                status = {"status": "pending"}
            result.append({"seed": seed, "n_features": count, **status})
    return result
