"""Environment relocation and native writer locks; no experiment data or fitting."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src import io, runs, runtime
from src.config import ExperimentConfig


@pytest.fixture
def portable_project(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)
    (root / "src" / "example.py").write_text("VALUE = 1\n")
    (root / "requirements.txt").write_text("example==1\n")
    environment = {
        "python": "3.13.11",
        "machine": "arm64",
        "platform": "test-platform",
        "executable": "/old/installation/python",
        "packages": {"torch": "2.14.0", "numpy": "2.4.1", "pandas": "3.0.0", "extra": "1"},
        "torch": "2.14.0",
        "device": "cpu",
        "dtype": "float64",
        "default_dtype": "torch.float32",
        "default_device": "cpu",
        "threads": 4,
        "interop_threads": 10,
        "torch_build": "recorded numerical backend",
        "thread_environment": {"OMP_NUM_THREADS": None},
    }
    monkeypatch.setattr(runs, "environment_record", lambda: copy.deepcopy(environment))
    monkeypatch.setattr(runtime, "environment_record", lambda: copy.deepcopy(environment))
    monkeypatch.setattr(runs, "configure_runtime", lambda config: config.validate())
    return root, environment


def test_new_identity_ignores_installation_path_and_unrelated_packages(portable_project):
    root, environment = portable_project
    run = runs.create_run(root, ExperimentConfig())
    saved = io.read_json(run.path / "environment" / "environment.json")
    environment["executable"] = "/new/location/another-env/python"
    environment["packages"]["extra"] = "2"
    environment["packages"]["notebook-extension"] = "99"
    run.verify_identity()
    assert saved["executable"] == "/old/installation/python"
    assert io.read_json(run.path / "environment" / "environment.json") == saved


@pytest.mark.parametrize("package", runs.NUMERICAL_PACKAGES)
def test_core_dependency_changes_reject_resume(portable_project, package):
    root, environment = portable_project
    run = runs.create_run(root, ExperimentConfig())
    environment["packages"][package] = "changed-version"
    with pytest.raises(ValueError, match="Environment"):
        run.verify_identity()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("python", "3.14.0"),
        ("machine", "x86_64"),
        ("platform", "different-platform"),
        ("device", "mps"),
        ("dtype", "float32"),
        ("threads", 8),
        ("interop_threads", 1),
        ("torch_build", "different numerical backend"),
        ("thread_environment", {"OMP_NUM_THREADS": "8"}),
    ],
)
def test_scientific_runtime_changes_remain_protected(portable_project, key, value):
    root, environment = portable_project
    run = runs.create_run(root, ExperimentConfig())
    environment[key] = value
    with pytest.raises(ValueError, match="Environment"):
        run.verify_identity()


def test_saved_informational_provenance_is_integrity_checked(portable_project):
    root, _ = portable_project
    run = runs.create_run(root, ExperimentConfig())
    path = run.path / "environment" / "environment.json"
    record = io.read_json(path)
    record["executable"] = "tampered-history"
    io.atomic_json(path, record)
    with pytest.raises(ValueError, match="Stored environment provenance"):
        run.verify_identity()


def test_relocated_checkout_and_runtime_receipt_share_compatibility_policy(
    portable_project, tmp_path, monkeypatch
):
    from src import tracking

    root, environment = portable_project
    run = runs.create_run(root, ExperimentConfig())
    # Exercise receipt construction without launching pytest or scientific checks.
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="stubbed checks", stderr=""),
    )
    receipt = runtime.validate_runtime(run)
    assert receipt["environment_policy"] == runs.ENVIRONMENT_POLICY
    moved = tmp_path / "different checkout location"
    shutil.copytree(root, moved)
    environment["executable"] = "/relocated/interpreter/python"
    environment["packages"]["extra"] = "2"
    monkeypatch.setattr(tracking, "refresh_tracking", lambda run: None)
    resumed = runs.resume_run(moved, run.path.name)
    runtime.require_runtime_validation(resumed)
    environment["packages"]["numpy"] = "changed-version"
    with pytest.raises(ValueError, match="correctness receipt"):
        runtime.require_runtime_validation(resumed)


def test_legacy_identities_and_runtime_receipts_remain_strict(portable_project):
    root, environment = portable_project
    run = runs.create_run(root, ExperimentConfig())
    path = run.path / "config" / "identity.json"
    identity = io.read_json(path)
    identity.pop("environment_policy")
    identity.pop("environment_record_hash")
    identity["environment_hash"] = io.canonical_hash(environment)
    io.atomic_json(path, identity)
    io.atomic_json(
        run.path / "logs" / "runtime_checks.json",
        {
            "passed": True,
            "source_hash": io.canonical_hash(runs.source_identity(root)),
            "environment_hash": io.canonical_hash(environment),
        },
    )
    run.verify_identity()
    runtime.require_runtime_validation(run)
    environment["executable"] = "/relocated/python"
    with pytest.raises(ValueError, match="Environment"):
        run.verify_identity()
    with pytest.raises(ValueError, match="correctness receipt"):
        runtime.require_runtime_validation(run)


def test_native_lock_excludes_another_process_and_releases_after_interrupt(portable_project):
    root, _ = portable_project
    run = runs.create_run(root, ExperimentConfig())
    script = """
import sys
from filelock import FileLock, Timeout
lock = FileLock(sys.argv[1], timeout=0, fallback_to_soft=False, preserve_lock_file=True)
try:
    with lock:
        pass
except Timeout:
    sys.exit(3)
"""

    def probe():
        return subprocess.run(
            [sys.executable, "-B", "-c", script, str(run.path / ".writer.lock")],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    with pytest.raises(KeyboardInterrupt):
        with run.writer_lock():
            result = probe()
            assert result.returncode == 3, result.stderr
            raise KeyboardInterrupt
    result = probe()
    assert result.returncode == 0, result.stderr
    assert run._lock_depth == 0
    assert (run.path / ".writer.lock").is_file()


def test_unsupported_native_lock_fails_without_silent_fallback(portable_project, monkeypatch):
    root, _ = portable_project
    run = runs.create_run(root, ExperimentConfig())

    class UnavailableLock:
        def __init__(self, path, **kwargs):
            assert kwargs["fallback_to_soft"] is False

        def acquire(self):
            raise OSError("native lock unavailable")

    monkeypatch.setattr(runs, "FileLock", UnavailableLock)
    with pytest.raises(OSError, match="native lock unavailable"):
        with run.writer_lock():
            pytest.fail("Writer entered without a native lock")
    assert run._lock_depth == 0
