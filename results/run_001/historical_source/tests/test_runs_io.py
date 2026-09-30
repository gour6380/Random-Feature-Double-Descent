"""Authored recovery/provenance tests; implementation handoff does not execute them."""

from __future__ import annotations

import json

import pytest

from src import io, runs
from src.config import ExperimentConfig


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "example.py").write_text("VALUE = 1\n")
    (root / "requirements.txt").write_text("example==1\n")
    monkeypatch.setattr(runs, "environment_record", lambda: {"device": "cpu", "dtype": "float64"})
    monkeypatch.setattr(runs, "configure_runtime", lambda config: config.validate())
    return root


def test_run_allocation_preserves_collisions(project):
    (project / "runs" / "run_001").mkdir(parents=True)
    first = runs.create_run(project, ExperimentConfig())
    second = runs.create_run(project, ExperimentConfig())
    assert first.path.name == "run_002"
    assert second.path.name == "run_003"


def test_source_environment_and_config_protection(project, monkeypatch):
    run = runs.create_run(project, ExperimentConfig())
    run.verify_identity()
    monkeypatch.setattr(runs, "environment_record", lambda: {"device": "mps"})
    with pytest.raises(ValueError, match="Environment"):
        run.verify_identity()
    monkeypatch.setattr(runs, "environment_record", lambda: {"device": "cpu", "dtype": "float64"})
    (project / "src" / "example.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Source"):
        run.verify_identity()


def test_stored_configuration_tampering(project):
    run = runs.create_run(project, ExperimentConfig())
    path = run.path / "config" / "experiment.json"
    value = io.read_json(path)
    value["protocol_version"] = "altered"
    io.atomic_json(path, value)
    with pytest.raises(ValueError, match="Stored configuration"):
        run.verify_identity()


def test_data_hash_protection(project):
    run = runs.create_run(project, ExperimentConfig())
    payload = run.path / "data" / "sample.json"
    io.atomic_json(payload, {"x": 1})
    run.register_data_files([payload])
    original = run.data_identity()
    assert original == run.data_identity()
    io.atomic_json(payload, {"x": 2})
    with pytest.raises(ValueError, match="Data identity"):
        run.verify_data()


def test_exclusive_writer_and_nested_lock(project):
    run = runs.create_run(project, ExperimentConfig())
    other = runs.Run(project, run.path, run.config)
    with run.writer_lock(), run.writer_lock():
        with pytest.raises(RuntimeError, match="Another process"):
            with other.writer_lock():
                pass


def test_budget_counts_interruption_and_rejects_new_work(project, monkeypatch):
    run = runs.create_run(project, ExperimentConfig())
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(runs.time, "monotonic", lambda: next(clock))
    with pytest.raises(KeyboardInterrupt):
        with run.work("partial unit"):
            raise KeyboardInterrupt
    assert run.budget_status()["used_seconds"] == 2.0
    budget = run.budget_status()
    budget["limit_seconds"] = 1.0
    io.atomic_json(run.path / "logs" / "budget.json", budget)
    with pytest.raises(runs.BudgetExceeded):
        with run.work("must not start"):
            pytest.fail("Work ran after budget exhaustion")
    runs.extend_budget(run, 5, "Explicit owner extension")
    assert run.budget_status()["limit_seconds"] == 6


def test_atomic_json_preserves_prior_commit_on_replace_failure(tmp_path, monkeypatch):
    target = tmp_path / "value.json"
    io.atomic_json(target, {"value": 1})

    def fail_replace(*args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(OSError):
        io.atomic_json(target, {"value": 2})
    assert io.read_json(target) == {"value": 1}
    assert list(tmp_path.iterdir()) == [target]


def test_directory_commit_hashes_and_completed_protection(tmp_path):
    stage = tmp_path / "staging"
    stage.mkdir()
    io.atomic_json(stage / "metrics.json", [{"value": 1}])
    target = tmp_path / "committed"
    io.publish_directory(stage, target)
    assert io.verify_directory(target)["files"]
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    with pytest.raises(FileExistsError):
        io.publish_directory(replacement, target)
    (target / "metrics.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        io.verify_directory(target)


def test_tracking_reconciles_only_committed_history(project):
    from src.tracking import refresh_tracking

    run = runs.create_run(project, ExperimentConfig())
    staging = run.path / ".unit"
    staging.mkdir()
    io.atomic_json(staging / "status.json", {"status": "complete"})
    row = {
        "seed": 2026,
        "n_features": 16,
        "noise_std": 0.0,
        "solver": "minimum_norm",
        "status": "complete",
        "train_signal_mse": 0.25,
    }
    io.atomic_json(staging / "metrics.json", [row])
    io.publish_directory(staging, run.path / "trials" / "seed_2026" / "features_0016")
    (run.path / "logs" / "metrics.jsonl").write_text('{"phantom": true}\n')
    (run.path / "tensorboard" / "uncommitted-event").write_text("stale")
    refresh_tracking(run)
    assert [
        json.loads(x) for x in (run.path / "logs" / "metrics.jsonl").read_text().splitlines()
    ] == [row]
    assert not (run.path / "tensorboard" / "uncommitted-event").exists()


def test_resume_rejects_implicit_latest(project):
    with pytest.raises(ValueError, match="explicit"):
        runs.resume_run(project, "latest")


def test_nonfinite_json_rejected(tmp_path):
    with pytest.raises(ValueError):
        io.atomic_json(tmp_path / "bad.json", {"loss": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_budget_final_write_failure_releases_depth_and_lock(project, monkeypatch):
    run = runs.create_run(project, ExperimentConfig())
    original = runs.atomic_json

    def fail_final(path, value):
        if path.name == "budget.json" and value.get("active") is None:
            raise OSError("final budget write failed")
        return original(path, value)

    monkeypatch.setattr(runs, "atomic_json", fail_final)
    with pytest.raises(OSError, match="final budget"):
        with run.work("simulated work"):
            pass
    assert run._work_depth == 0
    assert run._lock_depth == 0


def test_tracking_publish_failure_restores_old_events(project, monkeypatch):
    from src import tracking

    run = runs.create_run(project, ExperimentConfig())
    marker = run.path / "tensorboard" / "previous-events"
    marker.write_text("retained")
    rename = tracking.os.rename

    def fail_new_events(source, destination):
        if str(source.name).startswith(".tensorboard-") and destination.name == "tensorboard":
            raise OSError("new events could not be published")
        return rename(source, destination)

    monkeypatch.setattr(tracking.os, "rename", fail_new_events)
    with pytest.raises(OSError, match="new events"):
        tracking.refresh_tracking(run)
    assert marker.read_text() == "retained"
