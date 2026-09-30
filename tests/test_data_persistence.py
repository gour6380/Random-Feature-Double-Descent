"""Preparation recovery and held-out-data gates; authored, not executed at handoff."""

from dataclasses import replace

import pytest

from src import data, evaluate, runs
from src.config import DataConfig, ExecutionConfig, ExperimentConfig, FeatureConfig
from src.io import atomic_json, read_json, sha256_file


@pytest.fixture
def data_run(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "example.py").write_text("VALUE = 1\n")
    (root / "requirements.txt").write_text("example==1\n")
    monkeypatch.setattr(runs, "environment_record", lambda: {"backend": "cpu", "dtype": "float64"})
    monkeypatch.setattr(runs, "configure_runtime", lambda config: config.validate())
    config = ExperimentConfig(
        data=DataConfig(train_size=7, validation_size=5, test_size=9),
        features=FeatureConfig(counts=(2, 4, 8), seeds=(2026,)),
        execution=ExecutionConfig(preflight_widths=(2, 4, 8)),
    )
    return runs.create_run(root, config)


def test_development_preparation_never_generates_test(data_run, monkeypatch):
    generated = []
    original = data.generate_partition

    def observe(config, name):
        generated.append(name)
        return original(config, name)

    monkeypatch.setattr(data, "generate_partition", observe)
    first = data.prepare_data(data_run)
    second = data.prepare_data(data_run)
    assert generated == ["train", "validation"]
    assert data.partition_digest(first.train) == data.partition_digest(second.train)
    assert not (data_run.path / "data" / "test.pt").exists()
    assert data_run.data_identity("development")


def test_missing_registry_commit_recovers_identical_prepared_files(data_run, monkeypatch):
    original = data_run.register_data_files

    def crash(*args, **kwargs):
        raise OSError("simulated registry commit failure")

    monkeypatch.setattr(data_run, "register_data_files", crash)
    with pytest.raises(OSError, match="simulated"):
        data.prepare_data(data_run)
    prepared = {
        path: sha256_file(path) for path in (data_run.path / "data").rglob("*") if path.is_file()
    }
    assert not (data_run.path / "data" / "development_manifest.json").exists()
    monkeypatch.setattr(data_run, "register_data_files", original)
    data.prepare_data(data_run)
    assert all(sha256_file(path) == digest for path, digest in prepared.items())
    assert data_run.data_identity("development")


def test_partial_registry_is_not_trusted_as_complete(data_run):
    data.prepare_data(data_run)
    path = data_run.path / "data" / "development_manifest.json"
    registry = read_json(path)
    registry["files"].pop("data/development.json")
    atomic_json(path, registry)
    with pytest.raises(ValueError, match="exactly"):
        data.prepare_data(data_run)


def test_test_arrays_require_freeze_before_generation(data_run, monkeypatch):
    data.prepare_data(data_run)

    def reject(run):
        raise RuntimeError("selection is not frozen")

    monkeypatch.setattr(evaluate, "verify_frozen", reject)
    with pytest.raises(RuntimeError, match="not frozen"):
        data.prepare_test_data(data_run)
    assert not (data_run.path / "data" / "test.pt").exists()


def test_test_cache_is_bound_to_frozen_model_identities(data_run, monkeypatch):
    development = data.prepare_data(data_run)
    monkeypatch.setattr(evaluate, "verify_frozen", lambda run: {"verified": True})
    frozen_path = data_run.path / "evaluation" / "frozen.json"
    atomic_json(frozen_path, {"frozen": "version-a"})
    test = data.prepare_test_data(data_run)
    data.assert_disjoint(development.train, development.validation, test)
    assert test.name == "test" and len(test) == 9
    original_digest = data.partition_digest(test)
    assert data.partition_digest(data.prepare_test_data(data_run)) == original_digest
    atomic_json(frozen_path, {"frozen": "version-b"})
    with pytest.raises(ValueError, match="different frozen"):
        data.prepare_test_data(data_run)


def test_edited_config_cannot_reuse_a_data_stream_for_features():
    with pytest.raises(ValueError, match="independent seed streams"):
        replace(ExperimentConfig(), features=replace(FeatureConfig(), seeds=(1001,))).validate()
