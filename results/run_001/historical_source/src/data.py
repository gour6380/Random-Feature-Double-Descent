"""Fixed synthetic partitions; test arrays require a frozen experiment first."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .features import generate_bank
from .io import atomic_json, atomic_torch_save, canonical_hash, read_json, sha256_file

if TYPE_CHECKING:
    from .config import DataConfig
    from .runs import Run


@dataclass(frozen=True)
class Partition:
    name: str
    x: torch.Tensor
    clean: torch.Tensor
    noise: torch.Tensor
    labels: torch.Tensor
    ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.ids)

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x": self.x,
            "clean": self.clean,
            "noise": self.noise,
            "labels": self.labels,
            "ids": list(self.ids),
        }


@dataclass(frozen=True)
class DevelopmentData:
    train: Partition
    validation: Partition


def teacher(x: torch.Tensor) -> torch.Tensor:
    """Population-unit-variance signal for independent standard-normal inputs."""
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("The teacher needs at least three input coordinates.")
    scale = sqrt((1 - exp(-2)) / 2 * (1 + 0.5**2 + 0.25**2))
    return (x[:, 0].sin() + 0.5 * x[:, 1].sin() + 0.25 * x[:, 2].sin()) / scale


def partition_digest(partition: Partition) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps({"name": partition.name, "ids": partition.ids}, sort_keys=True).encode()
    )
    for name in ("x", "clean", "noise", "labels"):
        tensor = getattr(partition, name).detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def generate_partition(config: DataConfig, name: str) -> Partition:
    config.validate()
    names = ("train", "validation", "test")
    if name not in names:
        raise ValueError("Unknown partition.")
    index = names.index(name)
    size = (config.train_size, config.validation_size, config.test_size)[index]
    input_generator = torch.Generator(device="cpu").manual_seed(config.input_seeds[index])
    noise_generator = torch.Generator(device="cpu").manual_seed(config.noise_seeds[index])
    x = torch.randn(
        (size, config.input_dim), generator=input_generator, dtype=torch.float64, device="cpu"
    )
    clean = teacher(x)
    noise = torch.randn(size, generator=noise_generator, dtype=torch.float64, device="cpu")
    sigmas = torch.tensor(config.noise_stds, dtype=torch.float64, device="cpu")
    labels = clean[:, None] + noise[:, None] * sigmas[None, :]
    ids = tuple(f"{name}:{i:08d}" for i in range(size))
    return Partition(name, x, clean, noise, labels, ids)


def assert_disjoint(*partitions: Partition) -> None:
    seen_ids: set[str] = set()
    seen_inputs: set[bytes] = set()
    for partition in partitions:
        ids = set(partition.ids)
        inputs = {hashlib.sha256(row.numpy().tobytes()).digest() for row in partition.x}
        if len(ids) != len(partition) or len(inputs) != len(partition):
            raise ValueError("Duplicate sample identity/input within a partition.")
        if seen_ids & ids or seen_inputs & inputs:
            raise ValueError("Partitions overlap in identities or exact input rows.")
        seen_ids.update(ids)
        seen_inputs.update(inputs)


def load_partition(path: Path | str) -> Partition:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    partition = Partition(**{**payload["partition"], "ids": tuple(payload["partition"]["ids"])})
    if partition_digest(partition) != payload["digest"]:
        raise ValueError("Partition tensor/metadata identity mismatch.")
    return partition


def _save_partition(
    path: Path, partition: Partition, config: DataConfig, *, frozen_identity: str | None = None
) -> None:
    index = ("train", "validation", "test").index(partition.name)
    payload = {
        "partition": partition.payload(),
        "digest": partition_digest(partition),
        "generation": {
            "input_seed": config.input_seeds[index],
            "noise_seed": config.noise_seeds[index],
            "noise_stds": list(config.noise_stds),
            "input_dim": config.input_dim,
            "distribution": "independent standard normal inputs/noise",
            "teacher": "(sin(x0)+0.5*sin(x1)+0.25*sin(x2))/sqrt((1-exp(-2))/2*1.3125)",
            "frozen_identity": frozen_identity,
        },
    }
    if path.exists():
        prior = torch.load(path, map_location="cpu", weights_only=True)
        if (
            partition_digest(load_partition(path)) != payload["digest"]
            or prior["generation"] != payload["generation"]
        ):
            raise ValueError("An interrupted data preparation left incompatible artifacts.")
        return
    atomic_torch_save(path, payload)


def _verify_registry_contents(run: Run, scope: str, paths: list[Path]) -> None:
    run.data_identity(scope)
    expected = {path.relative_to(run.path).as_posix() for path in paths}
    actual = set(read_json(run.path / "data" / f"{scope}_manifest.json")["files"])
    if actual != expected:
        raise ValueError(f"{scope} registry does not cover exactly the required data artifacts.")


def prepare_data(run: Run) -> DevelopmentData:
    """Generate/cache train, validation and fixed feature banks; never touch test arrays."""
    with run.writer_lock(), run.work("prepare_development_data"):
        run.verify_identity()
        run.config.validate()
        directory = run.path / "data"
        marker = directory / "development.json"
        paths = [directory / "train.pt", directory / "validation.pt"]
        paths.extend(
            directory / "feature_banks" / f"seed_{seed}.pt" for seed in run.config.features.seeds
        )
        if (directory / "development_manifest.json").exists():
            _verify_registry_contents(run, "development", [*paths, marker])
            result = DevelopmentData(
                load_partition(directory / "train.pt"), load_partition(directory / "validation.pt")
            )
            assert_disjoint(result.train, result.validation)
            return result
        train = generate_partition(run.config.data, "train")
        validation = generate_partition(run.config.data, "validation")
        assert_disjoint(train, validation)
        _save_partition(paths[0], train, run.config.data)
        _save_partition(paths[1], validation, run.config.data)
        for seed in run.config.features.seeds:
            path = directory / "feature_banks" / f"seed_{seed}.pt"
            bank = generate_bank(run.config, seed).payload()
            if path.exists():
                prior = torch.load(path, map_location="cpu", weights_only=True)
                for key, value in bank.items():
                    same = (
                        torch.equal(prior[key], value)
                        if isinstance(value, torch.Tensor)
                        else prior[key] == value
                    )
                    if not same:
                        raise ValueError(
                            "Interrupted preparation contains an incompatible feature bank."
                        )
            else:
                atomic_torch_save(path, bank)
        manifest = {
            "partitions": {
                part.name: {"count": len(part), "digest": partition_digest(part)}
                for part in (train, validation)
            },
            "feature_seeds": list(run.config.features.seeds),
            "max_features": max(run.config.features.counts),
            "test_generated": False,
            "config_hash": canonical_hash(run.config.to_dict()),
        }
        if marker.exists() and read_json(marker) != manifest:
            raise ValueError("Interrupted preparation contains incompatible data metadata.")
        if not marker.exists():
            atomic_json(marker, manifest)
        run.register_data_files(
            [path.relative_to(run.path) for path in [*paths, marker]], scope="development"
        )
        return DevelopmentData(train, validation)


def prepare_test_data(run: Run) -> Partition:
    """Generate test data only after all registered fit identities are frozen."""
    from .evaluate import verify_frozen

    with run.writer_lock(), run.work("prepare_test_data"):
        run.verify_identity()
        verify_frozen(run)
        run.verify_data("development")
        path = run.path / "data" / "test.pt"
        frozen_identity = sha256_file(run.path / "evaluation" / "frozen.json")
        if (run.path / "data" / "test_manifest.json").exists():
            _verify_registry_contents(run, "test", [path])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["generation"]["frozen_identity"] != frozen_identity:
                raise ValueError("Test data was prepared for a different frozen experiment.")
            return load_partition(path)
        test = generate_partition(run.config.data, "test")
        assert_disjoint(
            load_partition(run.path / "data" / "train.pt"),
            load_partition(run.path / "data" / "validation.pt"),
            test,
        )
        _save_partition(path, test, run.config.data, frozen_identity=frozen_identity)
        run.register_data_files([path.relative_to(run.path)], scope="test")
        return test
