"""Small analytic solver and standalone-prediction checks, intentionally unexecuted."""

from dataclasses import replace

import pytest
import torch

from src.checkpoints import Regressor, load_regressor, save_regressor
from src.config import SolverConfig
from src.features import FeatureBank
from src.solvers import decompose, metrics, numerical_summary, solve_coefficients


def tensor(value):
    return torch.tensor(value, dtype=torch.float64, device="cpu")


@pytest.mark.parametrize(
    "design,targets,expected",
    [
        ([[1, 0, 1], [0, 1, 1]], [[1], [1]], [[1 / 3], [1 / 3], [2 / 3]]),
        ([[1], [1], [1]], [[1], [2], [3]], [[2]]),
        ([[1, 1], [2, 2], [3, 3]], [[2], [4], [6]], [[1], [1]]),
    ],
)
def test_known_minimum_norm_systems(design, targets, expected):
    phi, y = tensor(design), tensor(targets)
    decomposition = decompose(phi)
    fitted = solve_coefficients(decomposition, y, len(phi), SolverConfig())
    torch.testing.assert_close(fitted["minimum_norm"], tensor(expected), atol=1e-12, rtol=1e-12)
    assert decomposition.u.shape[1] == min(phi.shape)
    assert decomposition.vh.shape[0] == min(phi.shape)


def test_ridge_uses_number_of_samples_times_lambda():
    phi, y = tensor([[2, 0], [0, 3]]), tensor([[4, 8], [9, 18]])
    config = replace(SolverConfig(), ridge_lambda=0.25)
    fitted = solve_coefficients(decompose(phi), y, 2, config)
    expected = tensor([[8 / (4 + 0.5), 16 / (4 + 0.5)], [27 / (9 + 0.5), 54 / (9 + 0.5)]])
    torch.testing.assert_close(fitted["ridge"], expected)
    torch.testing.assert_close(fitted["minimum_norm"][:, 1], fitted["minimum_norm"][:, 0] * 2)


def test_zero_design_and_zero_targets_are_well_defined():
    phi, y = torch.zeros(3, 4, dtype=torch.float64), torch.zeros(3, 2, dtype=torch.float64)
    config = SolverConfig()
    decomposition = decompose(phi)
    fitted = solve_coefficients(decomposition, y, 3, config)
    assert all(torch.equal(beta, torch.zeros_like(beta)) for beta in fitted.values())
    summary = numerical_summary(phi, y, fitted["minimum_norm"], decomposition, config)
    assert summary[0]["numerical_rank"] == 0
    assert summary[0]["condition_number"] == float("inf")
    assert summary[0]["relative_residual"] == 0
    assert summary[0]["interpolates"] is True


def test_strict_cutoff_and_sensitivity_reuse_same_svd():
    phi, y = torch.diag(tensor([1.0, 1e-12, 1e-13])), tensor([[1], [1e-12], [1e-13]])
    config = SolverConfig()
    decomposition = decompose(phi)
    fitted = solve_coefficients(decomposition, y, 3, config)
    torch.testing.assert_close(fitted["minimum_norm"], tensor([[1], [0], [0]]))
    torch.testing.assert_close(fitted["cutoff_1e-14"], tensor([[1], [1], [1]]))
    summary = numerical_summary(phi, y, fitted["minimum_norm"], decomposition, config)
    assert summary[0]["numerical_rank"] == 1
    assert summary[0]["interpolates"] is True  # Rank alone does not decide interpolation.


def test_interpolation_uses_target_norm_clamped_to_one():
    phi, y = tensor([[1]]), tensor([[1e-10]])
    summary = numerical_summary(phi, y, tensor([[0]]), decompose(phi), SolverConfig())
    assert summary[0]["relative_residual"] == pytest.approx(1e-10)
    assert summary[0]["interpolates"] is True


def test_metrics_weight_incomplete_chunks_by_sample_count():
    pred, clean, observed = tensor([0, 0, 3]), tensor([0, 0, 0]), tensor([1, 1, 1])
    whole = metrics(pred, clean, observed)
    chunks = [
        metrics(pred[:2], clean[:2], observed[:2]),
        metrics(pred[2:], clean[2:], observed[2:]),
    ]
    assert whole["count"] == 3
    assert whole["signal_mse"] == 3
    assert whole["observed_mse"] == 2
    assert whole["signal_mse"] == sum(item["signal_sse"] for item in chunks) / sum(
        item["count"] for item in chunks
    )
    assert whole["signal_mse"] != sum(item["signal_mse"] for item in chunks) / 2


def test_nonfinite_values_fail_explicitly():
    with pytest.raises(FloatingPointError):
        decompose(tensor([[float("nan")]]))
    with pytest.raises(FloatingPointError):
        metrics(tensor([float("inf")]), tensor([0]), tensor([0]))
    with pytest.raises(ValueError, match="float64"):
        decompose(torch.ones(2, 2, dtype=torch.float32))


def test_standalone_checkpoint_preserves_encoding_and_chunking(tmp_path):
    bank = FeatureBank(
        tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), tensor([0.1, 0.2, 0.3]), 2026, 12026, 2.0
    )
    beta = tensor([1.5, -0.5])
    original = Regressor(bank, 2, beta, {"solver": "minimum_norm"})
    x = tensor([[0, 0, 0], [1, 2, 3], [-1, 1, 2], [0.5, 0.2, 0.8], [4, 3, 2]])
    path = tmp_path / "model.pt"
    save_regressor(path, bank, 2, beta, original.metadata)
    restored = load_regressor(path)
    assert restored.count == 2 and len(restored.bank.phases) == 2
    assert restored.metadata == original.metadata
    torch.testing.assert_close(
        restored.predict(x, batch_size=2),
        original.predict(x, batch_size=100),
        rtol=1e-14,
        atol=1e-14,
    )
    assert restored.predict(torch.empty((0, 3), dtype=torch.float64)).shape == (0,)
    with pytest.raises(ValueError, match="float64"):
        restored.predict(x.float())
