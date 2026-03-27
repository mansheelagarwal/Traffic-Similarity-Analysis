import numpy as np
import pytest
from src.metrics import energy_distance, sliced_wasserstein, compute_mmd, ks_test, bootstrap_ci
from src.preprocessing import bias_variance_decomposition


def test_energy_distance_identical():
    X = np.random.default_rng(0).standard_normal((200, 3))
    assert energy_distance(X, X) == pytest.approx(0.0, abs=1e-10)

def test_energy_distance_separated():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    Y = rng.standard_normal((200, 3)) + 5
    assert energy_distance(X, Y) > 1.0

def test_sliced_wasserstein_symmetry():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 2))
    Y = rng.standard_normal((100, 2)) + 1
    assert sliced_wasserstein(X, Y) == pytest.approx(sliced_wasserstein(Y, X), rel=0.05)

def test_mmd_identical():
    X = np.random.default_rng(2).standard_normal((100, 2))
    assert compute_mmd(X, X) == pytest.approx(0.0, abs=1e-6)

def test_ks_test_structure():
    result = ks_test(np.random.randn(100), np.random.randn(100) + 3)
    assert "ks_stat" in result and "ks_pvalue" in result
    assert 0 <= result["ks_stat"] <= 1
    assert 0 <= result["ks_pvalue"] <= 1

def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((50, 2))
    Y = rng.standard_normal((50, 2)) + 0.5
    result = bootstrap_ci(X, Y, energy_distance, n_resamples=100)
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]

def test_bias_decomp_zero_bias():
    y = np.linspace(0, 100, 200)
    result = bias_variance_decomposition(y, y)
    assert result["bias"] == pytest.approx(0.0, abs=1e-8)
    assert result["mse"] == pytest.approx(0.0, abs=1e-8)

def test_bias_decomp_pcts_sum_to_100():
    rng = np.random.default_rng(4)
    y_true = rng.standard_normal(200) * 10 + 50
    y_pred = y_true * 1.2 + 5
    result = bias_variance_decomposition(y_true, y_pred)
    total = result["bias_pct"] + result["variance_pct"] + result["noise_pct"]
    assert total == pytest.approx(100.0, abs=0.5)
