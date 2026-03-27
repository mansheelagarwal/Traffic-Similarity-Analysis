import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, ks_2samp, bootstrap
from sklearn.metrics.pairwise import rbf_kernel


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    XY = cdist(X, Y); XX = cdist(X, X); YY = cdist(Y, Y)
    return float(2 * np.mean(XY) - np.mean(XX) - np.mean(YY))


def sliced_wasserstein(X: np.ndarray, Y: np.ndarray, n_proj: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    projections = rng.standard_normal((n_proj, X.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(X @ p, Y @ p) for p in projections]))


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K_XX = rbf_kernel(X, X, gamma); K_YY = rbf_kernel(Y, Y, gamma); K_XY = rbf_kernel(X, Y, gamma)
    return float(np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))


def ks_test(x: np.ndarray, y: np.ndarray) -> dict:
    stat, p = ks_2samp(x, y)
    return {"ks_stat": float(stat), "ks_pvalue": float(p)}


def bootstrap_ci(X: np.ndarray, Y: np.ndarray, metric_fn, n_resamples: int = 500,
                 confidence_level: float = 0.95, seed: int = 0) -> dict:
    """
    Bootstrap CI by resampling row indices and applying metric_fn(X[idx], Y[idx]).
    Works for any metric that takes two 2D arrays.
    """
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y))
    estimates = []
    for _ in range(n_resamples):
        idx_x = rng.integers(0, len(X), size=n)
        idx_y = rng.integers(0, len(Y), size=n)
        estimates.append(metric_fn(X[idx_x], Y[idx_y]))
    estimates = np.array(estimates)
    alpha = 1 - confidence_level
    return {
        "estimate": metric_fn(X, Y),
        "ci_low": float(np.percentile(estimates, 100 * alpha / 2)),
        "ci_high": float(np.percentile(estimates, 100 * (1 - alpha / 2))),
    }
