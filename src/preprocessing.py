import numpy as np
import pandas as pd


def bias_variance_decomposition(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bias = np.mean(y_pred) - np.mean(y_true)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    var_true = np.var(y_true)
    mse = np.mean((y_pred - y_true) ** 2)
    bias_sq = bias ** 2
    variance_component = (np.std(y_pred) - corr * np.std(y_true)) ** 2
    noise = (1 - corr ** 2) * var_true
    return {
        "mse": float(mse),
        "bias": float(bias),
        "bias_sq": float(bias_sq),
        "variance_component": float(variance_component),
        "noise": float(noise),
        "bias_pct": float(100 * bias_sq / mse) if mse > 0 else 0,
        "variance_pct": float(100 * variance_component / mse) if mse > 0 else 0,
        "noise_pct": float(100 * noise / mse) if mse > 0 else 0,
    }


def compute_decomposition_table(hourly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, ts, dt), g in hourly_df.groupby(["region", "time_slice", "day_type"]):
        flow = bias_variance_decomposition(g["flow_pems"].values, g["flow_sl"].values)
        speed = bias_variance_decomposition(g["speed_pems"].values, g["speed_sl"].values)
        rows.append({
            "region": region, "time_slice": ts, "day_type": dt,
            **{f"flow_{k}": v for k, v in flow.items()},
            **{f"speed_{k}": v for k, v in speed.items()},
        })
    return pd.DataFrame(rows)
