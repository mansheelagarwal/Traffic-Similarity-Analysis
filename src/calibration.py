import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

MIN_GROUP_SIZE = 30  # skip groups too small to calibrate reliably

class TrafficCalibrator:
    def __init__(self, n_quantiles: int = 100):
        self.n_quantiles = n_quantiles
        self._models: dict = {}

    def fit(self, hourly_df: pd.DataFrame) -> "TrafficCalibrator":
        skipped = 0
        for key, g in hourly_df.groupby(["region", "time_slice", "day_type"]):
            if len(g) < MIN_GROUP_SIZE:
                skipped += 1
                continue
            qt = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(g)),
                output_distribution="normal", random_state=42
            )
            qt.fit(g[["flow_sl"]].values)
            pems_qt = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(g)),
                output_distribution="normal", random_state=42
            )
            pems_qt.fit(g[["flow_pems"]].values)
            self._models[key] = {"sl_qt": qt, "pems_qt": pems_qt}
        print(f"Calibrator fitted on {len(self._models)} groups, skipped {skipped} (n < {MIN_GROUP_SIZE})")
        return self

    def transform(self, hourly_df: pd.DataFrame) -> pd.DataFrame:
        out = hourly_df.copy()
        out["flow_sl_calibrated"] = np.nan
        for key, g_idx in out.groupby(["region", "time_slice", "day_type"]).groups.items():
            if key not in self._models:
                continue
            m = self._models[key]
            sl_vals = out.loc[g_idx, "flow_sl"].values.reshape(-1, 1)
            sl_uniform = m["sl_qt"].transform(sl_vals)
            sl_calibrated = m["pems_qt"].inverse_transform(sl_uniform)
            out.loc[g_idx, "flow_sl_calibrated"] = sl_calibrated.ravel()
        return out

    def calibration_report(self, hourly_df: pd.DataFrame) -> pd.DataFrame:
        cal = self.transform(hourly_df)
        rows = []
        for key, g in cal.groupby(["region", "time_slice", "day_type"]):
            if key not in self._models:
                continue  # skip groups that weren't fitted
            mae_before = np.mean(np.abs(g["flow_pems"] - g["flow_sl"]))
            mae_after = np.mean(np.abs(g["flow_pems"] - g["flow_sl_calibrated"]))
            rows.append({
                "region": key[0], "time_slice": key[1], "day_type": key[2],
                "n": len(g),
                "mae_before": round(mae_before, 2),
                "mae_after": round(mae_after, 2),
                "improvement_pct": round(100 * (mae_before - mae_after) / mae_before, 1)
            })
        return pd.DataFrame(rows).sort_values("improvement_pct", ascending=False)
