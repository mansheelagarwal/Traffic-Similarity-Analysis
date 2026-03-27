# Traffic Similarity Analysis — Technical Upgrades

This branch refactors the original notebook-based analysis into a structured Python package with tested modules and statistically rigorous metrics. The analysis itself is unchanged — this is about making it reproducible, testable, and extensible.

See the `main` branch for background on the dataset, regions, and analytical goals.

---

## What changed

The original code lived entirely in `SimilarityAnalysisExploration.ipynb`. The metric functions, preprocessing steps, and similarity computations are now in `src/` as importable modules with unit tests.

Three things were added on top of the existing analysis:

**Bootstrap confidence intervals** — the original Energy Distance, Wasserstein, and MMD scores were point estimates with no uncertainty quantification. All three now report 95% CIs via percentile bootstrap (500 resamples).

**Bias decomposition** — MAE tells you *how much* StreetLight is off. The Murphy (1988) decomposition tells you *why*: it splits MSE into bias², variance, and noise components. High-congestion PM peak, for example, has ~60% of its error coming from systematic bias rather than random noise — meaning calibration would actually help there.

**Quantile calibration** — a post-hoc correction layer that maps StreetLight flow distributions toward PeMS using quantile transformers fit per region/time slice. Requires n ≥ 30 observations per group; skips groups below that threshold rather than overfitting.

---

## Structure
```
src/
├── metrics.py        # energy_distance, sliced_wasserstein, mmd, ks_test, bootstrap_ci
├── preprocessing.py  # bias_variance_decomposition, compute_decomposition_table
├── models.py         # build_error_model (LightGBM + SHAP), umap embeddings, cross-correlation lag
└── calibration.py    # TrafficCalibrator (quantile mapping, per-group fit)

tests/
└── test_metrics.py   # 8 unit tests covering metrics and bias decomposition

dashboard/
└── geo_region_dashboard.py   # unchanged from main

dashboard_data/               # pre-aggregated outputs from the full dataset
```

---

## Setup
```bash
git clone https://github.com/mansheelagarwal/Traffic-Similarity-Analysis.git
cd Traffic-Similarity-Analysis
git checkout technical-upgrades

python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

---

## Running tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

8 tests, all passing. `metrics.py` at 100% coverage.

---

## Running the dashboard
```bash
python dashboard/geo_region_dashboard.py
```

Open `http://127.0.0.1:8050`. The dashboard reads from `dashboard_data/` which contains outputs pre-computed from the full PeMS/StreetLight dataset.

---

## Using the new modules
```python
import pandas as pd
from src.metrics import bootstrap_ci, energy_distance, ks_test
from src.preprocessing import compute_decomposition_table
from src.calibration import TrafficCalibrator

hourly_df = pd.read_csv("dashboard_data/hourly_df.csv")

# bias decomposition across all groups
decomp = compute_decomposition_table(hourly_df)

# calibration (needs full raw data for meaningful results — see note below)
cal = TrafficCalibrator().fit(hourly_df)
print(cal.calibration_report(hourly_df))
```

---

## A note on dashboard_data

The CSVs in `dashboard_data/` are aggregated summaries — `hourly_df.csv` stores hourly *averages* per group (4–11 rows per group), not raw observations. The dashboard metrics are computed from the full dataset and are correct. The calibration and error model modules are designed for the full raw hourly data from the source Excel files — to run them at scale, re-run `SimilarityAnalysisExploration.ipynb` with the original PeMS/StreetLight files.
