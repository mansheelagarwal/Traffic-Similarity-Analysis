## The goal is to understand
- Whether StreetLight captures real traffic behavior accurately
- Where it deviates (magnitude, variability, congestion patterns)
- How reliable it is for downstream transportation analysis

---

## Regions Analyzed

The study focuses on four representative regions:

| Region | Description |
|---|---|
| High Congestion | Dense urban corridors with peak-hour bottlenecks |
| Low Congestion | Lighter traffic areas with minimal delay |
| Rural | Low-volume roads with sparse sensor coverage |
| Urban | Mixed-use zones with moderate traffic patterns |

Each region aggregates multiple stations/zones into hourly observations.

---

## What the Project Does

### 1. Data Processing & Feature Engineering
- Aligns PeMS and StreetLight data at the hourly level
- Constructs a shared feature space including:
  - `flow` — vehicles/hour
  - `speed` — mph
  - `density proxy`
  - `dynamics` — change in flow and speed

### 2. Similarity Analysis
Compares distributions using three metrics:
- **Energy Distance**
- **Sliced Wasserstein Distance**
- **Maximum Mean Discrepancy (MMD)**

These metrics measure how similar the overall data distributions are across datasets.

### 3. Error Analysis
- Mean absolute error (flow and speed)
- Aggregated percentage error
- Hourly error trends

### 4. PCA-Based Comparison
PCA is used to:
- Project both datasets into a shared space
- Visually compare distributions
- Analyze variability differences

### 5. Interactive Dashboard
A Dash-based dashboard enables:
- Region selection via map
- Metric and feature space selection
- Visualization of:
  - Region ranking
  - Flow and speed overlays
  - Error profiles
  - PCA comparison

---

## How to Run the Project

### Step 1: Run Similarity Analysis

Run the similarity exploration notebook:
```text
similarity_analysis_exploration.ipynb
```

#### Expected Output Files

After running the notebook, confirm the following files are generated:
```text
data/dashboard_data/
├── summary_df.csv
├── hourly_df.csv
├── feature_df.csv
└── pca_dashboard_points.csv
```

| File | Contents |
|---|---|
| `summary_df.csv` | Region-level similarity metrics and errors |
| `hourly_df.csv` | Hourly aligned PeMS vs StreetLight values |
| `feature_df.csv` | Feature-level comparisons |
| `pca_dashboard_points.csv` | PCA projections for visualization |

> **Note:** These files are required for the dashboard to function.

---

### Step 2: Install Dependencies & Run the Dashboard
```bash
pip install -r requirements.txt
python dashboard/geo_region_dashboard.py
```

### Step 3: Open the Dashboard

Once the server starts, navigate to:
```
http://127.0.0.1:8050
```
