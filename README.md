## The Goal is to Understand
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
## Dashboard Explanation Loom Video 

Here's the link to the dashboard presentation : 

[https://www.loom.com/share/5a43523d38034a6e812c902a9f8ccdd4](https://www.loom.com/share/5a43523d38034a6e812c902a9f8ccdd4)

## How to Run the Project

This project has two stages:

1. Run the analysis notebook to generate dashboard-ready summary files.
2. Run the Dash dashboard to visualize the generated results.

The dashboard depends on generated CSV files. In particular, `pca_dashboard_points.csv` is not included in the repository because it is too large for GitHub. Run the analysis notebook first to create it locally.

---

### Step 1: Install Dependencies

From the project root, run:

```bash
pip install -r requirements.txt
```

---

### Step 2: Run the Similarity Analysis Notebook

Open and run the notebook from top to bottom:

```text
SimilarityAnalysisExploration.ipynb
```

The notebook performs the full analysis pipeline:

- loads PeMS and StreetLight data
- standardizes both sources into a shared hourly schema
- aligns timestamps across datasets
- engineers traffic features such as flow, speed, density proxy, and dynamics
- computes similarity metrics and error summaries
- generates PCA projection points for visualization
- exports dashboard-ready CSV files

After the notebook finishes, it should create:

```text
useful_dashboard_data/
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

> **Note:** `pca_dashboard_points.csv` is generated locally and is not committed to GitHub because of its file size.

---

### Step 3: Run the Dashboard

After the dashboard data files are generated, run:

```bash
python dashboard/geo_region_dashboard.py
```

The Dash server will start locally.

---

### Step 4: Open the Dashboard

Open the following link in your browser:

```text
http://127.0.0.1:8050
```

This link works only while the Dash server is running on your machine.

If port `8050` is already in use, change the port number near the bottom of `dashboard/geo_region_dashboard.py`:

```python
app.run(debug=True, port=8050)
```

---

### Expected Workflow

```text
Run notebook
    ↓
Generate useful_dashboard_data/*.csv
    ↓
Run Dash app
    ↓
Open local dashboard at http://127.0.0.1:8050
```
