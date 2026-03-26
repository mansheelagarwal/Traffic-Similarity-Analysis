# -*- coding: utf-8 -*-
"""
California Traffic Similarity Dashboard
(cleaned version: temporal removed, heatmap removed, feature mismatch removed)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Paths
# ============================================================
DATA_DIR = Path("useful_dashboard_data")
SUMMARY_PATH = DATA_DIR / "summary_df.csv"
HOURLY_PATH = DATA_DIR / "hourly_df.csv"
PCA_PATH = DATA_DIR / "pca_dashboard_points.csv"

# ============================================================
# Load data
# ============================================================
summary_df = pd.read_csv(SUMMARY_PATH)
hourly_df = pd.read_csv(HOURLY_PATH)

if PCA_PATH.exists():
    pca_df = pd.read_csv(PCA_PATH)
else:
    pca_df = pd.DataFrame(columns=["feature_group", "region", "source", "PC1", "PC2", "evr1", "evr2"])

# normalize strings
def normalize_str_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df

summary_df = normalize_str_cols(summary_df, ["region", "feature_group", "day_type", "time_slice"])
hourly_df = normalize_str_cols(hourly_df, ["region", "day_type", "time_slice"])
pca_df = normalize_str_cols(pca_df, ["region", "feature_group", "source"])

# ============================================================
# Config
# ============================================================
FEATURE_GROUPS = ["all", "flow", "speed", "dynamics"]

FEATURE_LABELS = {
    "all": "All Features",
    "flow": "Flow Only",
    "speed": "Speed Only",
    "dynamics": "Dynamics",
}

DAY_TYPES = ["weekday", "weekend"]

TIME_SLICES = ["am_peak", "midday", "pm_peak", "off_peak"]
TIME_SLICE_LABELS = {
    "am_peak": "AM Peak",
    "midday": "Midday",
    "pm_peak": "PM Peak",
    "off_peak": "Off-Peak",
}

METRICS = {
    "energy": "Energy Distance",
    "wasserstein": "Sliced Wasserstein",
    "mmd": "MMD",
    "flow_error_abs": "Flow Error (Absolute)",
    "flow_error_pct": "Flow Error (% of PeMS mean)",
    "speed_error_abs": "Speed Error (Absolute)",
    "speed_error_pct": "Speed Error (%)",
}

REGION_ORDER = ["high", "low", "rural", "urban"]
REGION_LABELS = {
    "high": "High Congestion",
    "low": "Low Congestion",
    "rural": "Rural",
    "urban": "Urban",
}

# approximate study-area centers from your screenshots
REGION_MAP = pd.DataFrame([
    {"region": "high",  "label": "High Congestion", "lat": 34.155, "lon": -118.365, "color": "#d62728"},
    {"region": "low",   "label": "Low Congestion",  "lat": 35.640, "lon": -120.670, "color": "#1f77b4"},
    {"region": "rural", "label": "Rural",           "lat": 38.500, "lon": -121.760, "color": "#2ca02c"},
    {"region": "urban", "label": "Urban",           "lat": 32.715, "lon": -117.145, "color": "#ff7f0e"},
])

# approximate corridor lines
REGION_LINES = {
    "high":  [(34.215, -118.490), (34.185, -118.450), (34.145, -118.395), (34.120, -118.345), (34.080, -118.305)],
    "low":   [(35.790, -120.720), (35.710, -120.690), (35.620, -120.670), (35.520, -120.690), (35.430, -120.730)],
    "rural": [(38.610, -121.560), (38.565, -121.650), (38.500, -121.760), (38.420, -121.860), (38.335, -121.955)],
    "urban": [(32.760, -117.220), (32.735, -117.185), (32.715, -117.145), (32.690, -117.105), (32.665, -117.070)],
}

# defaults
default_region = "high"
default_feature_group = "all"
default_metric = "energy"
default_day_type = "weekday"
default_time_slice = "am_peak"

# ============================================================
# Helpers
# ============================================================
def stat_card(title, value):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted", style={"fontSize": "0.85rem"}),
                html.H4(value, className="mb-0"),
            ]
        ),
        className="shadow-sm",
    )

def nice_region(region):
    return REGION_LABELS.get(region, region)

def safe_fig(title="No data"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=320,
        margin=dict(l=35, r=20, t=50, b=35),
    )
    return fig

def get_region_metric_slice(feature_group, day_type, time_slice):
    return summary_df[
        (summary_df["feature_group"] == feature_group) &
        (summary_df["day_type"] == day_type) &
        (summary_df["time_slice"] == time_slice)
    ].copy()

def compute_best_feature_group(day_type, time_slice):
    d = summary_df[
        (summary_df["day_type"] == day_type) &
        (summary_df["time_slice"] == time_slice)
    ].copy()

    if d.empty:
        return None

    rows = []
    for fg in FEATURE_GROUPS:
        sub = d[d["feature_group"] == fg]
        if sub.empty:
            continue
        score = sub[["energy", "wasserstein", "mmd"]].mean(axis=1).mean()
        rows.append((fg, score))

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x[1])
    return rows[0][0]

# ============================================================
# Plots
# ============================================================
def build_map(metric_df, metric, selected_region):
    if metric_df.empty:
        return safe_fig("No map data")

    d = metric_df.copy()
    d = REGION_MAP.merge(d[["region", metric, "n_pairs"]], on="region", how="left")

    fig = go.Figure()

    # corridor lines
    for _, row in d.iterrows():
        r = row["region"]
        coords = REGION_LINES[r]
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        line_width = 10 if r == selected_region else 6

        fig.add_trace(
            go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=line_width, color=row["color"]),
                name=row["label"],
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # clickable markers
    fig.add_trace(
        go.Scattermapbox(
            lat=d["lat"],
            lon=d["lon"],
            mode="markers+text",
            marker=dict(
                size=[22 if r == selected_region else 16 for r in d["region"]],
                color=d["color"],
                opacity=0.95,
            ),
            text=d["label"],
            textposition="top center",
            customdata=np.stack(
                [
                    d["region"],
                    d[metric].fillna(np.nan),
                    d["n_pairs"].fillna(np.nan),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                + METRICS[metric]
                + ": %{customdata[1]:.4f}<br>"
                + "Matched hourly observations: %{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"California Study Areas — {METRICS[metric]}",
        template="plotly_white",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=35.45, lon=-119.35),
            zoom=5.4,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=560,
        clickmode="event+select",
        annotations=[
            dict(
                text="Approximate geographic study corridors based on the 4 selected areas",
                x=0.5, y=0.02, xref="paper", yref="paper",
                showarrow=False, font=dict(size=11, color="gray")
            )
        ],
    )
    return fig

def build_ranking_chart(metric_df, metric, selected_region):
    if metric_df.empty:
        return safe_fig("No ranking data")

    d = metric_df.copy().sort_values(metric, ascending=False)
    d["region_label"] = d["region"].map(REGION_LABELS)
    d["selected"] = np.where(d["region"] == selected_region, "Selected", "Other")

    fig = px.bar(
        d,
        x="region_label",
        y=metric,
        color="selected",
        color_discrete_map={"Selected": "#d62728", "Other": "#1f77b4"},
        hover_data=["n_pairs"],
        title=f"Region Ranking — {METRICS[metric]}",
    )
    fig.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=35, r=20, t=45, b=35),
        showlegend=False,
    )
    fig.update_xaxes(title="")
    return fig

def build_flow_overlay(region, day_type):
    d = hourly_df[
        (hourly_df["region"] == region) &
        (hourly_df["day_type"] == day_type)
    ].sort_values("hour")

    if d.empty:
        return safe_fig("No hourly flow data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["hour"], y=d["flow_pems"], mode="lines+markers", name="PeMS"))
    fig.add_trace(go.Scatter(x=d["hour"], y=d["flow_sl"], mode="lines+markers", name="StreetLight"))
    fig.update_layout(
        title=f"Hourly Flow Overlay — {nice_region(region)} ({day_type})",
        xaxis_title="Hour of day",
        yaxis_title="Flow",
        template="plotly_white",
        height=300,
        margin=dict(l=35, r=20, t=50, b=35),
    )
    return fig

def build_speed_overlay(region, day_type):
    d = hourly_df[
        (hourly_df["region"] == region) &
        (hourly_df["day_type"] == day_type)
    ].sort_values("hour")

    if d.empty:
        return safe_fig("No hourly speed data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["hour"], y=d["speed_pems"], mode="lines+markers", name="PeMS"))
    fig.add_trace(go.Scatter(x=d["hour"], y=d["speed_sl"], mode="lines+markers", name="StreetLight"))
    fig.update_layout(
        title=f"Hourly Speed Overlay — {nice_region(region)} ({day_type})",
        xaxis_title="Hour of day",
        yaxis_title="Speed",
        template="plotly_white",
        height=300,
        margin=dict(l=35, r=20, t=50, b=35),
    )
    return fig

def build_error_profile(region, day_type):
    d = hourly_df[
        (hourly_df["region"] == region) &
        (hourly_df["day_type"] == day_type)
    ].sort_values("hour")

    if d.empty:
        return safe_fig("No hourly error data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["hour"], y=d["flow_abs_error"], mode="lines+markers", name="Flow abs error"))
    fig.add_trace(go.Scatter(x=d["hour"], y=d["speed_abs_error"], mode="lines+markers", name="Speed abs error"))
    fig.add_trace(go.Scatter(x=d["hour"], y=d["flow_pct_error"], mode="lines", name="Flow % error", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=d["hour"], y=d["speed_pct_error"], mode="lines", name="Speed % error", line=dict(dash="dash")))
    fig.update_layout(
        title=f"Hourly Error Profile — {nice_region(region)} ({day_type})",
        xaxis_title="Hour of day",
        yaxis_title="Error",
        template="plotly_white",
        height=320,
        margin=dict(l=35, r=20, t=50, b=35),
    )
    return fig

def build_pca_plot(region, feature_group):
    if pca_df.empty:
        return safe_fig("PCA file not found")

    d = pca_df[
        (pca_df["region"] == region) &
        (pca_df["feature_group"] == feature_group)
    ].copy()

    if d.empty:
        return safe_fig("No PCA data for selected region / feature group")

    d["source_label"] = d["source"].map({
        "pems": "PeMS",
        "streetlight": "StreetLight",
    }).fillna(d["source"])

    evr1 = d["evr1"].dropna().iloc[0] if d["evr1"].notna().any() else np.nan
    evr2 = d["evr2"].dropna().iloc[0] if d["evr2"].notna().any() else np.nan

    title = f"PCA — {nice_region(region)} | {FEATURE_LABELS.get(feature_group, feature_group)}"
    if pd.notna(evr1) and pd.notna(evr2):
        title += f" | PC1 {evr1:.1%}, PC2 {evr2:.1%}"

    fig = px.scatter(
        d,
        x="PC1",
        y="PC2",
        color="source_label",
        opacity=0.35,
        title=title,
        color_discrete_map={
            "PeMS": "#1f77b4",
            "StreetLight": "#d62728",
        },
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=35, r=20, t=50, b=35),
        legend_title_text="Source",
    )
    return fig

# ============================================================
# App
# ============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.Br(),
        html.H2("California Traffic Data Similarity Dashboard"),
        html.P(
            "Interactive comparison of PeMS and StreetLight across the four California study areas. "
            "Select a metric, feature space, and time slice; click the map or ranking chart to drill into a region."
        ),

        dcc.Store(id="selected-region-store", data=default_region),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Feature space"),
                        dcc.Dropdown(
                            id="feature-dropdown",
                            options=[{"label": FEATURE_LABELS[x], "value": x} for x in FEATURE_GROUPS],
                            value=default_feature_group,
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Metric"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=[{"label": v, "value": k} for k, v in METRICS.items()],
                            value=default_metric,
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Day type"),
                        dcc.Dropdown(
                            id="daytype-dropdown",
                            options=[{"label": x.title(), "value": x} for x in DAY_TYPES],
                            value=default_day_type,
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Time slice"),
                        dcc.Dropdown(
                            id="timeslice-dropdown",
                            options=[{"label": TIME_SLICE_LABELS[x], "value": x} for x in TIME_SLICES],
                            value=default_time_slice,
                            clearable=False,
                        ),
                    ],
                    md=3,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Alert(id="best-feature-alert", color="success", className="py-2"),
                    md=12,
                )
            ],
            className="mb-2",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id="geo-map", config={"displayModeBar": False})
                        ),
                        className="shadow-sm",
                    ),
                    md=7,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="card-region"), md=6),
                                dbc.Col(html.Div(id="card-rank"), md=6),
                            ],
                            className="mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="card-metric"), md=6),
                                dbc.Col(html.Div(id="card-pairs"), md=6),
                            ],
                            className="mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.Div(id="card-flow"), md=6),
                                dbc.Col(html.Div(id="card-speed"), md=6),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="ranking-chart")),
                            className="shadow-sm",
                        ),
                    ],
                    md=5,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(id="flow-overlay-chart")), className="shadow-sm"),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(id="speed-overlay-chart")), className="shadow-sm"),
                    md=6,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(id="error-profile-chart")), className="shadow-sm"),
                    md=12,
                ),
            ],
            className="mb-3",
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(dbc.CardBody(dcc.Graph(id="pca-chart")), className="shadow-sm"),
                    md=12,
                ),
            ],
            className="mb-3",
        ),
        html.Br(),
    ],
)

# ============================================================
# Region click handling
# ============================================================
@app.callback(
    Output("selected-region-store", "data"),
    Input("geo-map", "clickData"),
    Input("ranking-chart", "clickData"),
    State("selected-region-store", "data"),
    prevent_initial_call=False,
)
def update_selected_region(map_click, rank_click, current_region):
    if map_click and "points" in map_click and len(map_click["points"]) > 0:
        pt = map_click["points"][0]
        if "customdata" in pt and pt["customdata"] is not None:
            region = pt["customdata"][0]
            if region in REGION_ORDER:
                return region

    if rank_click and "points" in rank_click and len(rank_click["points"]) > 0:
        xval = rank_click["points"][0]["x"]
        reverse_labels = {v: k for k, v in REGION_LABELS.items()}
        region = reverse_labels.get(xval)
        if region in REGION_ORDER:
            return region

    return current_region or default_region

# ============================================================
# Main redraw
# ============================================================
@app.callback(
    Output("best-feature-alert", "children"),
    Output("geo-map", "figure"),
    Output("ranking-chart", "figure"),
    Output("flow-overlay-chart", "figure"),
    Output("speed-overlay-chart", "figure"),
    Output("error-profile-chart", "figure"),
    Output("pca-chart", "figure"),
    Output("card-region", "children"),
    Output("card-rank", "children"),
    Output("card-metric", "children"),
    Output("card-pairs", "children"),
    Output("card-flow", "children"),
    Output("card-speed", "children"),
    Input("feature-dropdown", "value"),
    Input("metric-dropdown", "value"),
    Input("daytype-dropdown", "value"),
    Input("timeslice-dropdown", "value"),
    Input("selected-region-store", "data"),
)
def redraw(feature_group, metric, day_type, time_slice, selected_region):
    best_fg = compute_best_feature_group(day_type, time_slice)
    if best_fg is None:
        best_msg = "No feature-space recommendation available for the selected slice."
    else:
        best_msg = f"Auto-selected best feature space: {FEATURE_LABELS[best_fg]}"

    metric_df = get_region_metric_slice(feature_group, day_type, time_slice)

    if metric_df.empty:
        empty = safe_fig("No data for selected slice")
        msg = stat_card("Status", "No data")
        return (
            best_msg,
            empty, empty, empty, empty, empty, empty,
            msg, html.Div(), html.Div(), html.Div(), html.Div(), html.Div(),
        )

    available_regions = metric_df["region"].tolist()
    if selected_region not in available_regions:
        selected_region = available_regions[0]

    row = metric_df[metric_df["region"] == selected_region].iloc[0]

    map_fig = build_map(metric_df, metric, selected_region)
    ranking_fig = build_ranking_chart(metric_df, metric, selected_region)
    flow_fig = build_flow_overlay(selected_region, day_type)
    speed_fig = build_speed_overlay(selected_region, day_type)
    error_fig = build_error_profile(selected_region, day_type)
    pca_fig = build_pca_plot(selected_region, feature_group)

    rank_order = metric_df.sort_values(metric, ascending=True)["region"].tolist()
    region_rank = rank_order.index(selected_region) + 1 if selected_region in rank_order else 1

    card_region = stat_card("Selected region", nice_region(selected_region))
    card_rank = stat_card("State rank", f"{region_rank} / {len(metric_df)}")
    card_metric = stat_card(METRICS[metric], f"{row[metric]:.4f}")
    card_pairs = stat_card("Matched hourly observations", f"{int(row['n_pairs'])}")
    card_flow = stat_card("Mean flow abs error", f"{row['flow_error_abs']:.2f}")
    card_speed = stat_card("Mean speed abs error", f"{row['speed_error_abs']:.2f}")

    return (
        best_msg,
        map_fig,
        ranking_fig,
        flow_fig,
        speed_fig,
        error_fig,
        pca_fig,
        card_region,
        card_rank,
        card_metric,
        card_pairs,
        card_flow,
        card_speed,
    )

if __name__ == "__main__":
    app.run(debug=True, port=8050)
