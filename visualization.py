"""
visualization.py – Plotly visualisations for the Juice Plant SimPy Simulation
==============================================================================
All functions accept pandas DataFrames produced by model.py / experiment.py
and return plotly.graph_objects.Figure objects.

Functions
---------
plot_container_levels(container_logs)
    Line chart: apple_delivery, apple_buffer, raw_juice_buffer, storage_tank levels over time.

plot_process_gantt(process_logs)
    Gantt-style activity chart: horizontal bars show when each step was ACTIVE.

plot_sweep_results(sweep_df, x_col, metrics)
    Line chart with error bands for a parameter sweep.

plot_seed_variation(summary_df, metric)
    Histogram + box plot showing seed-to-seed variability of one KPI.

plot_seed_container_variation(all_container_logs, column)
    Multi-line chart of a container level across multiple seeds.
"""

from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Colour palette used consistently across plots
_STEP_COLORS = {
    "arrival":     "#94a3b8",
    "wash":        "#38bdf8",
    "shred":       "#fb923c",
    "press":       "#a78bfa",
    "concentrate": "#34d399",
}

_CONTAINER_COLORS = {
    "apple_delivery":    "#facc15",
    "apple_buffer":     "#fb923c",
    "raw_juice_buffer": "#38bdf8",
    "storage_tank":     "#34d399",
}

_TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# 1. Container levels over time
# ---------------------------------------------------------------------------

def plot_container_levels(container_logs: pd.DataFrame) -> go.Figure:
    """Line chart of all three container / buffer levels over simulation time."""
    if container_logs.empty:
        return go.Figure()

    fig = go.Figure()
    labels = {
        "apple_delivery":    "Apple Delivery Buffer [apples]",
        "apple_buffer":     "Apple Buffer [shredded apples]",
        "raw_juice_buffer": "Raw Juice Buffer [liters]",
        "storage_tank":     "Storage Tank [liters]",
    }

    for col, label in labels.items():
        if col not in container_logs.columns:
            continue
        fig.add_trace(go.Scatter(
            x=container_logs["time"],
            y=container_logs[col],
            mode="lines",
            name=label,
            line=dict(color=_CONTAINER_COLORS[col], width=2),
        ))

    fig.update_layout(
        title="Container & Buffer Levels over Time",
        xaxis_title="Simulation Time [min]",
        yaxis_title="Level",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Gantt-style process activity chart
# ---------------------------------------------------------------------------

def plot_process_gantt(process_logs: pd.DataFrame) -> go.Figure:
    """
    Line chart showing the number of active machines per process step over simulation time.
    Calculates active count (+1 for started, -1 for finished) and plots as a step function.
    """
    if process_logs.empty:
        return go.Figure()

    steps = ["wash and shred", "press", "concentrate"]
    fig = go.Figure()

    for step in steps:
        df_step = process_logs[process_logs["step"] == step].copy()
        if df_step.empty:
            continue

        # +1 for started, -1 for finished
        df_step["change"] = df_step["event"].map({"started": 1, "finished": -1}).fillna(0)

        # Sum changes at the exact same time to avoid up/down spikes
        df_grouped = df_step.groupby("time", as_index=False)["change"].sum()
        df_grouped = df_grouped.sort_values("time")

        # Filter out points where net change is 0 (started and finished at same time)
        df_grouped = df_grouped[df_grouped["change"] != 0].copy()

        if df_grouped.empty:
            continue

        # Cumulative sum to get active machine count
        df_grouped["active"] = df_grouped["change"].cumsum()

        # Add a starting point at time 0 with 0 active machines if it doesn't exist
        if df_grouped["time"].iloc[0] > 0:
            start_row = pd.DataFrame({"time": [0], "change": [0], "active": [0]})
            df_grouped = pd.concat([start_row, df_grouped], ignore_index=True)
            
        # Add a final point extending to the maximum time in the logs to ensure the step function goes to the end
        max_time = process_logs["time"].max()
        if not df_grouped.empty and df_grouped["time"].iloc[-1] < max_time:
            end_row = pd.DataFrame({"time": [max_time], "change": [0], "active": [df_grouped["active"].iloc[-1]]})
            df_grouped = pd.concat([df_grouped, end_row], ignore_index=True)

        fig.add_trace(go.Scatter(
            x=df_grouped["time"],
            y=df_grouped["active"],
            mode="lines",
            name=step.capitalize(),
            line=dict(color=_STEP_COLORS.get(step, "grey"), width=2, shape="hv"),
        ))

    fig.update_layout(
        title="Active Process Steps over Time",
        xaxis_title="Simulation Time [min]",
        yaxis_title="Active Machines",
        template=_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Parameter sweep results
# ---------------------------------------------------------------------------

def plot_sweep_results(
    sweep_df: pd.DataFrame,
    x_col: str,
    metrics: Optional[List[str]] = None,
) -> go.Figure:
    """
    Line chart with error bands for a parameter sweep.
    If multiple seeds were run per value, mean ± std is shown.
    """
    if sweep_df.empty or x_col not in sweep_df.columns:
        return go.Figure()

    if metrics is None:
        # default KPIs to show
        metrics = [
            "press_batches_finished",
            "concentrate_quantity_out",
            "final_storage_liters",
            "avg_apple_buffer",
        ]
    metrics = [m for m in metrics if m in sweep_df.columns]
    if not metrics:
        return go.Figure()

    n = len(metrics)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[m.replace("_", " ").title() for m in metrics])

    has_seeds = "seed" in sweep_df.columns and sweep_df["seed"].nunique() > 1

    for idx, metric in enumerate(metrics):
        r = idx // cols + 1
        c = idx % cols + 1

        if has_seeds:
            stats = sweep_df.groupby(x_col)[metric].agg(["mean", "std"]).reset_index()
            fig.add_trace(go.Scatter(
                x=stats[x_col], y=stats["mean"],
                mode="lines+markers",
                name=metric,
                line=dict(color=px.colors.qualitative.Plotly[idx % 10]),
                error_y=dict(type="data", array=stats["std"], visible=True),
                showlegend=False,
            ), row=r, col=c)
        else:
            fig.add_trace(go.Scatter(
                x=sweep_df[x_col], y=sweep_df[metric],
                mode="lines+markers",
                name=metric,
                line=dict(color=px.colors.qualitative.Plotly[idx % 10]),
                showlegend=False,
            ), row=r, col=c)

        fig.update_xaxes(title_text=x_col.replace("_", " "), row=r, col=c)

    fig.update_layout(
        title=f"Parameter Sweep: {x_col}",
        template=_TEMPLATE,
        height=350 * rows,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Seed variation – distribution of one KPI
# ---------------------------------------------------------------------------

def plot_seed_variation(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Histogram + horizontal box plot showing run-to-run variability of `metric`.
    """
    if summary_df.empty or metric not in summary_df.columns:
        return go.Figure()

    data = summary_df[metric].dropna()
    metric_title = metric.replace("_", " ").title()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.7, 0.3])

    fig.add_trace(
        go.Histogram(x=data, name="Frequency", showlegend=False,
                     marker_color="#60a5fa", opacity=0.8),
        row=1, col=1,
    )
    fig.add_trace(
        go.Box(x=data, name="", showlegend=False, boxpoints="all",
               jitter=0.4, marker_color="#f472b6", line_color="#db2777"),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"Seed Variation – {metric_title}",
        template=_TEMPLATE,
        showlegend=False,
    )
    fig.update_xaxes(title_text=metric_title, row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# 5. Multi-seed container level chart
# ---------------------------------------------------------------------------

def plot_seed_container_variation(
    all_container_logs: pd.DataFrame,
    column: str = "storage_tank",
) -> go.Figure:
    """
    Multi-line chart of a container level across multiple seeds.
    """
    if all_container_logs.empty or column not in all_container_logs.columns:
        return go.Figure()

    col_label = column.replace("_", " ").title()
    df = all_container_logs

    # Downsample if data is very large
    if len(df) > 150_000:
        df = df.iloc[::max(1, len(df) // 150_000)].copy()

    n_seeds = df["seed"].nunique()
    fig = go.Figure()

    for seed in sorted(df["seed"].unique()):
        sub = df[df["seed"] == seed]
        fig.add_trace(go.Scatter(
            x=sub["time"], y=sub[column],
            mode="lines",
            name=f"Seed {seed}",
            opacity=0.6,
            line=dict(width=1),
            showlegend=bool(n_seeds <= 20),
        ))

    fig.update_layout(
        title=f"{col_label} Variation across Seeds",
        xaxis_title="Simulation Time [min]",
        yaxis_title=col_label,
        template=_TEMPLATE,
        showlegend=(n_seeds <= 20),
    )
    return fig
