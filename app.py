"""
app.py – Streamlit Dashboard for the Juice Plant SimPy Simulation
=================================================================
Run with:
    uv run streamlit run CaseStudy/app.py

Tabs:
  1. Single Run       – run one simulation, explore logs and all charts
  2. Seed Variation   – N seeds, stochastic sensitivity plots
  3. Parameter Sweep  – vary one parameter (+ optional seeds), see KPI trends
  4. Combined         – sweep × seed variation with error bands
"""

import sys
import os

# Make sure the CaseStudy folder is importable regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import replace

from model import Params
from experiment import (
    run_single,
    run_seed_variation,
    run_parameter_sweep,
)
from visualization import (
    plot_container_levels,
    plot_process_gantt,
    plot_sweep_results,
    plot_seed_variation,
    plot_seed_container_variation,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🍎 Juice Plant Simulation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍎 Juice Plant – Discrete Event Simulation")
st.caption("SimPy-based production simulation: apples → washing → shredding → pressing → concentration → storage")

# ---------------------------------------------------------------------------
# Sidebar – shared parameters
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Simulation Parameters")

with st.sidebar.expander("Simulation", expanded=True):
    sim_hours          = st.number_input("Sim Duration [hours]",     1, 20000, 72, 24)
    monitor_interval   = st.number_input("Monitor Interval [min]",   0.005, 1000.0, 1.0, 0.5)

with st.sidebar.expander("Arrival"):
    interarrival_min     = st.number_input("Mean Inter-Arrival [min]", 1, 50000, 300, 100)
    deterministic_arrival = st.checkbox("Deterministic Arrivals", value=True,
                                         help="Checked → fixed Inter-Arrival-Time; unchecked → random exponential")
    batch_size_min       = st.number_input("Batch Size Min [apples]",  50,  1000000,  25000, 50)
    batch_size_max       = st.number_input("Batch Size Max [apples]",  50,  1000000,  25000, 50)

with st.sidebar.expander("Wash & Shred"):
    wash_shred_lines = st.number_input("Wash & Shred Lines",           1, 10, 1)
    wash_shred_rate  = st.number_input("Wash & Shred Rate [apples/min]", 50.0, 2000.0, 120.0, 50.0)
    # entity_batch_size = st.number_input("Wash/Shred Batch Size [apples]", 10, 100000, 100, 100)

with st.sidebar.expander("Pressing"):
    presses            = st.number_input("Press Units",              1, 10, 1)
    press_batch_size   = st.number_input("Press Batch Size [apples]",100, 50000, 7000, 100)
    press_time         = st.number_input("Press Time [min]",         1.0, 500.0, 90.0, 1.0)
    liters_per_apple_mean = st.number_input("Yield Mean [L/apple]",  0.0, 1.0, 0.12, 0.01)
    liters_per_apple_sd   = st.number_input("Yield Std Dev [L/apple]",0.0, 1.0, 0.01, 0.01)

with st.sidebar.expander("Concentration"):
    concentrators      = st.number_input("Concentrator Units",       1, 10, 1)
    conc_rate          = st.number_input("Conc. Rate [L/min]",       1.0, 500.0, 8.0, 1.0)
    conc_factor        = st.slider("Concentration Factor",           0.05, 1.0, 0.30, 0.05)
    # conc_chunk         = st.number_input("Chunk Size [L]",           1.0, 100000.0, 10.0, 50.0)

base_params = Params(
    interarrival_min=interarrival_min,
    deterministic_arrival=bool(deterministic_arrival),
    batch_size_min=int(batch_size_min),
    batch_size_max=int(batch_size_max),
    wash_shred_lines=int(wash_shred_lines),
    wash_shred_rate_apples_per_min=wash_shred_rate,
    entity_batch_size_apples=10, # entity_batch_size
    presses=int(presses),
    press_batch_size_apples=int(press_batch_size),
    press_time_min=press_time,
    liters_per_apple_mean=liters_per_apple_mean,
    liters_per_apple_sd=liters_per_apple_sd,
    concentrators=int(concentrators),
    conc_rate_liters_per_min=conc_rate,
    conc_factor=conc_factor,
    conc_chunk_liters=10, # conc_chunk
    sim_time_min=sim_hours * 60,
    monitor_interval_min=monitor_interval,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Run",
    "🎲 Seed Variation",
    "🔬 Parameter Sweep",
    "🧪 Combined Experiment",
])


# ============================================================
# TAB 1 – Single Run
# ============================================================
with tab1:
    st.header("Single Simulation Run")
    col_seed, col_btn = st.columns([1, 3])
    with col_seed:
        seed = st.number_input("Random Seed", value=42, step=1, key="t1_seed")
    with col_btn:
        st.write("")  # spacer
        run_btn = st.button("▶ Run Simulation", key="btn_single", type="primary")

    if run_btn:
        p = replace(base_params, random_seed=int(seed))
        with st.spinner("Running simulation…"):
            result = run_single(p)

        proc_df = result["process_logs"]
        cont_df = result["container_logs"]

        # --- KPI tiles ------------------------------------------------------
        arrivals  = proc_df[(proc_df["step"] == "arrival") & (proc_df["event"] == "arrived")]
        presses_f = proc_df[(proc_df["step"] == "press")   & (proc_df["event"] == "finished")]
        conc_f    = proc_df[(proc_df["step"] == "concentrate") & (proc_df["event"] == "finished")]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Batches Arrived",         len(arrivals))
        k2.metric("Apples Arrived",          f"{arrivals['quantity'].sum():,.0f}")
        k3.metric("Press Batches Completed", len(presses_f))
        k4.metric("Conc. Cycles Completed",  len(conc_f))
        k5.metric("Final Storage [L]",
                  f"{cont_df['storage_tank'].iloc[-1]:.1f}" if not cont_df.empty else "–")

        st.divider()

        # --- Charts ---------------------------------------------------------
        st.subheader("Container & Buffer Levels")
        st.plotly_chart(plot_container_levels(cont_df), width='stretch')

        st.subheader("Process Step Activity")
        st.plotly_chart(plot_process_gantt(proc_df), width='stretch')

        # --- Raw data -------------------------------------------------------
        with st.expander("📋 Process Logs (raw)"):
            st.dataframe(proc_df, width='stretch')
        with st.expander("📋 Container Logs (raw)"):
            st.dataframe(cont_df, width='stretch')


# ============================================================
# TAB 2 – Seed Variation
# ============================================================
with tab2:
    st.header("Stochastic Seed Variation")
    st.markdown("Run the same parameter set with multiple random seeds to evaluate variance.")

    n_seeds = st.slider("Number of Seeds", 3, 50, 10, key="t2_seeds")
    seeds_list = list(range(n_seeds))

    if st.button("▶ Run Seed Variation", key="btn_seed", type="primary"):
        with st.spinner(f"Running {n_seeds} simulations…"):
            results = run_seed_variation(base_params, seeds_list)

        summary_df  = results["summary"]
        all_proc    = results["all_process_logs"]
        all_cont    = results["all_container_logs"]

        st.success(f"Completed {n_seeds} runs.")

        # KPI distributions
        st.subheader("KPI Distributions across Seeds")
        kpis = [
            "press_batches_finished",
            "concentrate_quantity_out",
            "final_storage_liters",
            "avg_apple_buffer",
        ]
        cols = st.columns(2)
        for i, kpi in enumerate(kpis):
            if kpi in summary_df.columns:
                with cols[i % 2]:
                    st.plotly_chart(plot_seed_variation(summary_df, kpi),
                                    width='stretch')

        st.subheader("Storage Tank Level across Seeds")
        st.plotly_chart(
            plot_seed_container_variation(all_cont, "storage_tank"),
            width='stretch',
        )

        st.subheader("Apple Buffer Level across Seeds")
        st.plotly_chart(
            plot_seed_container_variation(all_cont, "apple_buffer"),
            width='stretch',
        )

        with st.expander("📋 Summary Table"):
            st.dataframe(summary_df.describe(), width='stretch')


# ============================================================
# TAB 3 – Parameter Sweep
# ============================================================
with tab3:
    st.header("Parameter Sweep")
    st.markdown("Vary one simulation parameter and observe its impact on KPIs.")

    param_options = {
        # "Washers":               "washers",
        # "Shredder Lines":        "shredders",
        "Wash & Shred Lines":    "wash_shred_lines",
        "Press Units":           "presses",
        "Concentrators":         "concentrators",
        "Inter-arrival [min]":   "interarrival_min",
        "Press Batch Size":      "press_batch_size_apples",
        "Press Time [min]":      "press_time_min",
        "Conc. Factor":          "conc_factor",
        "Wash Rate [apples/min]":"wash_rate_apples_per_min",
    }

    col_p, col_s, col_e, col_st = st.columns(4)
    with col_p:
        param_label = st.selectbox("Parameter to Vary", list(param_options.keys()), key="t3_param")
    param_name = param_options[param_label]
    current_val = getattr(base_params, param_name)

    with col_s:
        start_val = st.number_input("Start Value", value=float(max(1, current_val * 0.5)), key="t3_start")
    with col_e:
        end_val   = st.number_input("End Value",   value=float(current_val * 2.0), key="t3_end")
    with col_st:
        step_val  = st.number_input("Step",        value=float(max(1, (current_val * 2 - current_val * 0.5) / 5)), key="t3_step")

    sweep_seed = st.number_input("Seed", value=42, step=1, key="t3_seed")

    if st.button("▶ Run Sweep", key="btn_sweep", type="primary"):
        is_int = isinstance(current_val, int)
        if is_int:
            values = list(range(int(start_val), int(end_val) + 1, max(1, int(step_val))))
        else:
            values = list(np.arange(start_val, end_val + step_val / 10, step_val))

        p = replace(base_params, random_seed=int(sweep_seed))
        with st.spinner(f"Running sweep over {param_label} ({len(values)} values)…"):
            sweep_df = run_parameter_sweep(p, param_name, values, seeds=[int(sweep_seed)])

        st.success(f"Sweep complete: {len(sweep_df)} runs.")
        st.plotly_chart(plot_sweep_results(sweep_df, param_name),
                        width='stretch')

        with st.expander("📋 Sweep Results Table"):
            st.dataframe(sweep_df, width='stretch')


# ============================================================
# TAB 4 – Combined Experiment (Sweep × Seeds)
# ============================================================
with tab4:
    st.header("Combined Experiment (Sweep × Seed Variation)")
    st.markdown("Vary a parameter while also running multiple seeds per value – reveals both trend and uncertainty.")

    param_options_c = {
        # "Washers":               "washers",
        "Wash & Shred Lines":    "wash_shred_lines",
        "Press Units":           "presses",
        "Concentrators":         "concentrators",
        "Inter-arrival [min]":   "interarrival_min",
        "Press Batch Size":      "press_batch_size_apples",
        "Conc. Factor":          "conc_factor",
    }

    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    with cc1:
        c_label  = st.selectbox("Parameter", list(param_options_c.keys()), key="t4_param")
    c_name = param_options_c[c_label]
    c_cur  = getattr(base_params, c_name)

    with cc2:
        c_start = st.number_input("Start", value=float(max(1, c_cur * 0.5)), key="t4_start")
    with cc3:
        c_end   = st.number_input("End",   value=float(c_cur * 2.0), key="t4_end")
    with cc4:
        c_step  = st.number_input("Step",  value=float(max(1, (c_cur * 2 - c_cur * 0.5) / 5)), key="t4_step")
    with cc5:
        c_seeds = st.slider("Seeds per value", 3, 20, 5, key="t4_nseeds")

    if st.button("▶ Run Combined Experiment", key="btn_comb", type="primary"):
        is_int_c = isinstance(c_cur, int)
        if is_int_c:
            c_values = list(range(int(c_start), int(c_end) + 1, max(1, int(c_step))))
        else:
            c_values = list(np.arange(c_start, c_end + c_step / 10, c_step))

        c_seeds_list = list(range(c_seeds))
        total_runs = len(c_values) * c_seeds

        with st.spinner(f"Running {total_runs} simulations ({len(c_values)} values × {c_seeds} seeds)…"):
            comb_df = run_parameter_sweep(base_params, c_name, c_values, seeds=c_seeds_list)

        st.success(f"Done: {len(comb_df)} runs.")
        st.plotly_chart(plot_sweep_results(comb_df, c_name),
                        width='stretch')

        with st.expander("📋 Full Results Table"):
            st.dataframe(comb_df, width='stretch')
