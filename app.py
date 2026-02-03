import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.model import SimulationConfig
from src.experiment import run_single_simulation, run_stochastic_variation, run_parameter_sweep
from src.visualization import (plot_queue_over_time, plot_queue_distribution, 
                              plot_server_utilization, plot_sojourn_times, plot_simulation_metrics)

st.set_page_config(page_title="Bank Simulation", layout="wide")

st.title("🏦 Bank Discrete Event Simulation")

# Sidebar - Global Configuration
st.sidebar.header("Simulation Parameters")

arrival_rate = st.sidebar.slider("Arrival Rate (entities/s)", 0.1, 2.0, 0.6, 0.1)
duration_hours = st.sidebar.number_input("Duration (hours)", 1, 48, 24)
duration_sec = duration_hours * 3600

st.sidebar.subheader("Counter Settings")
counter_cap = st.sidebar.number_input("Counter Capacity", 1, 10, 4)
counter_mean = st.sidebar.number_input("Counter Mean Service (s)", 5.0, 60.0, 13.0)
counter_std = st.sidebar.number_input("Counter Std Dev (s)", 0.0, 20.0, 5.0)

st.sidebar.subheader("ATM Settings")
atm_cap = st.sidebar.number_input("ATM Capacity", 1, 5, 1)
atm_mean = st.sidebar.number_input("ATM Mean Service (s)", 1.0, 20.0, 2.0)
atm_std = st.sidebar.number_input("ATM Std Dev (s)", 0.0, 5.0, 1.0)

# Base Config
base_config = SimulationConfig(
    arrival_rate=arrival_rate,
    simulation_duration=duration_sec,
    counter_capacity=counter_cap,
    counter_mean_service_time=counter_mean,
    counter_std_service_time=counter_std,
    atm_capacity=atm_cap,
    atm_mean_service_time=atm_mean,
    atm_std_service_time=atm_std
)

# Tabs for different modes
tab1, tab2, tab3, tab4 = st.tabs(["Single Run", "Stochastic Variation", "Parameter Sweep", "Combined Experiment"])

with tab1:
    st.header("Single Simulation Run")
    seed = st.number_input("Random Seed (Optional)", value=42, step=1, key="single_seed")
    
    if st.button("Run Simulation", key="btn_single"):
        config = base_config
        config.seed = seed
        
        with st.spinner("Running simulation..."):
            results = run_single_simulation(config)
            
        st.success("Simulation Complete!")
        
        # Metrics
        entity_logs = results['entity_logs']
        queue_logs = results['queue_logs']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Entities", len(entity_logs))
        c2.metric("Avg Wait (Counter)", f"{entity_logs[entity_logs['entity_type']=='counter']['wait_time'].mean():.2f} s")
        c3.metric("Avg Wait (ATM)", f"{entity_logs[entity_logs['entity_type']=='atm']['wait_time'].mean():.2f} s")
        c4.metric("Max Queue (Counter)", queue_logs['counter_queue'].max())
        
        # Plots
        st.subheader("Queue Analysis")
        st.plotly_chart(plot_queue_over_time(queue_logs), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_queue_distribution(queue_logs), use_container_width=True)
        with col2:
            st.plotly_chart(plot_server_utilization(queue_logs), use_container_width=True)
            
        st.subheader("Time Distribution")
        st.markdown("Distribution of Wait, Service, and Sojourn times.")
        st.plotly_chart(plot_sojourn_times(entity_logs), use_container_width=True)
        
        st.expander("Raw Data").dataframe(entity_logs)

with tab2:
    st.header("Stochastic Variation (Seed Sensitivity)")
    num_seeds = st.slider("Number of Random Seeds", 5, 50, 10)
    
    if st.button("Run Variation", key="btn_stoch"):
        seeds = list(range(num_seeds)) # Simple range of seeds
        
        with st.spinner(f"Running {num_seeds} simulations..."):
            df_results = run_stochastic_variation(base_config, seeds)
            
        st.subheader("Results Distribution")
        st.plotly_chart(plot_simulation_metrics(df_results), use_container_width=True)
        
        st.dataframe(df_results.describe())

with tab3:
    st.header("Parameter Sweep")
    param = st.selectbox("Parameter to Vary", [
        "arrival_rate", "counter_capacity", "counter_mean_service_time"
    ])
    
    start_val = st.number_input("Start Value", value=0.1)
    end_val = st.number_input("End Value", value=1.0)
    step_val = st.number_input("Step", value=0.1)
    
    if st.button("Run Sweep", key="btn_sweep"):
        # Generate values. Handle float vs int
        if "capacity" in param or "seed" in param:
             values = np.arange(int(start_val), int(end_val)+1, int(step_val))
        else:
             values = np.arange(start_val, end_val + step_val/10, step_val)
             
        with st.spinner(f"Running sweep over {param}..."):
            df_results = run_parameter_sweep(base_config, param, values) # seed=None by default
            
        st.subheader(f"Impact of {param}")
        st.plotly_chart(plot_simulation_metrics(df_results, x_col=param), use_container_width=True)
        st.dataframe(df_results)

with tab4:
    st.header("Combined Experiment (Sweep + Stochastic)")
    st.markdown("Varies a parameter while running multiple seeds per value to capture variance.")
    
    c_param = st.selectbox("Parameter to Vary", ["arrival_rate", "counter_capacity"], key="comb_param")
    
    c_start = st.number_input("Start", value=0.1, key="c_start")
    c_end = st.number_input("End", value=1.0, key="c_end")
    c_step = st.number_input("Step", value=0.1, key="c_step")
    c_seeds = st.slider("Seeds per value", 3, 20, 5, key="c_seeds")
    
    if st.button("Run Combined Experiment", key="btn_comb"):
        # Range generation
        if "capacity" in c_param:
             range_vals = np.arange(int(c_start), int(c_end)+1, int(c_step))
        else:
             range_vals = np.arange(c_start, c_end + c_step/10, c_step)
             
        seeds_list = list(range(c_seeds))
        
        with st.spinner("Running combined experiment..."):
            df_res = run_parameter_sweep(base_config, c_param, range_vals, seeds=seeds_list)
            
        st.subheader("Results with Confidence Intervals")
        st.plotly_chart(plot_simulation_metrics(df_res, x_col=c_param), use_container_width=True)
        
        st.dataframe(df_res)
