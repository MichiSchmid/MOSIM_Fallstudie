"""
experiment.py – Experiment runners for the Juice Plant SimPy Simulation
=======================================================================
Provides:
  run_single()           – one simulation run
  run_seed_variation()   – multiple seeds for stochastic sensitivity
  run_parameter_sweep()  – vary one parameter × seeds
"""

import random
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import List, Optional

import pandas as pd

from model import Params, create_and_run


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(params: Params) -> dict:
    """
    Run a single simulation.

    Returns
    -------
    dict with keys:
      "process_logs"   : pd.DataFrame – one row per step event
      "container_logs" : pd.DataFrame – periodic container level snapshots
      "params"         : Params used for this run
    """
    proc_df, cont_df = create_and_run(params)
    return {
        "process_logs":   proc_df,
        "container_logs": cont_df,
        "params":         params,
    }


# ---------------------------------------------------------------------------
# Stochastic variation (seed sweep)
# ---------------------------------------------------------------------------

def run_seed_variation(base_params: Params, seeds: List[int]) -> dict:
    """
    Run the simulation once per seed.

    Returns
    -------
    dict with keys:
      "summary"          : pd.DataFrame – one row per seed with KPIs
      "all_process_logs" : pd.DataFrame – all runs concatenated, with 'seed' column
      "all_container_logs": pd.DataFrame – all runs concatenated, with 'seed' column
    """
    all_proc = []
    all_cont = []
    summaries = []

    for seed in seeds:
        p = replace(base_params, random_seed=seed)
        proc_df, cont_df = create_and_run(p)

        proc_df["seed"] = seed
        cont_df["seed"] = seed

        all_proc.append(proc_df)
        all_cont.append(cont_df)
        summaries.append(_summarise(proc_df, cont_df, seed=seed))

    return {
        "summary":           pd.DataFrame(summaries),
        "all_process_logs":  pd.concat(all_proc,  ignore_index=True),
        "all_container_logs": pd.concat(all_cont, ignore_index=True),
    }


# ---------------------------------------------------------------------------
# Parameter sweep (optionally combined with seed variation)
# ---------------------------------------------------------------------------

def run_parameter_sweep(
    base_params: Params,
    param_name: str,
    param_values: list,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Vary one parameter across param_values; optionally run multiple seeds per value.

    Parameters
    ----------
    base_params  : baseline Params
    param_name   : attribute name of Params to vary (e.g. "washers")
    param_values : list of values to try
    seeds        : list of seeds per value; None → use base_params.random_seed only

    Returns
    -------
    pd.DataFrame with one row per (param_value × seed) combination.
    """
    if seeds is None:
        seeds = [base_params.random_seed]

    results = []
    for val in param_values:
        for seed in seeds:
            p = replace(base_params, random_seed=seed)
            setattr(p, param_name, val)

            proc_df, cont_df = create_and_run(p)
            row = _summarise(proc_df, cont_df, seed=seed)
            row[param_name] = val
            results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Internal KPI summariser
# ---------------------------------------------------------------------------

def _summarise(proc_df: pd.DataFrame, cont_df: pd.DataFrame,
               seed=None) -> dict:
    """Extract scalar KPIs from one simulation run."""
    row: dict = {"seed": seed}

    if proc_df.empty:
        return row

    # -- Throughput per step -------------------------------------------------
    for step in ["wash", "shred", "press", "concentrate"]:
        started  = proc_df[(proc_df["step"] == step) & (proc_df["event"] == "started")]
        finished = proc_df[(proc_df["step"] == step) & (proc_df["event"] == "finished")]
        row[f"{step}_batches_started"]  = len(started)
        row[f"{step}_batches_finished"] = len(finished)
        row[f"{step}_quantity_in"]  = started["quantity"].sum()
        row[f"{step}_quantity_out"] = finished["quantity"].sum()

    # -- Batches arrived -----------------------------------------------------
    arrivals = proc_df[(proc_df["step"] == "arrival") & (proc_df["event"] == "arrived")]
    row["batches_arrived"]     = len(arrivals)
    row["total_apples_arrived"] = arrivals["quantity"].sum()

    # -- Final storage level -------------------------------------------------
    if not cont_df.empty:
        row["final_storage_liters"]    = cont_df["storage_tank"].iloc[-1]
        row["max_apple_buffer"]        = cont_df["apple_buffer"].max()
        row["max_raw_juice_buffer"]    = cont_df["raw_juice_buffer"].max()
        row["avg_apple_buffer"]        = cont_df["apple_buffer"].mean()
        row["avg_raw_juice_buffer"]    = cont_df["raw_juice_buffer"].mean()

    return row
