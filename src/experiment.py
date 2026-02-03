from .model import BankSimulation, SimulationConfig
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union

def run_single_simulation(config: SimulationConfig) -> Dict[str, pd.DataFrame]:
    """Runs a single simulation and returns logs."""
    sim = BankSimulation(config)
    entity_logs, queue_logs = sim.run()
    return {
        "entity_logs": entity_logs,
        "queue_logs": queue_logs,
        "config": config
    }

def run_stochastic_variation(base_config: SimulationConfig, seeds: List[int]) -> pd.DataFrame:
    """Runs simulation for multiple seeds and aggregates results."""
    results = []
    
    for seed in seeds:
        config = SimulationConfig(**vars(base_config)) # Copy config
        config.seed = seed
        sim = BankSimulation(config)
        entity_logs, queue_logs = sim.run()
        
        # Calculate summary metrics for this run
        run_summary = {
            "seed": seed,
            "avg_wait_counter": entity_logs[entity_logs['entity_type'] == 'counter']['wait_time'].mean(),
            "avg_wait_atm": entity_logs[entity_logs['entity_type'] == 'atm']['wait_time'].mean(),
            "avg_sojourn_counter": entity_logs[entity_logs['entity_type'] == 'counter']['sojourn_time'].mean(),
            "avg_sojourn_atm": entity_logs[entity_logs['entity_type'] == 'atm']['sojourn_time'].mean(),
            "avg_queue_counter": queue_logs['counter_queue'].mean(),
            "avg_queue_atm": queue_logs['atm_queue'].mean(),
            "utilization_counter": queue_logs['counter_utilization'].mean(),
            "utilization_atm": queue_logs['atm_utilization'].mean()
        }
        results.append(run_summary)
        
    return pd.DataFrame(results)

def run_parameter_sweep(base_config: SimulationConfig, param_name: str, param_values: List[float], seeds: List[int] = [None]) -> pd.DataFrame:
    """Runs parameter sweep, optionally with multiple seeds per value."""
    results = []
    
    for val in param_values:
        for seed in seeds:
            config = SimulationConfig(**vars(base_config))
            setattr(config, param_name, val)
            if seed is not None:
                config.seed = seed
                
            sim = BankSimulation(config)
            entity_logs, queue_logs = sim.run()
            
            run_summary = {
                param_name: val,
                "seed": seed,
                "avg_wait_counter": entity_logs[entity_logs['entity_type'] == 'counter']['wait_time'].mean(),
                "avg_wait_atm": entity_logs[entity_logs['entity_type'] == 'atm']['wait_time'].mean(),
                "avg_sojourn_counter": entity_logs[entity_logs['entity_type'] == 'counter']['sojourn_time'].mean(),
                "avg_sojourn_atm": entity_logs[entity_logs['entity_type'] == 'atm']['sojourn_time'].mean(),
                "avg_queue_counter": queue_logs['counter_queue'].mean(),
                "avg_queue_atm": queue_logs['atm_queue'].mean(),
                "utilization_counter": queue_logs['counter_utilization'].mean(),
                "utilization_atm": queue_logs['atm_utilization'].mean()
            }
            results.append(run_summary)
            
    return pd.DataFrame(results)
