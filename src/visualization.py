import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional

def plot_queue_over_time(queue_logs: pd.DataFrame):
    """Plots queue length over time for both resources using Plotly."""
    if queue_logs.empty:
        return go.Figure()

    fig = go.Figure()

    # Raw Data - optionally hide or make faint? keeping it simpler for now
    fig.add_trace(go.Scatter(x=queue_logs['time'], y=queue_logs['counter_queue'], 
                             mode='lines', name='Counter Queue', line=dict(width=1, color='blue'), opacity=0.3))
    fig.add_trace(go.Scatter(x=queue_logs['time'], y=queue_logs['atm_queue'], 
                             mode='lines', name='ATM Queue', line=dict(width=1, color='orange'), opacity=0.3))

    # Moving Averages (10 min = 600s). Assuming data is 1s interval.
    # If interval is not 1s, window should be calculated based on time.
    # Since monitor is 1s, window=600 is correct.
    ma_window = 600
    
    fig.add_trace(go.Scatter(x=queue_logs['time'], y=queue_logs['counter_queue'].rolling(window=ma_window, min_periods=1).mean(), 
                             mode='lines', name='Counter Avg (10min)', line=dict(width=3, color='blue')))
    fig.add_trace(go.Scatter(x=queue_logs['time'], y=queue_logs['atm_queue'].rolling(window=ma_window, min_periods=1).mean(), 
                             mode='lines', name='ATM Avg (10min)', line=dict(width=3, color='orange')))

    fig.update_layout(title='Queue Length over Time',
                      xaxis_title='Time (s)',
                      yaxis_title='Queue Length',
                      template='plotly_white')
    return fig

def plot_queue_distribution(queue_logs: pd.DataFrame):
    """Plots distribution of queue lengths using Plotly."""
    if queue_logs.empty:
        return go.Figure()
        
    # Melt for easier plotting with express
    df_melt = queue_logs.melt(value_vars=['counter_queue', 'atm_queue'], var_name='Queue Type', value_name='Length')
    df_melt['Queue Type'] = df_melt['Queue Type'].map({'counter_queue': 'Counter', 'atm_queue': 'ATM'})
    
    fig = px.histogram(df_melt, x='Length', color='Queue Type', barmode='group',
                       title='Queue Length Distribution', nbins=20)
    fig.update_layout(template='plotly_white')
    return fig

def plot_server_utilization(queue_logs: pd.DataFrame):
    """Plots server utilization using Plotly."""
    if queue_logs.empty:
        return go.Figure()

    avg_counter = queue_logs['counter_utilization'].mean() * 100
    avg_atm = queue_logs['atm_utilization'].mean() * 100
    
    fig = go.Figure(data=[
        go.Bar(name='Utilization', x=['Counter', 'ATM'], y=[avg_counter, avg_atm],
               text=[f"{avg_counter:.1f}%", f"{avg_atm:.1f}%"], textposition='auto')
    ])
    
    fig.update_layout(title='Average Server Utilization (%)', 
                      yaxis_title='Utilization (%)', 
                      yaxis_range=[0, 100],
                      template='plotly_white')
    return fig

def plot_sojourn_times(entity_logs: pd.DataFrame):
    """Plots distribution of Wait, Service, and Sojourn times."""
    if entity_logs.empty:
        return go.Figure()

    # We want to show 3 metrics for potentially both entity types.
    # Let's create a melted dataframe for all metrics
    
    # Filter valid logs (completed services)
    df = entity_logs[entity_logs['sojourn_time'] >= 0].copy()
    
    # We want to compare Sojourn, Wait, Service
    # Option 1: Box plots grouped by Entity Type and Time Metric
    # Create a long format: Entity Type | Metric | Time
    
    df_melted = df.melt(id_vars=['entity_type'], 
                        value_vars=['wait_time', 'sojourn_time', 'service_time' if 'service_time' in df else 'wait_time'], # Service time impl?
                        var_name='Metric', value_name='Time')
    
    # If service_time wasn't explicitly logged as a column, derive it or ensure model logs it. 
    # Checking model: model logs 'wait_time' and 'sojourn_time'. Service = Sojourn - Wait.
    if 'service_time' not in df.columns:
        df['service_time'] = df['sojourn_time'] - df['wait_time']
        
    df_melted = df.melt(id_vars=['entity_type'], 
                        value_vars=['wait_time', 'service_time', 'sojourn_time'],
                        var_name='Metric', value_name='Time')
    
    df_melted['Metric'] = df_melted['Metric'].str.replace('_', ' ').str.title()
    df_melted['Entity Type'] = df_melted['entity_type'].str.title()
    
    fig = px.box(df_melted, x='Entity Type', y='Time', color='Metric',
                 title='Time Distribution: Wait vs Service vs Sojourn',
                 points="outliers") # Show outliers, clean look
    fig.update_layout(template='plotly_white', yaxis_title='Time (s)')
    return fig

def plot_simulation_metrics(results_df: pd.DataFrame, x_col: Optional[str] = None):
    """Plots metrics for comparison (Sweep or Stochastic) using Plotly."""
    metrics = ['avg_wait_counter', 'avg_wait_atm', 'avg_queue_counter', 'utilization_counter']
    
    # Use subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[m.replace('_', ' ').title() for m in metrics])
    
    row_col_map = [(1,1), (1,2), (2,1), (2,2)]
    
    if x_col:
        # Parameter Sweep -> Line plots with error bars if available
        # Check if we have std dev cols (from combined exp)
        # If raw data has multiple points per x, we can aggregate
        
        # If it's a raw sweep output with seeds, calculate mean/std
        if 'seed' in results_df.columns and results_df['seed'].notnull().any():
            stats = results_df.groupby(x_col).agg(['mean', 'std']).reset_index()
            # Flatten columns
            stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in stats.columns.values]
            x_data = stats[x_col]
        else:
             # simple sweep
             stats = results_df
             x_data = stats[x_col]

        for idx, metric in enumerate(metrics):
            r, c = row_col_map[idx]
            if metric in results_df.columns:
                # If we have aggregated stats
                y_mean = stats[metric + '_mean'] if f'{metric}_mean' in stats.columns else stats[metric]
                y_std = stats[metric + '_std'] if f'{metric}_std' in stats.columns else None
                
                trace = go.Scatter(x=x_data, y=y_mean, mode='lines+markers', name=metric)
                if y_std is not None:
                     # Add error bars
                     trace.error_y = dict(type='data', array=y_std, visible=True)
                
                fig.add_trace(trace, row=r, col=c)
                fig.update_xaxes(title_text=x_col, row=r, col=c)
    else:
        # Stochastic Var -> Box Plots (if multiple runs)
        for idx, metric in enumerate(metrics):
            r, c = row_col_map[idx]
            if metric in results_df.columns:
                fig.add_trace(go.Box(y=results_df[metric], name='Distribution'), row=r, col=c)

    fig.update_layout(height=800, title_text="Simulation Metrics", showlegend=False, template='plotly_white')
    return fig
