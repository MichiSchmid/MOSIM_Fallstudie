# Notizen intern 

    * Auslastung (Utilization) ist statisch einfach berechenbar mittels ρ = λ/μ = (Ankunftsrate)/(Servicerate)  Prozess nur stabil, solange ρ<1
    * Für M/M/1-Queue gibt es noch genauere Formeln… 
    * Für weitere Analysen --> Simulation 


# Bank Simulation Project

This project implements a Discrete Event Simulation (DES) of a bank branch using **SimPy** and provides an interactive web interface using **Streamlit**.

## Simulation Process

The simulation models the flow of customers through a bank branch with two service options: a Service Counter and an ATM.

1.  **Source (Customer Arrival)**:
    *   Customers arrive according to a Poisson process (exponentially distributed inter-arrival times).
    *   The arrival rate ($\lambda$) is configurable (default: 0.6 entities/sec).

2.  **Splitting**:
    *   Upon arrival, customers are randomly assigned to either the **Counter** or the **ATM** with a 50/50 probability.

3.  **Queueing**:
    *   Customers enter a FIFO (First-In-First-Out) queue for their assigned resource.
    *   They wait until a server is available.

4.  **Service**:
    *   **Counter**: Has a capacity of 4 servers. Service time follows a Normal distribution ($\mu=13s, \sigma=5s$).
    *   **ATM**: Has a capacity of 1 server. Service time follows a Normal distribution ($\mu=2s, \sigma=1s$).
    *   Service times are clamped to be non-negative.

5.  **Sink**:
    *   After service is completed, customers leave the system.

## Data Structures

The simulation collects data in two primary structures, which are converted to Pandas DataFrames for analysis.

### 1. Entity Logs (`EntityLog`) 
Records the lifecycle of each customer.
*   `entity_id`: Unique identifier.
*   `entity_type`: 'counter' or 'atm'.
*   `arrival_time`: Simulation time when customer appeared.
*   `service_start_time`: Time when customer started being served.
*   `service_end_time`: Time when customer finished service.
*   `wait_time`: Duration spent in queue (`service_start_time` - `arrival_time`).
*   `sojourn_time`: Total time in system (`service_end_time` - `arrival_time`).

### 2. Queue Logs (`queue_logs`)
Snapshots of the system state taken every second.
*   `time`: Simulation time.
*   `counter_queue`: Number of people waiting for Counter.
*   `atm_queue`: Number of people waiting for ATM.
*   `counter_utilization`: Fraction of busy counter servers (0.0 to 1.0).
*   `atm_utilization`: Fraction of busy ATM servers (0.0 to 1.0).

## Metrics & Visualization
*   **Queue Length**: Moving average over 10 minutes (600s) to smooth out short-term fluctuations.
*   **Time Distributions**: Breakdown of Wait Time vs. Service Time vs. Total Sojourn Time.
*   **Utilization**: Average server occupancy.

## Technologies
*   **SimPy**: For discrete event simulation engine.
*   **Streamlit**: For the interactive web UI.
*   **Plotly**: For interactive and responsive charts.
*   **Pandas/NumPy**: For data manipulation and statistics.
