"""
model.py – Juice Plant Discrete Event Simulation (SimPy)
=========================================================
Process steps:
  1) Wash & Shred  – combined batch step: wash_and_shred() pulls a fixed chunk from
                     apple_delivery, washes then shreds at wash_shred_rate_apples_per_min
                     using wash_shred_lines parallel lines (single shared resource).
  3) Pressing      – batch process collecting apples into press_batch_size_apples,
                     then runs for press_time_min and converts apples → raw liters
  4) Concentration – continuous, rate-based [liters/min], converts raw juice to
                     concentrated juice via conc_factor, parallel concentrators
  5) Storage       – simpy.Container accumulating concentrated liters

Logging (replaces the old Metrics class):
  process_logs  – one row per process-step START and FINISH event
  container_logs – periodic snapshots of all container / buffer levels
"""

import random
import simpy
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Params:
    # Arrival process (batches of apples)
    interarrival_min: float = 300          # mean inter-arrival time [min]
    deterministic_arrival: bool = True     # True → fixed ia = interarrival_min, False → exponential
    batch_size_min: int = 25000            # min apples per arriving batch
    batch_size_max: int = 25000            # max apples per arriving batch

    # Wash & Shred step (wash_and_shred) – single combined resource
    wash_shred_rate_apples_per_min: float = 120   # throughput per line [apples/min] (wash and shred share this rate)
    wash_shred_lines: int = 1                     # parallel wash+shred lines (shared capacity)
    entity_batch_size_apples: int = 10           # apples pulled from apple_delivery per wash/shred cycle

    # Batch pressing step
    press_batch_size_apples: int = 7000     # apples collected before one press run
    press_time_min: float = 90              # time one press batch takes [min]
    presses: int = 1                        # parallel press units

    # Apple-to-juice transformation (stochastic yield)
    liters_per_apple_mean: float = 0.12     # mean yield [L/apple]
    liters_per_apple_sd: float = 0.01       # std dev of yield

    # Continuous concentration step
    conc_rate_liters_per_min: float = 8     # throughput per concentrator [L/min]
    concentrators: int = 1                  # parallel concentrator units
    conc_factor: float = 0.30               # output = input * conc_factor (water removed)
    conc_chunk_liters: float = 10.0         # chunk size pulled from raw_juice_buffer per cycle

    # Simulation control
    sim_time_min: float = 72 * 60            # total simulated time [min]
    monitor_interval_min: float = 1.0       # container snapshot interval [min]
    random_seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Juice Plant Model
# ---------------------------------------------------------------------------

class JuicePlant:
    """
    Self-contained SimPy model of a juice production plant.

    Arrival process is integrated as a method and started in __init__.
    Storage is a simpy.Container; no separate StorageTank class.
    All monitoring is done via event logs (process_logs, container_logs).
    """

    def __init__(self, env: simpy.Environment, p: Params):
        self.env = env
        self.p = p

        # ---- Resources (parallel machine lines) ----------------------------
        self.wash_shred_lines = simpy.Resource(env, capacity=p.wash_shred_lines)  # shared by wash + shred
        self.presses          = simpy.Resource(env, capacity=p.presses)
        self.concentrators = simpy.Resource(env, capacity=p.concentrators)

        # ---- Containers (intermediate buffers + final storage) --------------
        self.apple_delivery   = simpy.Container(env, init=0,   capacity=10**8)  # arrived, awaiting wash_and_shred
        self.apple_buffer     = simpy.Container(env, init=0,   capacity=20000)  # 20000
        self.raw_juice_buffer = simpy.Container(env, init=0.0, capacity=10**8)   # 2000
        self.storage_tank     = simpy.Container(env, init=0.0, capacity=10**8)   # 10**6

        # ---- Logs ----------------------------------------------------------
        # process_logs: one entry per START and FINISH event of each process step
        self.process_logs: List[Dict[str, Any]] = []
        # container_logs: periodic snapshots of buffer / storage levels
        self.container_logs: List[Dict[str, Any]] = []

        # ---- Internal counters ---------------------------------------------
        self._batch_counter = 0

        # ---- Start background processes ------------------------------------
        env.process(self.arrival())
        # Start one wash_and_shred loop per wash/shred line
        for _ in range(p.wash_shred_lines):
            env.process(self.wash_and_shred())
        # Start as many pressing processes as there are presses
        for _ in range(p.presses):
            env.process(self.pressing_process())
        # Start as many concentration processes as there are concentrators
        for _ in range(p.concentrators):
            env.process(self.concentration_process())
        env.process(self.monitor())

    # -----------------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------------

    def _log_process(self, event: str, step: str, batch_id: int,
                     quantity: float, unit: str):
        """Append one process-step log entry."""
        self.process_logs.append({
            "time":     self.env.now,
            "event":    event,       # "started" or "finished"
            "step":     step,        # "wash", "shred", "press", "concentrate"
            "batch_id": batch_id,
            "quantity": quantity,
            "unit":     unit,        # "apples" or "liters"
        })

    # -----------------------------------------------------------------------
    # Arrival process (integrated into the plant)
    # -----------------------------------------------------------------------

    def arrival(self):
        """
        Generates arriving apple batches (exponential or deterministic inter-arrival).
        Puts apples into apple_delivery; wash_and_shred() consumes them.
        """
        # initial delay of 1 time step before first arrival (optional)
        yield self.env.timeout(1)

        while True:

            apples = random.randint(self.p.batch_size_min, self.p.batch_size_max)
            self._batch_counter += 1
            batch_id = self._batch_counter

            self._log_process("arrived", "arrival", batch_id, apples, "apples")
            # Place apples in the delivery buffer; batch_entity() will consume them
            yield self.apple_delivery.put(apples)


            if self.p.deterministic_arrival:
                ia = self.p.interarrival_min
            else:
                ia = random.expovariate(1.0 / self.p.interarrival_min)
            yield self.env.timeout(ia)

    # -----------------------------------------------------------------------
    # Wash & Shred combined process: apple_delivery → wash → shred → apple_buffer
    # -----------------------------------------------------------------------

    def wash_and_shred(self):
        """
        Persistent loop: pulls entity_batch_size_apples from apple_delivery,
        washes then shreds (both at wash_shred_rate_apples_per_min, sharing
        the same wash_shred_lines resource), then feeds apple_buffer.
        One loop is started per wash_shred_lines.
        """
        p = self.p
        chunk = p.entity_batch_size_apples
        run = 0
        while True:
            # Wait until enough apples are available in the delivery buffer
            yield self.apple_delivery.get(chunk)
            run += 1

            # --- Step 1: Washing and Shredding  ----------------------------
            self._log_process("started", "wash and shred", run, chunk, "apples")
            with self.wash_shred_lines.request() as req:
                yield req
                t = chunk / p.wash_shred_rate_apples_per_min
                yield self.env.timeout(t)
            self._log_process("finished", "wash and shred", run, chunk, "apples")

            # Feed shredded apples into the press buffer
            yield self.apple_buffer.put(chunk)

    # -----------------------------------------------------------------------
    # Pressing process (batch collector)
    # -----------------------------------------------------------------------

    def pressing_process(self):
        """
        Continuously pulls press_batch_size_apples from apple_buffer,
        runs one press cycle, converts to raw juice liters, feeds raw_juice_buffer.
        """
        p = self.p
        press_run = 0
        while True:
            # Wait until enough apples for one press batch
            yield self.apple_buffer.get(p.press_batch_size_apples)
            press_run += 1

            self._log_process("started", "press", press_run, p.press_batch_size_apples, "apples")

            with self.presses.request() as req:
                yield req
                yield self.env.timeout(p.press_time_min)

            # Stochastic yield (apples → liters)
            y = random.gauss(p.liters_per_apple_mean, p.liters_per_apple_sd)
            y = max(0.01, min(5.0, y))  # clamp yield to a reasonable range 0.01-5
            raw_liters = p.press_batch_size_apples * y

            self._log_process("finished", "press", press_run, raw_liters, "liters")

            yield self.raw_juice_buffer.put(raw_liters)

    # -----------------------------------------------------------------------
    # Concentration process
    # -----------------------------------------------------------------------

    def concentration_process(self):
        """
        Continuously pulls chunks of raw juice from raw_juice_buffer,
        concentrates them (applies conc_factor), deposits to storage_tank.
        """
        p = self.p
        chunk = p.conc_chunk_liters
        conc_run = 0
        while True:
            yield self.raw_juice_buffer.get(chunk)
            conc_run += 1

            self._log_process("started", "concentrate", conc_run, chunk, "liters")

            with self.concentrators.request() as req:
                yield req
                t = chunk / p.conc_rate_liters_per_min
                yield self.env.timeout(t)

            out_liters = chunk * p.conc_factor
            self._log_process("finished", "concentrate", conc_run, out_liters, "liters")

            yield self.storage_tank.put(out_liters)

    # -----------------------------------------------------------------------
    # Container level monitor
    # -----------------------------------------------------------------------

    def monitor(self):
        """Periodic snapshot of all container / buffer levels."""
        while True:
            self.container_logs.append({
                "time":               self.env.now,
                "apple_delivery":     self.apple_delivery.level,
                "apple_buffer":       self.apple_buffer.level,
                "raw_juice_buffer":   self.raw_juice_buffer.level,
                "storage_tank":       self.storage_tank.level,
            })
            yield self.env.timeout(self.p.monitor_interval_min)

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the simulation until sim_time_min.
        Returns (process_logs_df, container_logs_df).
        """
        self.env.run(until=self.p.sim_time_min)
        return (
            pd.DataFrame(self.process_logs),
            pd.DataFrame(self.container_logs),
        )


# ---------------------------------------------------------------------------
# Factory convenience
# ---------------------------------------------------------------------------

def create_and_run(p: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set seed, build plant, run simulation, return DataFrames."""
    if p.random_seed is not None:
        random.seed(p.random_seed)

    env = simpy.Environment()
    plant = JuicePlant(env, p)
    return plant.run()
