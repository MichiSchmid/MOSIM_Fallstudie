import simpy
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    arrival_rate: float = 0.6  # Entities per second
    counter_capacity: int = 4
    counter_mean_service_time: float = 13.0
    counter_std_service_time: float = 5.0
    atm_capacity: int = 1
    atm_mean_service_time: float = 2.0
    atm_std_service_time: float = 1.0
    simulation_duration: float = 24 * 60 * 60  # 24 hours in seconds
    seed: Optional[int] = None

@dataclass
class EntityLog:
    """DS for entity logging."""
    entity_id: int
    entity_type: str # 'counter' or 'atm'
    arrival_time: float
    service_start_time: float = -1
    service_end_time: float = -1
    wait_time: float = -1
    sojourn_time: float = -1

class BankSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        self.env = simpy.Environment()
        self.counter = simpy.Resource(self.env, capacity=config.counter_capacity)
        self.atm = simpy.Resource(self.env, capacity=config.atm_capacity)
        
        self.logs: List[EntityLog] = []
        self.queue_logs: List[Dict[str, Any]] = []
        self.utilization_logs: List[Dict[str, Any]] = []
        
        # Start processes
        self.env.process(self.source())
        self.env.process(self.monitor())

    def source(self):
        """Generates entities."""
        entity_id = 0
        while True:
            # Inter-arrival time is exponential (1/lambda)
            inter_arrival = random.expovariate(self.config.arrival_rate)
            yield self.env.timeout(inter_arrival)
            
            entity_id += 1
            # 50/50 split
            if random.random() < 0.5:
                self.env.process(self.customer_process(entity_id, 'counter'))
            else:
                self.env.process(self.customer_process(entity_id, 'atm'))

    def customer_process(self, entity_id: int, service_type: str):
        """Customer behavior."""
        arrival_time = self.env.now
        log_entry = EntityLog(entity_id=entity_id, entity_type=service_type, arrival_time=arrival_time)
        
        resource = self.counter if service_type == 'counter' else self.atm
        
        # Request resource
        with resource.request() as req:
            yield req
            
            # Service start
            service_start = self.env.now
            log_entry.service_start_time = service_start
            log_entry.wait_time = service_start - arrival_time
            
            # Service time
            if service_type == 'counter':
                duration = max(0, np.random.normal(self.config.counter_mean_service_time, self.config.counter_std_service_time))
            else:
                duration = max(0, np.random.normal(self.config.atm_mean_service_time, self.config.atm_std_service_time))
            
            yield self.env.timeout(duration)
            
            # Service end
            service_end = self.env.now
            log_entry.service_end_time = service_end
            log_entry.sojourn_time = service_end - arrival_time
            
            self.logs.append(log_entry)

    def monitor(self):
        """Monitors queue lengths and utilization periodically."""
        while True:
            self.queue_logs.append({
                'time': self.env.now,
                'counter_queue': len(self.counter.queue),
                'atm_queue': len(self.atm.queue),
                'counter_utilization': self.counter.count / self.counter.capacity,
                'atm_utilization': self.atm.count / self.atm.capacity
            })
            yield self.env.timeout(1.0) # Monitor every second

    def run(self):
        self.env.run(until=self.config.simulation_duration)
        return pd.DataFrame([vars(l) for l in self.logs]), pd.DataFrame(self.queue_logs)
