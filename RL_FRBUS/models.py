
from pydantic import BaseModel
from typing import Dict, Optional, List

class EconomicMetrics(BaseModel):
    quarter: str
    metrics: Dict[str, float] = {
        'hggdp': 0.0,  # GDP growth
        'xgdpn': 0.0,  # Nominal GDP
        'xgdp': 0.0,   # Real GDP
        'tpn': 0.0,    # Personal tax revenues
        'tcin': 0.0,   # Corporate tax revenues
        'trptx': 0.0,    # Personal tax rates
        'trcit': 0.0,   # Corporate tax rates
        'gtrt': 0.0,   # Transfer payments ratio
        'egfe': 0.0,  # Federal expenditures
        'rff': 0.0,  # Interest rate
        'pcpi': 0.0,   # PCI value
        'lur': 0.0,    # Unemployment rate
        'gfdbtn': 0.0, # Debt-to-GDP ratio
        'emn': 0.0,    # Imports
        'exn': 0.0     # Exports
    }
    previous_metrics: Optional[Dict[str, float]]
    targets: Optional[Dict[str, float]]

class SimulationComparison(BaseModel):
    rl_decision: EconomicMetrics
    without_tariff: EconomicMetrics
    base_simulation: EconomicMetrics

# Pydantic model for request body
class SimulationRequest(BaseModel):
    checkpoint_path: str
    simulation_start: str
    simulation_end: str 
    simulation_replications: int 
    key_checkpoint_path: str

class SimulationResponse(BaseModel):
    simulation_results: str
