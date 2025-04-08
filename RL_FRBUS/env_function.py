import torch
import numpy as np
import logging
from ppo_agent import ACTION_BOUNDS

logger = logging.getLogger(__name__)

def calculate_reward_policy_v1(solution, quarter, end_quarter):
    """
    Enhanced reward calculation based on economic outcomes with normalized scaling,
    smoothness incentives, and forward-looking components.
    
    Args:
        solution (DataFrame): Simulation results from FRB/US
        quarter (str): Current quarter (e.g., "2025q1")
        end_quarter (str): Final quarter of simulation
    
    Returns:
        float: Calculated reward value
    """
    # 1. Initialize reward and define normalized targets
    reward = torch.tensor(0.0).clone().detach()
    max_expected_deviation = 10.0  # Used for normalization
    
    # Dynamic targets that adjust based on conditions
    targets = {
        'hggdp': 4.0,   # 4% GDP growth target annualized
        # 'hggdp': 3.5,   # 3.5% GDP growth target annualized
        # 'hggdp': 3.0,   # 3.0% GDP growth target annualized
    }
    
    # 2. Calculate current economic indicators
    gdp_growth_rl = ((solution.loc[quarter, 'xgdp'] - solution.shift(1).loc[quarter, 'xgdp']) / 
                     solution.shift(1).loc[quarter, 'xgdp'])
    gdp_growth_rl_annualized = ((1 + gdp_growth_rl)**4 - 1) * 100
    logger.info(f"GDP growth rate {quarter}: {gdp_growth_rl_annualized}")
    gdp_growth_dev = gdp_growth_rl_annualized - targets['hggdp']
    normalized_gdp_dev = torch.clamp(
        torch.tensor(float(gdp_growth_dev) / max_expected_deviation, dtype=torch.float32), 
        -1, 
        1
    )

    reward += normalized_gdp_dev
    return reward 