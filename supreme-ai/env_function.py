import torch
import numpy as np
import logging

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
    
    annual_target = 2.0
    quarterly_target = ((1 + annual_target/100)**(1/4) - 1) * 100
    
    # Dynamic targets that adjust based on conditions
    targets = {
        'pcpi': quarterly_target,    # 2% inflation target
        'lur': 4.0,     # 4% unemployment target
        'hggdp': 4.0,   # 4% GDP growth target annualized
        'rff': 0.5,     # Limit on interest rate changes
        'gfdbtn': 60,   # Debt-to-GDP target
    }
    
    # 2. Calculate current economic indicators
    current_pcpi = solution.loc[quarter, 'pcpi']
    previous_pcpi = solution.shift(1).loc[quarter, 'pcpi']
    quarterly_inflation = ((current_pcpi - previous_pcpi) / previous_pcpi) * 100
    
    gdp_growth_rl = ((solution.loc[quarter, 'xgdp'] - solution.shift(1).loc[quarter, 'xgdp']) / 
                     solution.shift(1).loc[quarter, 'xgdp'] * 400.0)
    
    unemployment = solution.loc[quarter, 'lur']
    
    # # 3. Dynamic target adjustment
    # if gdp_growth_rl < 0:  # Recession conditions
    #     targets['lur'] = 5.0  # More flexible unemployment target
    #     targets['hggdp'] = 2.0  # Lower growth expectations
    
    inflation_trend = solution.loc[:quarter, 'pcpi'].pct_change().mean()
    if inflation_trend > 3.0:
        targets['pcpi'] *= 0.9  # Tighter policy when inflation trending up
    
    # 4. Calculate normalized deviations
    inflation_dev = abs(quarterly_inflation - targets['pcpi'])
    unemployment_dev = abs(unemployment - targets['lur'])
    gdp_growth_dev = gdp_growth_rl - targets['hggdp']
    
    # Fix 1: Convert numpy/float values to torch tensors
    normalized_inflation_dev = torch.clamp(
        torch.tensor(float(inflation_dev) / max_expected_deviation, dtype=torch.float32), 
        -1, 
        1
    )
    
    normalized_unemployment_dev = torch.clamp(
        torch.tensor(float(unemployment_dev) / max_expected_deviation, dtype=torch.float32), 
        -1, 
        1
    )
    
    normalized_gdp_dev = torch.clamp(
        torch.tensor(float(gdp_growth_dev) / max_expected_deviation, dtype=torch.float32), 
        -1, 
        1
    )
    # 5. Smoothness incentives    
    # Convert pandas Series to torch tensor before calculating std
    gdp_volatility = torch.std(
        torch.tensor(
            solution.loc[:quarter, 'xgdp'].pct_change().dropna().values,
            dtype=torch.float32
        )
    )
    
    inflation_volatility = torch.std(
        torch.tensor(
            solution.loc[:quarter, 'pcpi'].pct_change().dropna().values,
            dtype=torch.float32
        )
    )
    
    rate_volatility = torch.std(
        torch.tensor(
            solution.loc[:quarter, 'rff'].diff().dropna().values,
            dtype=torch.float32
        )
    )
    normalized_volatility = torch.clamp((gdp_volatility + inflation_volatility + rate_volatility) / 3, 0, 1)
      
    # 6. Relative performance metrics
    historical_avg_growth = solution.loc[:quarter, 'xgdp'].pct_change().mean()
    growth_vs_history = 1.0 if gdp_growth_rl > historical_avg_growth else -0.5
    
    # 7. Composite health indicators
    misery_index = torch.tensor(unemployment + quarterly_inflation)
    normalized_misery = torch.clamp(misery_index / 20.0, 0, 1)  # Normalize assuming max misery of 20
    
    debt_gdp = solution.loc[quarter, 'gfdbtn'] / solution.loc[quarter, 'xgdpn'] * 100
    normalized_debt = torch.clamp(torch.tensor((debt_gdp - targets['gfdbtn']) / 100), 0, 1)
    
    # 8. Combine all components with balanced weights
    # reward += -normalized_inflation_dev * 3.0          # Price stability
    # reward += -normalized_unemployment_dev * 3.0       # Employment
    # reward += normalized_gdp_dev * 10.0                # Growth
    # reward += -normalized_volatility * 4.0            # Smoothness
    # reward += growth_vs_history * 10.0                # Historical comparison - We want to make sure the AI prioritizes GDP performance
    # reward += -normalized_debt * 4.0                 # Fiscal sustainability 
    reward += normalized_gdp_dev
    return reward 