import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_reward(solution, solution_without_rl, quarter, end_quarter):
    """
    Calculate reward based on economic outcomes.
    
    Args:
        solution (DataFrame): Simulation results from FRB/US
        quarter (str): Current quarter (e.g., "2025q1")
    
    Returns:
        float: Calculated reward value
    """
    annual_target = 2.0  # 2% annual inflation target
    quarterly_target = ((1 + annual_target/100)**(1/4) - 1) * 100
    # Define target values
    targets = {
        'pcpi': quarterly_target,    # 2% inflation target
        'lur': 4.0,     # 4% unemployment target
        'hggdp': 4.0,    # 4% GDP growth target annualized
        'rff': 0.5,  # Limit on interest rate changes
        'gfdbtn': 80,  # Debt-to-GDP target
    }
    
    # Calculate components of the reward
    reward = torch.tensor(0.0)

    # 1. Price Stability (inflation targeting)
    # Calculate quarter-over-quarter percentage change in PCPI
    current_pcpi = solution.loc[quarter, 'pcpi']
    previous_pcpi = solution.shift(1).loc[quarter, 'pcpi']
    quarterly_inflation = ((current_pcpi - previous_pcpi) / previous_pcpi) * 100
    
    # Compare with target
    inflation_dev = abs(quarterly_inflation - targets['pcpi'])
    
    # 2. Employment (unemployment targeting)
    unemployment_dev = abs(solution.loc[quarter, 'lur'] - targets['lur'])
    reward -= unemployment_dev * 0.5
    
    # 3. Economic Growth
    gdp_growth_rl = ((solution.loc[quarter, 'xgdp'] - solution.shift(1).loc[quarter, 'xgdp']) / 
                     solution.shift(1).loc[quarter, 'xgdp'] * 400.0)  # *400 for annualized rate
    gdp_growth = gdp_growth_rl - targets['hggdp']
    reward += gdp_growth * 3  # Bounded GDP contribution
    
    # 4. Financial Stability
    rate_change = abs(solution.loc[quarter, 'rff'] - solution.shift(1).loc[quarter, 'rff'])
    reward -= rate_change * 1.0  # Penalty for large rate changes
    
    # 5. Fiscal Sustainability
    debt_gdp = solution.loc[quarter, 'gfdbtn'] / solution.loc[quarter, 'xgdpn'] * 100
    reward -= torch.clamp(torch.tensor((debt_gdp - targets['gfdbtn'])/100), min=0) * 0.5

    # 6. Real GDP Comparison
    real_gdp_comparison = (solution.loc[quarter, 'xgdp'] - solution_without_rl.loc[quarter, 'xgdp']) / solution_without_rl.loc[quarter, 'xgdp'] * 100
    
    logger.info(f"Real GDP Comparison: {real_gdp_comparison} for quarter {quarter}")
    # Push the AI to make better decisions in the long run
    if int(end_quarter.split('q')[0]) >= 2030:
        if real_gdp_comparison > 0:
            reward += real_gdp_comparison * 100  # Bounded real GDP contribution
    else:
        reward += real_gdp_comparison * 20  # Bounded real GDP contribution
    
    # Additional penalties for extreme outcomes
    if inflation_dev > 5.0 or unemployment_dev > 8.0:
        reward -= 10.0
    
    # Additional penalties for extreme outcomes
    if real_gdp_comparison < -1.0:
        reward -= 1000.0
    
    
    logger.info(f"Quarter {quarter} - GDP Growth: {gdp_growth_rl:.2f}, Inflation: {quarterly_inflation:.2f}, "
                f"Unemployment: {unemployment_dev:.2f}, Reward: {reward:.2f}, Real GDP Comparison: {real_gdp_comparison:.2f}")
    
    return reward
def calculate_reward_policy(solution, solution_without_rl, quarter, end_quarter):
    """
    Enhanced reward calculation based on economic outcomes with normalized scaling,
    smoothness incentives, and forward-looking components.
    
    Args:
        solution (DataFrame): Simulation results from FRB/US
        solution_without_rl (DataFrame): Baseline simulation without RL
        quarter (str): Current quarter (e.g., "2025q1")
        end_quarter (str): Final quarter of simulation
    
    Returns:
        float: Calculated reward value
    """
    # 1. Initialize reward and define normalized targets
    reward = torch.tensor(0.0)
    max_expected_deviation = 10.0  # Used for normalization
    
    annual_target = 2.0
    quarterly_target = ((1 + annual_target/100)**(1/4) - 1) * 100
    
    # Dynamic targets that adjust based on conditions
    targets = {
        'pcpi': quarterly_target,    # 2% inflation target
        'lur': 4.0,     # 4% unemployment target
        'hggdp': 4.0,   # 4% GDP growth target annualized
        'rff': 0.5,     # Limit on interest rate changes
        'gfdbtn': 80,   # Debt-to-GDP target
    }
    
    # 2. Calculate current economic indicators
    current_pcpi = solution.loc[quarter, 'pcpi']
    previous_pcpi = solution.shift(1).loc[quarter, 'pcpi']
    quarterly_inflation = ((current_pcpi - previous_pcpi) / previous_pcpi) * 100
    
    gdp_growth_rl = ((solution.loc[quarter, 'xgdp'] - solution.shift(1).loc[quarter, 'xgdp']) / 
                     solution.shift(1).loc[quarter, 'xgdp'] * 400.0)
    
    unemployment = solution.loc[quarter, 'lur']
    
    # 3. Dynamic target adjustment
    if gdp_growth_rl < 0:  # Recession conditions
        targets['lur'] = 5.0  # More flexible unemployment target
        targets['hggdp'] = 2.0  # Lower growth expectations
    
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
    
    # 6. Forward-looking components
    future_quarters = solution.loc[quarter:].head(4)
    if len(future_quarters) >= 4:
        trend_improvement = (
            future_quarters['xgdp'].pct_change().mean() > 0 and
            future_quarters['lur'].mean() < unemployment
        )
        forward_bonus = 2.0 if trend_improvement else -1.0
    else:
        forward_bonus = 0.0
    
    # 7. Relative performance metrics
    historical_avg_growth = solution.loc[:quarter, 'xgdp'].pct_change().mean()
    growth_vs_history = 1.0 if gdp_growth_rl > historical_avg_growth else -0.5
    
    real_gdp_comparison = (solution.loc[quarter, 'xgdp'] - solution_without_rl.loc[quarter, 'xgdp']) / \
                         solution_without_rl.loc[quarter, 'xgdp'] * 100
    
    # 8. Composite health indicators
    misery_index = torch.tensor(unemployment + quarterly_inflation)
    normalized_misery = torch.clamp(misery_index / 20.0, 0, 1)  # Normalize assuming max misery of 20
    
    debt_gdp = solution.loc[quarter, 'gfdbtn'] / solution.loc[quarter, 'xgdpn'] * 100
    normalized_debt = torch.clamp(torch.tensor((debt_gdp - targets['gfdbtn']) / 100), 0, 1)
    
    # 9. Combine all components with balanced weights
    reward += -normalized_inflation_dev * 3.0          # Price stability
    reward += -normalized_unemployment_dev * 3.0       # Employment
    reward += normalized_gdp_dev * 3.0                # Growth
    reward += -normalized_volatility * 2.0            # Smoothness
    reward += forward_bonus                          # Forward-looking
    reward += growth_vs_history * 5.0                      # Historical comparison
    reward += -normalized_misery * 2.0               # Economic health
    reward += -normalized_debt * 2.0                 # Fiscal sustainability
    
    # 10. Long-term incentives
    if int(end_quarter.split('q')[0]) >= 2030:
        reward += torch.clamp(torch.tensor(real_gdp_comparison), -5, 5) * 5.0
    
    # 11. Extreme outcome penalties
    if inflation_dev > 5.0 or unemployment_dev > 8.0:
        reward -= 5.0
    
    if real_gdp_comparison < -2.0:
        reward -= 5.0
    
    logger.info(f"Quarter {quarter} - GDP Growth: {gdp_growth_rl:.2f}, Inflation: {quarterly_inflation:.2f}, "
                f"Unemployment: {unemployment:.2f}, Reward: {reward:.2f}, Real GDP Comparison: {real_gdp_comparison:.2f}")
    
    return reward

def calculate_reward_policy_v1(solution, solution_without_rl, quarter, end_quarter):
    """
    Enhanced reward calculation based on economic outcomes with normalized scaling,
    smoothness incentives, and forward-looking components.
    
    Args:
        solution (DataFrame): Simulation results from FRB/US
        solution_without_rl (DataFrame): Baseline simulation without RL
        quarter (str): Current quarter (e.g., "2025q1")
        end_quarter (str): Final quarter of simulation
    
    Returns:
        float: Calculated reward value
    """
    # 1. Initialize reward and define normalized targets
    reward = torch.tensor(0.0)
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
    
    # 3. Dynamic target adjustment
    if gdp_growth_rl < 0:  # Recession conditions
        targets['lur'] = 5.0  # More flexible unemployment target
        targets['hggdp'] = 2.0  # Lower growth expectations
    
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
    
    real_gdp_comparison = (solution.loc[quarter, 'xgdp'] - solution_without_rl.loc[quarter, 'xgdp']) / \
                         solution_without_rl.loc[quarter, 'xgdp'] * 100
    
    # 7. Composite health indicators
    misery_index = torch.tensor(unemployment + quarterly_inflation)
    normalized_misery = torch.clamp(misery_index / 20.0, 0, 1)  # Normalize assuming max misery of 20
    
    debt_gdp = solution.loc[quarter, 'gfdbtn'] / solution.loc[quarter, 'xgdpn'] * 100
    normalized_debt = torch.clamp(torch.tensor((debt_gdp - targets['gfdbtn']) / 100), 0, 1)
    
    # 8. Combine all components with balanced weights
    reward += -normalized_inflation_dev * 3.0          # Price stability
    reward += -normalized_unemployment_dev * 3.0       # Employment
    reward += normalized_gdp_dev * 3.0                # Growth
    reward += -normalized_volatility * 4.0            # Smoothness
    reward += growth_vs_history * 5.0                # Historical comparison - We want to make sure the AI prioritizes GDP performance
    reward += -normalized_misery * 2.0               # Economic health
    reward += -normalized_debt * 4.0                 # Fiscal sustainability
    
    # 10. Long-term incentives
    if int(end_quarter.split('q')[0]) >= 2030:
        reward += torch.clamp(torch.tensor(real_gdp_comparison), -5, 5) * 5.0
    
    # 11. Extreme outcome penalties
    if inflation_dev > 5.0 or unemployment_dev > 8.0:
        reward -= 5.0
    
    if real_gdp_comparison < -2.0:
        reward -= 5.0
    
    logger.info(f"Quarter {quarter} - GDP Growth: {gdp_growth_rl:.2f}, Inflation: {quarterly_inflation:.2f}, "
                f"Unemployment: {unemployment:.2f}, Reward: {reward:.2f}, Real GDP Comparison: {real_gdp_comparison:.2f}")
    
    return reward


