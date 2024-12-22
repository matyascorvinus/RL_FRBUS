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
            reward += real_gdp_comparison * 50  # Bounded real GDP contribution
        else:
            reward += real_gdp_comparison * 50  # Bounded real GDP contribution
    else:
        reward += real_gdp_comparison * 20  # Bounded real GDP contribution
    
    # Additional penalties for extreme outcomes
    if inflation_dev > 5.0 or unemployment_dev > 8.0:
        reward -= 10.0
    
    # Additional penalties for extreme outcomes
    if real_gdp_comparison < -2.0:
        reward -= 100.0
    
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
    
    normalized_inflation_dev = torch.clamp(inflation_dev / max_expected_deviation, -1, 1)
    normalized_unemployment_dev = torch.clamp(unemployment_dev / max_expected_deviation, -1, 1)
    normalized_gdp_dev = torch.clamp(gdp_growth_dev / max_expected_deviation, -1, 1)
    
    # 5. Smoothness incentives
    gdp_volatility = torch.std(solution.loc[:quarter, 'xgdp'].pct_change().dropna())
    inflation_volatility = torch.std(solution.loc[:quarter, 'pcpi'].pct_change().dropna())
    rate_volatility = torch.std(solution.loc[:quarter, 'rff'].diff().dropna())
    
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
    misery_index = unemployment + quarterly_inflation
    normalized_misery = torch.clamp(misery_index / 20.0, 0, 1)  # Normalize assuming max misery of 20
    
    debt_gdp = solution.loc[quarter, 'gfdbtn'] / solution.loc[quarter, 'xgdpn'] * 100
    normalized_debt = torch.clamp((debt_gdp - targets['gfdbtn']) / 100, 0, 1)
    
    # 9. Combine all components with balanced weights
    reward += -normalized_inflation_dev * 3.0          # Price stability
    reward += -normalized_unemployment_dev * 3.0       # Employment
    reward += normalized_gdp_dev * 3.0                # Growth
    reward += -normalized_volatility * 2.0            # Smoothness
    reward += forward_bonus                          # Forward-looking
    reward += growth_vs_history                      # Historical comparison
    reward += -normalized_misery * 2.0               # Economic health
    reward += -normalized_debt * 2.0                 # Fiscal sustainability
    
    # 10. Long-term incentives
    if int(end_quarter.split('q')[0]) >= 2030:
        reward += torch.clamp(torch.tensor(real_gdp_comparison), -5, 5) * 2.0
    
    # 11. Extreme outcome penalties
    if inflation_dev > 5.0 or unemployment_dev > 8.0:
        reward -= 5.0
    
    if real_gdp_comparison < -2.0:
        reward -= 5.0
    
    logger.info(f"Quarter {quarter} - GDP Growth: {gdp_growth_rl:.2f}, Inflation: {quarterly_inflation:.2f}, "
                f"Unemployment: {unemployment:.2f}, Reward: {reward:.2f}")
    
    return reward

def update_ppo(agent, experiences):
    """
    Update PPO agent with collected experiences.
    
    Args:
        agent (PPOAgent): The PPO agent to update
        experiences (list): List of dictionaries containing experience data
    """
    # Convert experiences to torch tensors, handling numpy arrays
    states = torch.stack([torch.from_numpy(e['state']).float() for e in experiences])
    
    # For actions, we need to handle the case where it's already a tensor
    actions_list = []
    log_probs_list = []
    rewards_list = []
    values_list = []
    dones_list = []
    
    for e in experiences:
        # Handle actions
        if isinstance(e['actions'], (list, tuple)):
            actions_list.append(torch.stack(e['actions']))
        else:
            actions_list.append(e['actions'])
            
        # Handle log_probs
        if isinstance(e['log_probs'], (list, tuple)):
            log_probs_list.append(torch.stack(e['log_probs']))
        else:
            log_probs_list.append(e['log_probs'])
            
        # Handle rewards
        if isinstance(e['reward'], (np.ndarray, float, int)):
            rewards_list.append(torch.tensor(float(e['reward'])))
        else:
            rewards_list.append(e['reward'])
            
        # Handle values
        if isinstance(e['value'], (np.ndarray, float, int)):
            values_list.append(torch.tensor(float(e['value'])))
        else:
            values_list.append(e['value'])
            
        # Handle dones
        dones_list.append(torch.tensor(float(e['done'])))
    
    actions = torch.stack(actions_list)
    log_probs = torch.stack(log_probs_list)
    rewards = torch.stack(rewards_list)
    values = torch.stack(values_list)
    dones = torch.stack(dones_list)
    
    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Calculate advantages using GAE
    advantages = []
    gae = torch.tensor(0.0)
    for i in reversed(range(len(rewards))):
        if dones[i]:
            gae = torch.tensor(0.0)
        next_value = values[i+1] if i < len(rewards)-1 else torch.tensor(0.0)
        delta = rewards[i] + agent.gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + agent.gamma * 0.95 * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.stack(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Create mini-batches for updating
    batch_size = min(64, len(rewards))  # Ensure batch size isn't larger than dataset
    n_samples = len(rewards)
    indices = torch.randperm(n_samples)
    
    # Multiple epochs of updating
    for _ in range(10):
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_log_probs = log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_rewards = rewards[batch_indices]
            batch_values = values[batch_indices]
            batch_dones = dones[batch_indices]

            # Update the policy and value function
            agent.update(
                batch_states,
                batch_actions,
                batch_log_probs,
                batch_rewards,
                batch_values,
                batch_dones
            )
    
    return agent
