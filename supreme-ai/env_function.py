import torch
import numpy as np

def calculate_reward(solution, quarter):
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
        'hggdp': 1.0,    # 1% GDP growth target quarter over quarter
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
    
    # Additional penalties for extreme outcomes
    if inflation_dev > 5.0 or unemployment_dev > 8.0 or gdp_growth < -4.0:
        reward -= 10.0
    
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
