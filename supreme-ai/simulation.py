import os
from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import stochsim_plot
from pyfrbus.load_data import load_data
from ppo_agent import PPOAgent
import numpy as np
import torch
from env_function import calculate_reward, update_ppo
import pandas as pd
import time


policy_vars = ['frs10', 'gtrt', 'egfet', 'trp', 'trci']
# Load data and model
data = load_data("../data/LONGBASE.TXT")
frbus = Frbus("../models/model.xml")

# Initialize PPO agent
# state_dim = number of economic indicators you're monitoring
# action_dim = number of policy tools 
ppo_agent = PPOAgent(state_dim=8, action_dim=len(policy_vars))

# Simulation parameters
residstart = "1975q1"
residend = "2018q4"
simstart = "2023q1"  # Starting from 2023
simend = "2045q4" 

# Number of replications
nrepl = 1000
# Run up to 5 extra replications, in case of failures
nextra = 1
experiences = []
# Policy settings
data.loc[simstart:simend, "dfpdbt"] = 0
data.loc[simstart:simend, "dfpsrp"] = 1
data.loc[simstart:simend, "eps_s"] = 0
data.loc[simstart:simend, "eps_i"] = 0
def get_state(data, current_quarter):
    """Extract state variables from data"""  
    return np.array([
        data.loc[current_quarter, 'hggdp'],  # Growth rate of GDP, cw 2012$ (annual rate)
        data.loc[current_quarter, 'pcpi'],   # Inflation
        data.loc[current_quarter, 'lur'],    # Unemployment
        data.loc[current_quarter, 'frs10'],  # Interest rate
        data.loc[current_quarter, 'gtrt'],  # Trend ratio of transfer payments to GDP.
        data.loc[current_quarter, 'egfet'],  # Trend level of federal government expenditures.
        data.loc[current_quarter, 'trp'],  # Personal tax revenues rates
        data.loc[current_quarter, 'trci'],  # Corporate tax revenues rates
        # Add other relevant state variables
    ])

def apply_actions(data, current_quarter, actions):
    """Apply PPO agent's actions to the data"""
    # Map actions to specific policy variables
    for var, action in zip(policy_vars, actions): 
        print(f"Quarter: {current_quarter} {var} with action {action}")
        if var == 'egfet':
            data.loc[current_quarter, var] += action * 1000  # Convert to the magnitude of 1 trillion dollars
        if var == 'frs10':
            if data.loc[current_quarter, var] + action > 0.025:
                data.loc[current_quarter, var] += action
        else:
            data.loc[current_quarter, var] += action 
    return data

def apply_tariff(data, current_quarter, tariff_rate):
    """
    Apply tariff effects to the economy
    
    Args:
        data: DataFrame containing economic variables
        current_quarter: Current simulation quarter
        tariff_rate: Rate of tariff (e.g., 0.05 for 5% tariff)
    
    Returns:
        Modified data with tariff effects
    """
    # Get current imports and exports
    imports = data.loc[current_quarter, 'emn']  # Nominal imports
    exports = data.loc[current_quarter, 'exn']  # Nominal exports
    print(f"Quarter: {current_quarter} with imports {imports}")
    print(f"Quarter: {current_quarter} with exports {exports}")
    # Calculate tariff revenue
    tariff_revenue = imports * tariff_rate
    print(f"Quarter: {current_quarter} with tariff revenue {tariff_revenue}")
    
    # Add tariff revenue to government receipts
    data.loc[current_quarter, 'gfrecn'] += tariff_revenue
    
    # Reduce imports due to higher prices (simple elasticity)
    import_elasticity = -0.5  # This is a simplified assumption
    import_reduction = imports * (1 + tariff_rate * import_elasticity)
    data.loc[current_quarter, 'emn'] = import_reduction
    
    # Potential retaliation effects on exports (optional)
    retaliation_factor = 0.5  # Assumption: 50% reciprocal tariff
    export_reduction = exports * (1 - (tariff_rate * retaliation_factor))
    data.loc[current_quarter, 'exn'] = export_reduction
    
    return data

now = time.strftime("%Y%m%d_%H%M%S")
# Modified simulation loop
for rep in range(nrepl):
    sim_data = data.copy()
    
    # Compute baseline tracking factors
    with_adds = frbus.init_trac(residstart, simend, sim_data)
    
    # Quarterly policy decisions
    for quarter in pd.date_range(start=simstart, end=simend, freq='Q'):
        q = (quarter.month - 1) // 3 + 1
        quarter_str = f"{quarter.year}q{q}".lower() 
        print("--------------------------------")
        print("USA Economic Indicators Before Policy Actions")
        print(f"Quarter: {quarter_str} {quarter} with GDP growth (2012$) {sim_data.loc[quarter_str, 'hggdp']}")
        print(f"Quarter: {quarter_str} {quarter} with Nominal GDP {sim_data.loc[quarter_str, 'xgdpn']}")
        print(f"Quarter: {quarter_str} {quarter} with Real GDP {sim_data.loc[quarter_str, 'xgdp']}")
        print(f"Quarter: {quarter_str} {quarter} with Personal tax revenues {sim_data.loc[quarter_str, 'tpn']}")
        print(f"Quarter: {quarter_str} {quarter} with Corporate tax revenues {sim_data.loc[quarter_str, 'tcin']}")
        print(f"Quarter: {quarter_str} {quarter} with Personal tax revenues rates {sim_data.loc[quarter_str, 'trp']}")
        print(f"Quarter: {quarter_str} {quarter} with Corporate tax revenues rates {sim_data.loc[quarter_str, 'trci']}")
        print(f"Quarter: {quarter_str} {quarter} with Trend ratio of transfer payments to GDP {sim_data.loc[quarter_str, 'gtrt']}")
        print(f"Quarter: {quarter_str} {quarter} with Trend level of federal government expenditures {sim_data.loc[quarter_str, 'egfet']}")
        print(f"Quarter: {quarter_str} {quarter} with Federal Reserve Short-term Interest rate {sim_data.loc[quarter_str, 'frs10']}")
        print(f"Quarter: {quarter_str} {quarter} with Current PCI value: {sim_data.loc[quarter_str, 'pcpi']}")

        annual_target = 2.0  # 2% annual inflation target
        quarterly_target = ((1 + annual_target/100)**(1/4) - 1) * 100
        # This gives approximately 0.495% per quarter
        # Define target values
        targets = {
            'pcpi': quarterly_target,    # 2% inflation target
            'lur': 4.0,     # 4% unemployment target
            'hggdp': 4.0,    # 4% GDP growth target
            'frs10': 0.5,  # Limit on interest rate changes
            'gfdbtn': 80,  # Debt-to-GDP target
        }

        # Get current state
        state = get_state(sim_data, quarter_str)
        
        # Get PPO action
        actions, log_probs, state_value = ppo_agent.forward(state)
        
        # Apply actions to the data
        sim_data = apply_actions(sim_data, quarter_str, actions)
        
        # Apply tariff (if active)
        tariff_rate = 0.5  # 50% tariff
        sim_data = apply_tariff(sim_data, quarter_str, tariff_rate)
        print("--------------------------------")
        print("USA Economic Targets After Policy Actions")# Calculate percentage change directly
        print(f"Quarter: {quarter_str} {quarter} with GDP growth (2012$) {sim_data.loc[quarter_str, 'hggdp']}")
        print(f"Quarter: {quarter_str} {quarter} with Nominal GDP {sim_data.loc[quarter_str, 'xgdpn']}")
        print(f"Quarter: {quarter_str} {quarter} with Real GDP {sim_data.loc[quarter_str, 'xgdp']}")
        print(f"Quarter: {quarter_str} {quarter} with Personal tax revenues {sim_data.loc[quarter_str, 'tpn']}")
        print(f"Quarter: {quarter_str} {quarter} with Corporate tax revenues {sim_data.loc[quarter_str, 'tcin']}")
        print(f"Quarter: {quarter_str} {quarter} with Personal tax revenues rates {sim_data.loc[quarter_str, 'trp']}")
        print(f"Quarter: {quarter_str} {quarter} with Corporate tax revenues rates {sim_data.loc[quarter_str, 'trci']}")
        print(f"Quarter: {quarter_str} {quarter} with Trend ratio of transfer payments to GDP {sim_data.loc[quarter_str, 'gtrt']}")
        print(f"Quarter: {quarter_str} {quarter} with Trend level of federal government expenditures {sim_data.loc[quarter_str, 'egfet']}")
        print(f"Quarter: {quarter_str} {quarter} with Federal Reserve Short-term Interest rate {sim_data.loc[quarter_str, 'frs10']}")
        print(f"Quarter: {quarter_str} {quarter} with Current PCI value: {sim_data.loc[quarter_str, 'pcpi']}")

        pct_change = sim_data['pcpi'].pct_change().loc[quarter_str] * 100

        print(f"Quarter: {quarter_str} {quarter} with PCI Inflation Percentage Change target {targets['pcpi']} compared to previous quarter: {pct_change:.2f}")
        print(f"Quarter: {quarter_str} {quarter} with Unemployment target {targets['lur']} compared to {sim_data.loc[quarter_str, 'lur']}")
        print(f"Quarter: {quarter_str} {quarter} with GDP growth target {targets['hggdp']} compared to {sim_data.loc[quarter_str, 'hggdp']}")
        print(f"Quarter: {quarter_str} {quarter} with Interest rate target {targets['frs10']} compared to {sim_data.loc[quarter_str, 'frs10']}")
        print(f"Quarter: {quarter_str} {quarter} with Debt-to-GDP target {targets['gfdbtn']} compared to {sim_data.loc[quarter_str, 'gfdbtn'] / sim_data.loc[quarter_str, 'xgdpn'] * 100}")

        
        # Run one quarter of simulation
        try:
            solution = frbus.stochsim(
                1, with_adds, quarter_str, quarter_str,
                residstart, residend, nextra=1
            )[0]
            
            # Calculate reward based on economic outcomes 
            reward = calculate_reward(solution, quarter_str)
            
            # Store experience for PPO update
            experience = {
                'state': state,
                'actions': torch.tensor(actions).float(),
                'log_probs': log_probs,
                'reward': torch.tensor(reward).float(),
                'value': state_value,
                'done': quarter_str == simend
            }   
        except Exception as e:
            print(f"Simulation stochsim failed for quarter {quarter_str}: {e}")
            raise
        
        try:
            experiences.append(experience)
            print(f"Experience added for quarter {quarter_str} with experience {experience}") 
            
            # Update PPO agent after every 8 quarters (2 years)
            if len(experiences) >= 8:
                print(f"Updating PPO agent after collecting {len(experiences)} quarters of experience")
                ppo_agent = update_ppo(ppo_agent, experiences)
                experiences = []  # Clear experiences after update
        
            # Save checkpoint every 5 years (after every 20 quarters)
            current_quarter = pd.to_datetime(quarter_str)
            quarters_since_start = (current_quarter - pd.to_datetime(simstart)).days / 365.25 * 4
            
        except Exception as e:
            print(f"Simulation experience failed for quarter {quarter_str}: {e}") 
            raise
        
        try:
            if quarters_since_start >= 4 and quarters_since_start % 4 == 0: 
                checkpoint_path = f'checkpoints_{now}/ppo_agent_year_{quarter_str}_replication_{rep}.pt'
                os.makedirs(f'checkpoints_{now}', exist_ok=True)
                
                # Save the model state
                checkpoint = {
                    'actor_state_dict': ppo_agent.actor.state_dict(),
                    'critic_state_dict': ppo_agent.critic.state_dict(),
                    'year': int(quarters_since_start/4),
                    'quarter': quarter_str,
                    'action_bounds': ppo_agent.action_bounds
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at year {int(quarters_since_start/4)} ({quarter_str})")
                
        except Exception as e:
            print(f"Simulation checkpoint failed for quarter {quarter_str}: {e}")
            raise
    
   

def load_checkpoint(path, ppo_agent):
    checkpoint = torch.load(path)
    ppo_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"Loaded checkpoint from year {checkpoint['year']} ({checkpoint['quarter']})")
    return ppo_agent

