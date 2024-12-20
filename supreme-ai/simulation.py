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

import json
from typing import List, Dict, Optional
import aiohttp
import asyncio
from datetime import datetime

# Import the SimulationAPI class
from api_client import SimulationAPI

policy_vars = ['frs10', 'gtrt', 'egfet', 'trp', 'trci']

async def run_simulation_function(ppo_agent, simulation_start, simulation_end, simulation_replications, key_checkpoint_path):
    # Initialize API client
    async with SimulationAPI() as api_client:

        # Load data and model
        data = load_data("../data/LONGBASE.TXT")
        frbus = Frbus("../models/model.xml")

        # Initialize PPO agent
        # state_dim = number of economic indicators you're monitoring
        # action_dim = number of policy tools 

        # Simulation parameters
        residstart = "1975q1"
        residend = "2018q4"
        simstart = simulation_start  # Starting the simulation from the checkpoint
        simend = simulation_end 

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
            
            # Get current and previous quarter imports and exports
            current_imports = data.loc[current_quarter, 'emn']  # Current nominal imports
            prev_imports = data.shift(1).loc[current_quarter, 'emn']  # Previous quarter imports
            
            current_exports = data.loc[current_quarter, 'exn']  # Current nominal exports
            prev_exports = data.shift(1).loc[current_quarter, 'exn']  # Previous quarter exports
            
            print(f"Quarter: {current_quarter} with imports: Current {current_imports:.3f}, Previous Quarter {prev_imports:.3f} (Change: {current_imports - prev_imports:.3f})")
            print(f"Quarter: {current_quarter} with exports: Current {current_exports:.3f}, Previous Quarter {prev_exports:.3f} (Change: {current_exports - prev_exports:.3f})")
            
            # Calculate current and previous tariff revenue
            current_tariff_revenue = current_imports * tariff_rate
            prev_tariff_revenue = prev_imports * tariff_rate
            
            print(f"Quarter: {current_quarter} with tariff revenue: Current {current_tariff_revenue:.3f}, Previous Quarter {prev_tariff_revenue:.3f} (Change: {current_tariff_revenue - prev_tariff_revenue:.3f})")
            
            return data
    
        # Modified simulation loop
        for rep in range(nrepl):
            sim_data = data.copy()
            sim_data_without_tariff = data.copy()
            sim_data_without_rl = data.copy()
            # Quarterly policy decisions
            for quarter in pd.date_range(start=simstart, end=simend, freq='Q'):
                q = (quarter.month - 1) // 3 + 1
                quarter_str = f"{quarter.year}q{q}".lower() 
                # Store pre-policy values
                old_variables = {
                    'hggdp': sim_data.loc[quarter_str, 'hggdp'],
                    'xgdpn': sim_data.loc[quarter_str, 'xgdpn'],
                    'xgdp': sim_data.loc[quarter_str, 'xgdp'],
                    'tpn': sim_data.loc[quarter_str, 'tpn'],
                    'tcin': sim_data.loc[quarter_str, 'tcin'],
                    'trp': sim_data.loc[quarter_str, 'trp'],
                    'trci': sim_data.loc[quarter_str, 'trci'],
                    'gtrt': sim_data.loc[quarter_str, 'gtrt'],
                    'egfet': sim_data.loc[quarter_str, 'egfet'],
                    'frs10': sim_data.loc[quarter_str, 'frs10'],
                    'pcpi': sim_data.loc[quarter_str, 'pcpi'],
                    'emn': sim_data.loc[quarter_str, 'emn'],
                    'exn': sim_data.loc[quarter_str, 'exn']
                }
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
                state_without_tariff = get_state(sim_data_without_tariff, quarter_str) 
                
                # Get PPO action
                actions, log_probs, state_value = ppo_agent.forward(state)

                actions_without_tariff, log_probs_without_tariff, state_value_without_tariff = ppo_agent.forward(state_without_tariff)
                
                # Apply actions to the data
                sim_data = apply_actions(sim_data, quarter_str, actions)
                sim_data_without_tariff = apply_actions(sim_data_without_tariff, quarter_str, actions_without_tariff)

                # Apply tariff (if active)
                tariff_rate = 0.5  # 50% tariff
                sim_data = apply_tariff(sim_data, quarter_str, tariff_rate)
            
                # Compute baseline tracking factors
                with_adds = frbus.init_trac(residstart, simend, sim_data)
                with_adds_without_tariff = frbus.init_trac(residstart, simend, sim_data_without_tariff)
                with_adds_without_rl = frbus.init_trac(residstart, simend, sim_data_without_rl)


                # Run one quarter of simulation
                try:
                    solution = frbus.stochsim(
                        1, with_adds, quarter_str, quarter_str,
                        residstart, residend, nextra=1
                    )[0]

                    solution_without_tariff = frbus.stochsim(
                        1, with_adds_without_tariff, quarter_str, quarter_str,
                        residstart, residend, nextra=1
                    )[0]

                    solution_without_rl = frbus.stochsim(
                        1, with_adds_without_rl, quarter_str, quarter_str,
                        residstart, residend, nextra=1
                    )[0]


                    # Send metrics update to API
                    await api_client.send_metrics_update(
                        solution=solution,
                        solution_without_tariff=solution_without_tariff,
                        solution_without_rl=solution_without_rl,
                        quarter_str=quarter_str,
                        targets=targets
                    )
                    print("--------------------------------")
                    print("USA Economic Macro Indicators After Policy Actions and Simulation") 
                    
                    # Create dictionary mapping variables to their descriptions
                    variable_descriptions = {
                        'hggdp': 'GDP growth (2012$)',
                        'xgdpn': 'Nominal GDP',
                        'xgdp': 'Real GDP',
                        'tpn': 'Personal tax revenues',
                        'tcin': 'Corporate tax revenues',
                        'trp': 'Personal tax revenues rates',
                        'trci': 'Corporate tax revenues rates',
                        'gtrt': 'Trend ratio of transfer payments to GDP',
                        'egfet': 'Trend level of federal government expenditures',
                        'frs10': 'Federal Reserve Short-term Interest rate',
                        'pcpi': 'Current PCI value',
                        'lur': 'Unemployment rate',
                        'gfdbtn': 'Debt-to-GDP ratio',
                        'emn': 'Imports Revenue',
                        'exn': 'Exports Revenue'
                    }

                    # Compare solutions for each variable
                    for var in ['hggdp', 'xgdpn', 'xgdp', 'tpn', 'tcin', 'emn', 'exn']:
                        print(f"\nComparison for {variable_descriptions[var]} ({var}) in {quarter_str} {quarter}:")
                        
                        # RL Decision Maker
                        current_rl = solution.loc[quarter_str, var]
                        prev_rl = solution.shift(1).loc[quarter_str, var]
                        print(f"RL Decision Maker: Current {current_rl:.3f}, Previous Quarter {prev_rl:.3f} (Change: {current_rl - prev_rl:.3f})")
                        
                        # RL without tariff
                        current_notariff = solution_without_tariff.loc[quarter_str, var]
                        prev_notariff = solution_without_tariff.shift(1).loc[quarter_str, var]
                        print(f"RL Decision Maker without tariff: Current {current_notariff:.3f}, Previous Quarter {prev_notariff:.3f} (Change: {current_notariff - prev_notariff:.3f})")
                        print(f"  Diff from RL Decision Maker: Current {current_notariff - current_rl:.3f}, Previous Quarter {prev_notariff - prev_rl:.3f}")
                        
                        # Base without RL and tariff
                        current_base = solution_without_rl.loc[quarter_str, var]
                        prev_base = solution_without_rl.shift(1).loc[quarter_str, var]
                        print(f"Base Simulation without RL and Tariff: Current {current_base:.3f}, Previous Quarter {prev_base:.3f} (Change: {current_base - prev_base:.3f})")
                        print(f"  Diff from RL Decision Maker: Current {current_base - current_rl:.3f}, Previous Quarter {prev_base - prev_rl:.3f}")

                    print("\nPolicy Action Variables Comparison:")
                    for var in ['trp', 'trci', 'gtrt', 'egfet', 'frs10', 'pcpi']:
                        print(f"\n{variable_descriptions[var]} ({var}) in {quarter_str} {quarter}:")
                        # RL Decision Maker
                        current_rl = solution.loc[quarter_str, var]
                        prev_rl = solution.shift(1).loc[quarter_str, var]
                        print(f"RL Decision Maker: Current {current_rl:.3f}, Previous Quarter {prev_rl:.3f} (Change: {current_rl - prev_rl:.3f})")
                        
                        # RL without tariff
                        current_notariff = solution_without_tariff.loc[quarter_str, var]
                        prev_notariff = solution_without_tariff.shift(1).loc[quarter_str, var]
                        print(f"RL Decision Maker without tariff: Current {current_notariff:.3f}, Previous Quarter {prev_notariff:.3f} (Change: {current_notariff - prev_notariff:.3f})")
                        print(f"  Diff from RL Decision Maker: Current {current_notariff - current_rl:.3f}, Previous Quarter {prev_notariff - prev_rl:.3f}")
                        
                        # Base without RL and tariff
                        current_base = solution_without_rl.loc[quarter_str, var]
                        prev_base = solution_without_rl.shift(1).loc[quarter_str, var]
                        print(f"Base Simulation without RL and Tariff: Current {current_base:.3f}, Previous Quarter {prev_base:.3f} (Change: {current_base - prev_base:.3f})")
                        print(f"  Diff from RL Decision Maker: Current {current_base - current_rl:.3f}, Previous Quarter {prev_base - prev_rl:.3f}")

                    print("\n--------------------------------")
                    print("USA Economic Targets Comparison")
                    
                    # Compare targets across solutions
                    print(f"\nPCI Inflation Rate:")
                    for sol, name in [(solution, "RL Decision Maker"), (solution_without_tariff, "RL Decision Maker without tariff"), (solution_without_rl, "Base Simulation without RL and Tariff")]:
                        current = sol.loc[quarter_str, 'pcpi']
                        prev = sol.shift(1).loc[quarter_str, 'pcpi']
                        pct_change = ((current - prev) / prev) * 100
                        print(f"{name}: Current {current:.3f}, Previous {prev:.3f} (Change: {pct_change:.2f}%) (Target: {targets['pcpi']}%)")

                    print(f"\nUnemployment Rate:")
                    for sol, name in [(solution, "RL Decision Maker"), (solution_without_tariff, "RL Decision Maker without tariff"), (solution_without_rl, "Base Simulation without RL and Tariff")]:
                        current = sol.loc[quarter_str, 'lur']
                        prev = sol.shift(1).loc[quarter_str, 'lur']
                        print(f"{name}: Current {current:.2f}%, Previous {prev:.2f}% (Change: {current - prev:.2f}%) (Target: {targets['lur']}%)")

                    print(f"\nGDP Growth:")
                    for sol, name in [(solution, "RL Decision Maker"), (solution_without_tariff, "RL Decision Maker without tariff"), (solution_without_rl, "Base Simulation without RL and Tariff")]:
                        current = sol.loc[quarter_str, 'hggdp']
                        prev = sol.shift(1).loc[quarter_str, 'hggdp']
                        print(f"{name}: Current {current:.2f}%, Previous {prev:.2f}% (Change: {current - prev:.2f}%) (Target: {targets['hggdp']}%)")

                    print(f"\nInterest Rate:")
                    for sol, name in [(solution, "RL Decision Maker"), (solution_without_tariff, "RL Decision Maker without tariff"), (solution_without_rl, "Base Simulation without RL and Tariff")]:
                        current = sol.loc[quarter_str, 'frs10']
                        prev = sol.shift(1).loc[quarter_str, 'frs10']
                        print(f"{name}: Current {current:.2f}%, Previous {prev:.2f}% (Change: {current - prev:.2f}%) (Target: {targets['frs10']}%)")

                    print(f"\nDebt-to-GDP Ratio:")
                    for sol, name in [(solution, "RL Decision Maker"), (solution_without_tariff, "RL Decision Maker without tariff"), (solution_without_rl, "Base Simulation without RL and Tariff")]:
                        current_debt_gdp = sol.loc[quarter_str, 'gfdbtn'] / sol.loc[quarter_str, 'xgdpn'] * 100
                        prev_debt_gdp = sol.shift(1).loc[quarter_str, 'gfdbtn'] / sol.shift(1).loc[quarter_str, 'xgdpn'] * 100
                        print(f"{name}: Current {current_debt_gdp:.2f}%, Previous {prev_debt_gdp:.2f}% (Change: {current_debt_gdp - prev_debt_gdp:.2f}%) (Target: {targets['gfdbtn']}%)")

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
                        checkpoint_path = f'checkpoints_{key_checkpoint_path}/ppo_agent_year_{quarter_str}_replication_{rep}.pt'
                        os.makedirs(f'checkpoints_{key_checkpoint_path}', exist_ok=True)
                        
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
        
        return "Simulation completed successfully"

def load_checkpoint(path, ppo_agent):
    checkpoint = torch.load(path)
    ppo_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"Loaded checkpoint from year {checkpoint['year']} ({checkpoint['quarter']})")
    return ppo_agent

# Add the main execution block
async def main():
    # Your existing setup code - example values shown below
    ppo_agent = PPOAgent(state_dim=8, action_dim=5)  # Adjust dimensions as needed
    simulation_start = "2024q1"
    simulation_end = "2045q4"
    simulation_replications = 1000
    key_checkpoint_path = "baseline"
    
    result = await run_simulation_function(
        ppo_agent, 
        simulation_start, 
        simulation_end, 
        simulation_replications, 
        key_checkpoint_path
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())