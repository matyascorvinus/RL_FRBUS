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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the SimulationAPI class
from api_client import SimulationAPI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Economic FRB/US Simulation API",
    description="Real-time economic metrics streaming API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


policy_vars = ['rff', 'gtrt', 'egfen', 'trp', 'trci'] 

async def run_the_training_for_simulation_function(ppo_agent, simulation_start, simulation_end, simulation_replications, key_checkpoint_path):
    # Initialize API client
    async with SimulationAPI() as api_client:

        # Load data and model
        data = load_data("../data/LONGBASE.TXT")
        frbus = Frbus("../models/model.xml") 

        # Simulation parameters
        residstart = "1975q1"
        residend = "2018q4"
        simstart = simulation_start  # Starting the simulation from the checkpoint
        simend = simulation_end 

        # Number of replications
        nrepl = simulation_replications
        # Run up to 5 extra replications, in case of failures
        nextra = 1
        experiences = []
        # Policy settings
        data.loc[simstart:simend, "dfpdbt"] = 0
        data.loc[simstart:simend, "dfpsrp"] = 1
        # data.loc[simstart:simend, "eps_s"] = 0
        # data.loc[simstart:simend, "eps_i"] = 0
        def get_state(data, current_quarter):
            """Extract state variables from data"""  
            return np.array([
                data.loc[current_quarter, 'hggdp'],  # Growth rate of GDP, cw 2012$ (annual rate)
                data.loc[current_quarter, 'pcpi'],   # Inflation
                data.loc[current_quarter, 'lur'],    # Unemployment
                data.loc[current_quarter, 'rff'],  # Interest rate
                data.loc[current_quarter, 'gtrt'],  # Trend ratio of transfer payments to GDP.
                data.loc[current_quarter, 'egfen'],  # Trend level of federal government expenditures.
                data.loc[current_quarter, 'trp'],  # Personal tax revenues rates
                data.loc[current_quarter, 'trci'],  # Corporate tax revenues rates
                # Add other relevant state variables
            ])

        def apply_actions(data, current_quarter, actions):
            """Apply PPO agent's actions to the data"""
            # Map actions to specific policy variables
            for var, action in zip(policy_vars, actions):  
                if var == 'rff': 
                    data.loc[current_quarter, var] = action
                if var == 'trp':
                    # trp * (ypn - gtn)
                    # if data.loc[current_quarter, var] + action > 0.025:
                    data.loc[current_quarter, var] = action                
                # gtn - gtn_aerr = .01*pgdp*gtr
                elif var == 'gtr':
                    data.loc[current_quarter, var] = action
                else:
                    data.loc[current_quarter, var] = action 
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
            # Calculate tariff revenue
            tariff_revenue = imports * tariff_rate
            # logger.info(f"Quarter: {current_quarter} with tariff revenue {tariff_revenue}")
            
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
    
        # Modified simulation loop
        for rep in range(nrepl):
            sim_data = data.copy()
            sim_data_without_tariff = data.copy()
            sim_data_without_rl = data.copy()
            initial_simulation = True
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
                    'egfen': sim_data.loc[quarter_str, 'egfen'],
                    'rff': sim_data.loc[quarter_str, 'rff'],
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
                    'rff': 0.5,  # Limit on interest rate changes
                    'gfdbtn': 80,  # Debt-to-GDP target
                }

                # Get current state
                state = get_state(sim_data, quarter_str)
                state_without_tariff = get_state(sim_data_without_tariff, quarter_str) 
                
                # Get PPO action
                actions, log_probs, state_value = ppo_agent.forward(state)

                actions_without_tariff, log_probs_without_tariff, state_value_without_tariff = ppo_agent.forward(state_without_tariff)
                

                # Apply tariff (if active)
                tariff_rate = 0.5  # 50% tariff
                sim_data = apply_tariff(sim_data, quarter_str, tariff_rate)
            
                # Compute baseline tracking factors
                if initial_simulation:
                    with_adds = frbus.init_trac(residstart, simend, sim_data)
                    with_adds_without_tariff = frbus.init_trac(residstart, simend, sim_data_without_tariff)
                    with_adds_without_rl = frbus.init_trac(residstart, simend, sim_data_without_rl)
                
                # Apply actions to the data
                with_adds = apply_actions(with_adds, quarter_str, actions)
                with_adds_without_tariff = apply_actions(with_adds_without_tariff, quarter_str, actions_without_tariff)
                
                
                # Run one quarter of simulation
                try: 

                    solution = frbus.solve(quarter_str, quarter_str, with_adds)
                    solution_without_tariff = frbus.solve(quarter_str, quarter_str, with_adds_without_tariff)
                    solution_without_rl = frbus.solve(quarter_str, quarter_str, with_adds_without_rl)
                    initial_simulation = False
                    for element in solution:
                        # repleace the solution to the with_adds 
                        with_adds.loc[quarter_str, element] = solution.loc[quarter_str, element]
                        with_adds_without_tariff.loc[quarter_str, element] = solution_without_tariff.loc[quarter_str, element]
                        with_adds_without_rl.loc[quarter_str, element] = solution_without_rl.loc[quarter_str, element]
                    # Send metrics update to API
                    await api_client.send_metrics_update(
                        solution=solution,
                        solution_without_tariff=solution_without_tariff,
                        solution_without_rl=solution_without_rl,
                        quarter_str=quarter_str.upper(),
                        targets=targets
                    )
                    logger.info("--------------------------------")
                    logger.info("USA Economic Macro Indicators After Policy Actions and Simulation") 
                    
                    
                    # Calculate reward based on economic outcomes 
                    reward = calculate_reward(solution, solution_without_rl, quarter_str, simend)
                    
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
                    logger.error(f"Simulation stochsim failed for quarter {quarter_str}: {e}")
                    raise
                
                try:
                    experiences.append(experience)
                    logger.info(f"Experience added for quarter {quarter_str} with experience {experience}") 
                    
                    # Update PPO agent after every 8 quarters (2 years)
                    if len(experiences) >= 8:
                        logger.info(f"Updating PPO agent after collecting {len(experiences)} quarters of experience")
                        ppo_agent = update_ppo(ppo_agent, experiences)
                        experiences = []  # Clear experiences after update
                
                    # Save checkpoint every 5 years (after every 20 quarters)
                    current_quarter = pd.to_datetime(quarter_str)
                    quarters_since_start = (current_quarter - pd.to_datetime(simstart)).days / 365.25 * 4
                    
                except Exception as e:
                    logger.error(f"Simulation experience failed for quarter {quarter_str}: {e}") 
                    raise
                
            try: 
                checkpoint_path = f'checkpoints_{key_checkpoint_path}/ppo_agent_replication_{rep}.pt'
                os.makedirs(f'checkpoints_{key_checkpoint_path}', exist_ok=True)
                
                # Save the model state
                checkpoint = {
                    'actor_state_dict': ppo_agent.actor.state_dict(),
                    'critic_state_dict': ppo_agent.critic.state_dict(),
                    'replication': rep,
                    'action_bounds': ppo_agent.action_bounds
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint at replication {rep}")
                    
            except Exception as e:
                logger.error(f"Simulation checkpoint failed for replication {rep}: {e}")
                raise
    
        return "Simulation completed successfully"


async def run_the_simulation_with_one_replication(ppo_agent, simulation_start, simulation_end, key_checkpoint_path):
    # Initialize API client
    async with SimulationAPI() as api_client:

        # Load data and model
        data = load_data("../data/LONGBASE.TXT")
        frbus = Frbus("../models/model.xml") 

        # Simulation parameters
        residstart = "1975q1"
        residend = "2018q4"
        simstart = simulation_start  # Starting the simulation from the checkpoint
        simend = simulation_end 

        # Number of replications
        nrepl = 1
        # Run up to 5 extra replications, in case of failures
        nextra = 1
        experiences = []
        # Policy settings
        data.loc[simstart:simend, "dfpdbt"] = 0
        data.loc[simstart:simend, "dfpsrp"] = 1
        # data.loc[simstart:simend, "eps_s"] = 0
        # data.loc[simstart:simend, "eps_i"] = 0
        def get_state(data, current_quarter):
            """Extract state variables from data"""  
            return np.array([
                data.loc[current_quarter, 'hggdp'],  # Growth rate of GDP, cw 2012$ (annual rate)
                data.loc[current_quarter, 'pcpi'],   # Inflation
                data.loc[current_quarter, 'lur'],    # Unemployment
                data.loc[current_quarter, 'rff'],  # Interest rate
                data.loc[current_quarter, 'gtrt'],  # Trend ratio of transfer payments to GDP.
                data.loc[current_quarter, 'egfen'],  # Trend level of federal government expenditures.
                data.loc[current_quarter, 'trp'],  # Personal tax revenues rates
                data.loc[current_quarter, 'trci'],  # Corporate tax revenues rates
                # Add other relevant state variables
            ])

        def apply_actions(data, current_quarter, actions):
            """Apply PPO agent's actions to the data"""
            # Map actions to specific policy variables
            for var, action in zip(policy_vars, actions):  
                if var == 'rff': 
                    data.loc[current_quarter, var] = action
                if var == 'trp':
                    # trp * (ypn - gtn)
                    # if data.loc[current_quarter, var] + action > 0.025:
                    data.loc[current_quarter, var] = action                
                # gtn - gtn_aerr = .01*pgdp*gtr
                elif var == 'gtr':
                    data.loc[current_quarter, var] = action
                else:
                    data.loc[current_quarter, var] = action 
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
            # Calculate tariff revenue
            tariff_revenue = imports * tariff_rate
            # logger.info(f"Quarter: {current_quarter} with tariff revenue {tariff_revenue}")
            
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
    
        # Modified simulation loop
        for rep in range(nrepl):
            sim_data = data.copy()
            sim_data_without_tariff = data.copy()
            sim_data_without_rl = data.copy()
            initial_simulation = True
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
                    'egfen': sim_data.loc[quarter_str, 'egfen'],
                    'rff': sim_data.loc[quarter_str, 'rff'],
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
                    'rff': 0.5,  # Limit on interest rate changes
                    'gfdbtn': 80,  # Debt-to-GDP target
                }

                # Get current state
                state = get_state(sim_data, quarter_str)
                state_without_tariff = get_state(sim_data_without_tariff, quarter_str) 
                
                # Get PPO action
                actions, log_probs, state_value = ppo_agent.forward(state)

                actions_without_tariff, log_probs_without_tariff, state_value_without_tariff = ppo_agent.forward(state_without_tariff)
                

                # Apply tariff (if active)
                tariff_rate = 0.5  # 50% tariff
                sim_data = apply_tariff(sim_data, quarter_str, tariff_rate)
            
                # Compute baseline tracking factors
                if initial_simulation:
                    with_adds = frbus.init_trac(residstart, simend, sim_data)
                    with_adds_without_tariff = frbus.init_trac(residstart, simend, sim_data_without_tariff)
                    with_adds_without_rl = frbus.init_trac(residstart, simend, sim_data_without_rl)
                
                # Apply actions to the data
                with_adds = apply_actions(with_adds, quarter_str, actions)
                with_adds_without_tariff = apply_actions(with_adds_without_tariff, quarter_str, actions_without_tariff)
                
                
                # Run one quarter of simulation
                try: 

                    solution = frbus.solve(quarter_str, quarter_str, with_adds)
                    solution_without_tariff = frbus.solve(quarter_str, quarter_str, with_adds_without_tariff)
                    solution_without_rl = frbus.solve(quarter_str, quarter_str, with_adds_without_rl)
                    initial_simulation = False
                    for element in solution:
                        # repleace the solution to the with_adds 
                        with_adds.loc[quarter_str, element] = solution.loc[quarter_str, element]
                        with_adds_without_tariff.loc[quarter_str, element] = solution_without_tariff.loc[quarter_str, element]
                        with_adds_without_rl.loc[quarter_str, element] = solution_without_rl.loc[quarter_str, element]
                    # Send metrics update to API
                    await api_client.send_metrics_update(
                        solution=solution,
                        solution_without_tariff=solution_without_tariff,
                        solution_without_rl=solution_without_rl,
                        quarter_str=quarter_str.upper(),
                        targets=targets
                    )
                    logger.info("--------------------------------")
                    logger.info("USA Economic Macro Indicators After Policy Actions and Simulation") 
                    
                    
                    # Calculate reward based on economic outcomes 
                    reward = calculate_reward(solution, solution_without_rl, quarter_str, simend)
                     
                except Exception as e:
                    logger.error(f"Simulation stochsim failed for quarter {quarter_str}: {e}")
                    raise
                 
    
        return "Simulation completed successfully"


def load_checkpoint(path, ppo_agent):
    try:
        checkpoint = torch.load(path)
        ppo_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        logger.info(f"Loaded checkpoint from year {checkpoint['year']} ({checkpoint['quarter']})")
        return ppo_agent
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}") 
        return ppo_agent

# Add the main execution block
async def main_training():
    # Your existing setup code - example values shown below
    key_checkpoint_path = "trump"
    checkpoint_path = f"checkpoints_{key_checkpoint_path}/ppo_agent_replication_53.pt"
    ppo_agent = load_checkpoint(checkpoint_path, PPOAgent(state_dim=8, action_dim=len(policy_vars)))
    # ppo_agent = PPOAgent(state_dim=8, action_dim=len(policy_vars))  # Adjust dimensions as needed
    simulation_start = "2024q1"
    simulation_end = "2044q1"
    simulation_replications = 1000
    
    result = await run_the_training_for_simulation_function(
        ppo_agent, 
        simulation_start, 
        simulation_end, 
        simulation_replications, 
        key_checkpoint_path
    )
    logger.info(result)

# Add the main execution block
async def main_simulation():
    # Your existing setup code - example values shown below
    key_checkpoint_path = "trump"
    checkpoint_path = f"checkpoints_{key_checkpoint_path}/ppo_agent_replication_53.pt"
    ppo_agent = load_checkpoint(checkpoint_path, PPOAgent(state_dim=8, action_dim=len(policy_vars)))
    # ppo_agent = PPOAgent(state_dim=8, action_dim=len(policy_vars))  # Adjust dimensions as needed
    simulation_start = "2024q1"
    simulation_end = "2044q1" 
    
    result = await run_the_simulation_with_one_replication(
        ppo_agent, 
        simulation_start, 
        simulation_end, 
        key_checkpoint_path
    )
    logger.info(result)

@app.get("/run_simulation_training")
def run_simulation_training():
    asyncio.run(main_training())
    return {"message": "Simulation completed successfully"}


@app.get("/run_simulation_with_one_replication")
def run_simulation_with_one_replication():
    asyncio.run(main_simulation())
    return {"message": "Simulation completed successfully"}
