import os
import shutil
from pyfrbus.frbus import Frbus
from pyfrbus.sim_lib import stochsim_plot
from pyfrbus.load_data import load_data
from ppo_agent import PPOAgent
import numpy as np
import torch
from env_function import calculate_reward, calculate_reward_policy, calculate_reward_policy_v1
import pandas as pd
import time

import json
from typing import List, Dict, Optional
import aiohttp
import asyncio
from datetime import datetime
import logging
import time

from models import EconomicMetrics, SimulationComparison
from active_learning_ppo import ActiveLearningPPOAgent
from active_learning_effective_relocation_ppo import ActiveLearningEffectiveRelocationPPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
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

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                await self.disconnect(connection)

# policy_vars = ['rff', 'gtrt', 'egfe', 'trptx', 'trcit']  # old policies

manager = ConnectionManager()

latest_metrics: Dict[str, SimulationComparison] = {}

async def broadcast_metrics_update(
    solution: pd.DataFrame,
    solution_without_tariff: pd.DataFrame,
    solution_without_rl: pd.DataFrame,
    solution_without_rl_tariff: pd.DataFrame,
    quarter_str: str,
    targets: dict
):
    """
    Directly broadcast metrics to websocket clients instead of making API calls
    """
    try:
        def create_metrics(df: pd.DataFrame) -> Dict:
            current = df.loc[quarter_str]
            previous = df.shift(1).loc[quarter_str]
            
            return {
                "quarter": quarter_str.upper(),
                "metrics": {
                    'hggdp': float(current['hggdp']),
                    'xgdpn': float(current['xgdpn']),
                    'xgdp': float(current['xgdp']),
                    'tpn': float(current['tpn']),
                    'tcin': float(current['tcin']),
                    'trptx': float(current['trptx']),
                    'trcit': float(current['trcit']),
                    'gtrt': float(current['gtrt']),
                    'egfe': float(current['egfe']),
                    'rff': float(current['rff']),
                    'pcpi': float(current['pcpi']),
                    'lur': float(current['lur']),
                    'gfdbtn': float(current['gfdbtn']),
                    'emn': float(current['emn']),
                    'exn': float(current['exn']),
                    'gtn': float(current['gtn']),
                    'gfsrpn': float(current['gfsrpn'])
                },
                "previous_metrics": {
                    'hggdp': float(previous['hggdp']),
                    'xgdpn': float(previous['xgdpn']),
                    'xgdp': float(previous['xgdp']),
                    'tpn': float(previous['tpn']),
                    'tcin': float(previous['tcin']),
                    'trptx': float(previous['trptx']),
                    'trcit': float(previous['trcit']),
                    'gtrt': float(previous['gtrt']),
                    'egfe': float(previous['egfe']),
                    'rff': float(previous['rff']),
                    'pcpi': float(previous['pcpi']),
                    'lur': float(previous['lur']),
                    'gfdbtn': float(previous['gfdbtn']),
                    'emn': float(previous['emn']),
                    'exn': float(previous['exn']),
                    'gtn': float(previous['gtn']),
                    'gfsrpn': float(previous['gfsrpn'])
                },
                "targets": targets
            }

        metrics_data = {
            "rl_decision": create_metrics(solution),
            "rl_decision_without_tariff": create_metrics(solution_without_tariff),
            "base_simulation": create_metrics(solution_without_rl),
            "base_simulation_with_tariff": create_metrics(solution_without_rl_tariff)
        }
        # Broadcast to all connected clients
        await manager.broadcast(metrics_data)
        
    except Exception as e:
        logger.error(f"Error broadcasting metrics: {e}")
        raise

policy_vars = ['gtrt', 'egfe', 'trptx', 'trcit']  
async def run_the_simulation_function(ppo_agent, ppo_agent_without_tariff, simulation_start, simulation_end, simulation_replications, key_checkpoint_path, is_training=True, replication_restart=0, highest_score= float('-inf'), tariff_rate=0):
    # Load data and model
    data = load_data("../data/LONGBASE.TXT")
    history_data = load_data("../data/HISTDATA.TXT")
    frbus = Frbus("../models/model.xml") 

    # Simulation parameters
    residstart = "1975q1"
    residend = "2024q4"
    simstart = simulation_start  # Starting the simulation from the checkpoint
    simend = simulation_end 

    # Number of replications
    nrepl = simulation_replications
    # Run up to 5 extra replications, in case of failures
    nextra = 1
    experiences = []
    experiences_without_tariff = []
    # Policy settings
    data.loc[simstart:simend, "dfpdbt"] = 0
    data.loc[simstart:simend, "dfpsrp"] = 1
    # data.loc[simstart:simend, "eps_s"] = 0
    # data.loc[simstart:simend, "eps_i"] = 0
    def get_state(data, current_quarter):
        """Extract state variables from data from the previous quarter"""   
        previous_quarter = pd.Period(current_quarter) - 1
        return data.loc[previous_quarter].values

    def apply_actions(data, current_quarter, actions):
        """Apply PPO agent's actions to the data""" 
        
        previous_quarter = pd.Period(current_quarter) - 1 
        # # Action bounds for different policy tools 
        # Map actions to specific policy variables
        # Convert actions tensor to numpy if it's a torch tensor
        if torch.is_tensor(actions):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions 
        for var, action in zip(policy_vars, actions_np):
            if var == 'rff':
                continue
            if var == 'egfe': 
                data.loc[current_quarter, var] = data.loc[previous_quarter, var] + action * 1000  
            else: 
                if data.loc[previous_quarter, var] + action > 0.1 and data.loc[previous_quarter, var] + action < 0.4:
                    data.loc[current_quarter, var] = data.loc[previous_quarter, var] + action
                else:
                    data.loc[current_quarter, var] = data.loc[previous_quarter, var] 
        return data

    def apply_tariff_enhanced(data, current_quarter, tariff_rate):
        """
        Apply tariff effects to the U.S. economy with more nuanced logic.
        
        Args:
            data: DataFrame containing economic variables (indexed by quarter).
            current_quarter: Current simulation quarter (string or datetime).
            tariff_rate: Base rate of tariff (e.g., 0.05 for 5%).
        
        Returns:
            Modified DataFrame with updated import/export data and govt. revenue.
        """
        if tariff_rate == 0:
            return data
        # 1) Access import and export levels
        imports = data.loc[current_quarter, 'emn']  # Nominal imports
        exports = data.loc[current_quarter, 'exn']  # Nominal exports
        
        # 2) Tiered or adaptive elasticity approach
        # For demonstration, let's do a logistic-based elasticity
        # That shifts from -0.7 to -0.3 over the range of tariff_rate [0, 0.2] as an example
        import_elasticity = -0.5 + 0.4 / (1 + np.exp(-(tariff_rate - 0.1)*10))
        
        # Optionally mix with a base elasticity from some table or categories
        # import_elasticity could be a weighted average of multiple sub-elasticities.
        
        # 3) Compute tariff revenue
        tariff_revenue = imports * tariff_rate
        data.loc[current_quarter, 'gfrecn'] += tariff_revenue
        
        # 4) Adjust imports with partial or immediate effect
        # immediate effect = imports * (1 + tariff_rate * import_elasticity)
        # partial approach = alpha * immediate + (1-alpha)*previous
        alpha = 0.5  # partial adjustment factor
        desired_import = imports * (1 + tariff_rate * import_elasticity)
        new_imports = alpha * desired_import + (1 - alpha) * imports
        
        data.loc[current_quarter, 'emn'] = max(new_imports, 0)  # ensure not negative
        
        # 5) More nuanced retaliation model
        # maybe the partner imposes a tariff that is 70% of ours once we exceed 5%:
        min_tariff_threshold = 0.05
        if tariff_rate >= min_tariff_threshold:
            partner_tariff = 0.7 * tariff_rate
        else:
            partner_tariff = 0.0

        # Then the elasticity of demand for U.S. exports might differ from our import elasticity
        # We do a simple approach for demonstration:
        export_elasticity = -0.4
        # If partner_tariff > 0, reduce exports
        desired_export = exports * (1 + partner_tariff * export_elasticity)
        new_exports = alpha * desired_export + (1 - alpha) * exports
        data.loc[current_quarter, 'exn'] = max(new_exports, 0)
        
        return data

    # Initialize variables for tracking best replication
    score_replications = {}
    the_best_replication = 0 
    score_replications_without_tariff = {}
    the_best_replication_without_tariff = 0
    highest_score_without_tariff = float('-inf')
    # Modified simulation loop
    for rep in range(nrepl):
        start_time = time.time()
        sim_data = data.copy()        
        simstart_year = int(simstart.split('q')[0])
        if 1970 <= simstart_year <= 2023 and not is_training:
            sim_data_without_tariff = history_data.copy()
        else: 
            sim_data_without_tariff = data.copy()
        sim_data_without_rl = data.copy()
        sim_data_without_rl_tariff = data.copy()
        initial_simulation = True
        total_reward = torch.tensor(0.0).clone().detach()
        total_reward_without_tariff = torch.tensor(0.0).clone().detach()
                
        # If using active learning agent, set up the FRB/US components
        if isinstance(ppo_agent, ActiveLearningPPOAgent):
            ppo_agent.set_frbus_components(
                frbus_model=frbus,
                apply_actions_fn=apply_actions,
                get_state_fn=get_state,
                calculate_reward_fn=calculate_reward_policy_v1
            )
            if rep == 0:
                # Initialize the uncertainty model at the start of simulation
                ppo_agent.initialize_uncertainty_model(
                    solutions=frbus.init_trac(residstart, simend, sim_data).copy(),
                    quarter_str=f"{simstart_year}q1".lower(),
                    solution_without_rl=frbus.solve(simstart, simend, frbus.init_trac(residstart, simend, sim_data_without_rl)),
                    simend=simend,
                    tariff_rate=tariff_rate
                )
        # If using active learning agent, set up the FRB/US components
        if isinstance(ppo_agent_without_tariff, ActiveLearningPPOAgent):
            ppo_agent_without_tariff.set_frbus_components(
                frbus_model=frbus,
                apply_actions_fn=apply_actions,
                get_state_fn=get_state,
                calculate_reward_fn=calculate_reward_policy_v1
            )
            
            if rep == 0:
                # Initialize the uncertainty model at the start of simulation
                ppo_agent_without_tariff.initialize_uncertainty_model(
                    solutions=frbus.init_trac(residstart, simend, sim_data_without_tariff).copy(),
                    quarter_str=f"{simstart_year}q1".lower(),
                    solution_without_rl=frbus.solve(simstart, simend, frbus.init_trac(residstart, simend, sim_data_without_rl_tariff)),
                    simend=simend,
                    tariff_rate=tariff_rate
                )
        annual_inflation_target = 2.0  # 2% annual inflation target
        quarterly_inflation_target = ((1 + annual_inflation_target/100)**(1/4) - 1) * 100
        # This gives approximately 0.495% per quarter
        # Define target values
        targets = {
            'pcpi': quarterly_inflation_target,    # 2% inflation target
            'lur': 4.0,     # 4% unemployment target
            'hggdp': 4.0,    # 4% GDP growth target
            'rff': 0.5,  # Limit on interest rate changes
            'gfdbtn': 80,  # Debt-to-GDP target
        } 
        # Quarterly policy decisions
        if not isinstance(ppo_agent, ActiveLearningEffectiveRelocationPPOAgent):
            for quarter in pd.date_range(start=simstart, end=simend, freq='Q'):
                q = (quarter.month - 1) // 3 + 1
                quarter_str = f"{quarter.year}q{q}".lower()  
                # Compute baseline tracking factors
                if initial_simulation:
                    # Policy settings
                    # Standard configuration, use surplus ratio targeting 
                    sim_data_without_rl.loc[simstart:simend, "dfpdbt"] = 0
                    sim_data_without_rl.loc[simstart:simend, "dfpsrp"] = 1
                    sim_data_without_rl_tariff = sim_data_without_rl.copy()

                    solutions = frbus.init_trac(residstart, simend, sim_data)
                    solutions_without_tariff = frbus.init_trac(residstart, simend, sim_data_without_tariff)
                    with_adds_without_rl = frbus.init_trac(residstart, simend, sim_data_without_rl)
                    with_adds_without_rl_tariff = frbus.init_trac(residstart, simend, sim_data_without_rl_tariff)

                    # 100 bp monetary policy shock and solve
                    with_adds_without_rl.loc[simstart, "rffintay_aerr"] += 1 
                    with_adds_without_rl_tariff.loc[simstart, "rffintay_aerr"] += 1 

                # Apply actions to the data
                if tariff_rate > 0:
                    solutions = apply_tariff_enhanced(solutions, quarter_str, tariff_rate)
                    sim_data_without_rl_tariff = apply_tariff_enhanced(sim_data_without_rl_tariff, quarter_str, tariff_rate)

                # Get current state
                state = get_state(solutions, quarter_str)
                state_without_tariff = get_state(solutions_without_tariff, quarter_str) 

                # Get PPO action with active learning if applicable
                if isinstance(ppo_agent, ActiveLearningPPOAgent):
                    # Active learning mode selects actions based on uncertainty/information gain
                    actions, log_probs, state_value, uncertainty = ppo_agent.forward_with_uncertainty(state)
                    logger.info(f"Action uncertainty: {uncertainty}")
                else:
                    # Standard PPO action selection
                    actions, log_probs, state_value = ppo_agent.forward(state)

                # Apply PPO actions to the data
                solutions = apply_actions(solutions, quarter_str, actions)
                
                # Apply PPO actions to the data without tariff
                if not (1970 <= simstart_year <= 2023 and not is_training):
                    if isinstance(ppo_agent_without_tariff, ActiveLearningPPOAgent):
                        actions_without_tariff, log_probs_without_tariff, state_value_without_tariff, uncertainty_without_tariff = ppo_agent_without_tariff.forward_with_uncertainty(state_without_tariff)
                        logger.info(f"Action uncertainty without tariff: {uncertainty_without_tariff}")
                    else:
                        actions_without_tariff, log_probs_without_tariff, state_value_without_tariff = ppo_agent_without_tariff.forward(state_without_tariff)
                    solutions_without_tariff = apply_actions(solutions_without_tariff, quarter_str, actions_without_tariff)
                 
                # Run one quarter of simulation
                try: 
                    solutions = frbus.solve(quarter_str, quarter_str, solutions)
                    solution_without_rl_tariff = frbus.solve(quarter_str, quarter_str, with_adds_without_rl_tariff)
                    if not (1970 <= simstart_year <= 2023 and not is_training):
                        solutions_without_tariff = frbus.solve(quarter_str, quarter_str, solutions_without_tariff)
                    
                    if initial_simulation: 
                        solution_without_rl = frbus.solve(simstart, simend, with_adds_without_rl) 
                    initial_simulation = False  
    

                    # Send metrics update to API
                    if not is_training:
                        await broadcast_metrics_update(
                            solution=solutions,
                            solution_without_tariff=solutions_without_tariff,
                            solution_without_rl=solution_without_rl,
                            solution_without_rl_tariff=solution_without_rl_tariff,
                            quarter_str=quarter_str.upper(),
                            targets=targets
                        )
                        
                    
                    # Calculate reward based on economic outcomes 
                    reward = calculate_reward_policy_v1(solutions, quarter_str, simend)    
                    if not (1970 <= simstart_year <= 2023 and not is_training):
                        reward_without_tariff = calculate_reward_policy_v1(solutions_without_tariff, quarter_str, simend)
                        total_reward_without_tariff += reward_without_tariff
                    total_reward += reward
                    
                    # Store experience for PPO update
                    if is_training:
                        if isinstance(ppo_agent, ActiveLearningPPOAgent):
                            # For active learning, we store the uncertainty along with other experience data
                            experience = {
                                'state': state,
                                'actions': torch.as_tensor(actions).float(),
                                'log_probs': log_probs,
                                'reward': torch.as_tensor(reward).float(),
                                'value': state_value,
                                'uncertainty': uncertainty,  # Store the uncertainty to weight this experience
                                'done': quarter_str.lower() == "2022q4".lower()
                            }
                        if isinstance(ppo_agent_without_tariff, ActiveLearningPPOAgent):
                            experience_without_tariff = {
                                'state': state_without_tariff,
                                'actions': torch.tensor(actions_without_tariff).float() if not (1970 <= simstart_year <= 2023) else torch.tensor(actions).float(),
                                'log_probs': log_probs_without_tariff if not (1970 <= simstart_year <= 2023) else log_probs,
                                'reward': torch.tensor(reward_without_tariff).float() if not (1970 <= simstart_year <= 2023) else torch.tensor(reward).float(),
                                'value': state_value_without_tariff if not (1970 <= simstart_year <= 2023) else state_value,
                                'done': quarter_str.lower() == "2022q4".lower()
                            }
                        if not isinstance(ppo_agent, ActiveLearningPPOAgent):
                            # Standard experience storing
                            experience = {
                                'state': state,
                                'actions': torch.as_tensor(actions).float(),
                                'log_probs': log_probs,
                                'reward': torch.as_tensor(reward).float(),
                                'value': state_value,
                                'done': quarter_str.lower() == "2022q4".lower()
                            }
                        if not isinstance(ppo_agent_without_tariff, ActiveLearningPPOAgent):
                            experience_without_tariff = {
                                'state': state_without_tariff,
                                'actions': torch.tensor(actions_without_tariff).float() if not (1970 <= simstart_year <= 2023) else torch.tensor(actions).float(),
                                'log_probs': log_probs_without_tariff if not (1970 <= simstart_year <= 2023) else log_probs,
                                'reward': torch.tensor(reward_without_tariff).float() if not (1970 <= simstart_year <= 2023) else torch.tensor(reward).float(),
                                'value': state_value_without_tariff if not (1970 <= simstart_year <= 2023) else state_value,
                                'done': quarter_str.lower() == "2022q4".lower()
                            }
                except Exception as e:
                    previous_quarter = pd.Period(quarter_str) - 1
                    logger.error(f"Actions of Decision Maker: {actions}") 
                    logger.error(f"Current quarter: {quarter_str}, Previous quarter: {previous_quarter}")
                    for var, action in zip(policy_vars, actions):
                        logger.error(f"Variable: {var}, Action: {action}, Previous Data: {solutions.loc[previous_quarter, var]}, Current Data: {solutions.loc[quarter_str, var]}")
                    logger.error(f"Simulation stochsim failed for quarter {quarter_str}: {e}")
                    raise

                if is_training:     
                    try:
                        experiences.append(experience) 
                        experiences_without_tariff.append(experience_without_tariff) 
                        
                        # Update PPO agent after every 40 quarters (10 years)
                        if len(experiences) >= 40:
                            if isinstance(ppo_agent, ActiveLearningPPOAgent):
                                # Update the agent with active learning, giving more weight to high-uncertainty experiences
                                ppo_agent.update_ppo_active_learning(experiences)
                                # Update the uncertainty model based on prediction errors
                                ppo_agent.update_uncertainty_model(experiences)

                            if isinstance(ppo_agent_without_tariff, ActiveLearningPPOAgent):
                                ppo_agent_without_tariff.update_ppo_active_learning(experiences_without_tariff)
                                ppo_agent_without_tariff.update_uncertainty_model(experiences_without_tariff)

                            if not isinstance(ppo_agent, ActiveLearningPPOAgent):
                                # Standard PPO update
                                ppo_agent.update_ppo(experiences)
                            if not isinstance(ppo_agent_without_tariff, ActiveLearningPPOAgent):
                                # Standard PPO update
                                ppo_agent_without_tariff.update_ppo(experiences_without_tariff)
                            experiences = []  # Clear experiences after update
                            experiences_without_tariff = []  # Clear experiences after update
                        
                    except Exception as e:
                        logger.error(f"Simulation experience failed for quarter {quarter_str}: {e}") 
                        raise

        if is_training: 
            end_time = time.time()                    
            if total_reward >= highest_score:
                highest_score = total_reward
                the_best_replication = rep + replication_restart
            if rep + replication_restart == the_best_replication:
                checkpoint_path = f'checkpoints_{key_checkpoint_path}/ppo_agent/ppo_agent_replication_{rep + replication_restart}.pt'
                os.makedirs(f'checkpoints_{key_checkpoint_path}/ppo_agent', exist_ok=True)
                checkpoint = {
                    'actor_state_dict': ppo_agent.actor.state_dict(),
                    'critic_state_dict': ppo_agent.critic.state_dict(),
                    'replication': rep + replication_restart,
                    'action_bounds': ppo_agent.action_bounds
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved the bestcheckpoint at replication {rep + replication_restart}")
                
            if total_reward_without_tariff >= highest_score_without_tariff:
                highest_score_without_tariff = total_reward_without_tariff
                the_best_replication_without_tariff = rep + replication_restart
            if rep + replication_restart == the_best_replication_without_tariff:
                checkpoint_path_without_tariff = f'checkpoints_{key_checkpoint_path}/ppo_agent_without_tariff/ppo_agent_replication_{rep + replication_restart}.pt'
                os.makedirs(f'checkpoints_{key_checkpoint_path}/ppo_agent_without_tariff', exist_ok=True)
                checkpoint_without_tariff = {
                    'actor_state_dict': ppo_agent_without_tariff.actor.state_dict(),
                    'critic_state_dict': ppo_agent_without_tariff.critic.state_dict(),
                    'replication': rep + replication_restart,
                    'action_bounds': ppo_agent_without_tariff.action_bounds
                }
                torch.save(checkpoint_without_tariff, checkpoint_path_without_tariff)
                logger.info(f"Saved the best checkpoint without tariff at replication {rep + replication_restart}")

            logger.info(f"Total reward for replication {rep + replication_restart}: {total_reward}")
            logger.info(f"Time taken for replication {rep + replication_restart}: {end_time - start_time} seconds")
            logger.info(f"Highest reward for replication {the_best_replication} with tariff: {highest_score}")
            logger.info(f"Highest reward for replication {the_best_replication_without_tariff} without tariff: {highest_score_without_tariff}")
            logger.info(f"ETA taken for remaining replications {nrepl - rep - replication_restart}: {(end_time - start_time) * (nrepl - rep - replication_restart)} seconds")

    if is_training:
        logger.info(f"The best replication is {the_best_replication} with score {highest_score}")
        logger.info(f"The best replication without tariff is {the_best_replication_without_tariff} with score {highest_score_without_tariff}")
        # Save the score replications to a text file
        with open(f'checkpoints_{key_checkpoint_path}/score_replications.txt', 'w') as f:
            f.write(str(score_replications))
        with open(f'checkpoints_{key_checkpoint_path}/score_replications_without_tariff.txt', 'w') as f:
            f.write(str(score_replications_without_tariff))
        with open(f'checkpoints_{key_checkpoint_path}/the_best_replication.txt', 'w') as f:
            f.write(str(the_best_replication))
        with open(f'checkpoints_{key_checkpoint_path}/the_best_replication_without_tariff.txt', 'w') as f:
            f.write(str(the_best_replication_without_tariff))
        # Copy the best checkpoint to the best_checkpoint folder
        os.makedirs(f'checkpoints_{key_checkpoint_path}/best_checkpoint', exist_ok=True)
        shutil.copy(f'checkpoints_{key_checkpoint_path}/ppo_agent/ppo_agent_replication_{the_best_replication}.pt', f'checkpoints_{key_checkpoint_path}/best_checkpoint/ppo_agent_best_replication.pt')
        shutil.copy(f'checkpoints_{key_checkpoint_path}/ppo_agent_without_tariff/ppo_agent_replication_{the_best_replication_without_tariff}.pt', f'checkpoints_{key_checkpoint_path}/best_checkpoint/ppo_agent_best_replication_without_tariff.pt')
    

    return "Simulation completed successfully"

async def run_the_simulation_effective_relocation_function(ppo_agent: ActiveLearningEffectiveRelocationPPOAgent, simulation_start: str, simulation_end: str, simulation_replications: int, key_checkpoint_path: str, is_training: bool = True, replication_restart: int = 0, highest_score: float = float('-inf'), tariff_rate: float = 0):
    # Load data and model
    data = load_data("../data/LONGBASE.TXT")
    history_data = load_data("../data/HISTDATA.TXT")
    frbus = Frbus("../models/model.xml") 

    # Simulation parameters
    residstart = "1975q1"
    residend = "2024q4"
    simstart = simulation_start  # Starting the simulation from the checkpoint
    simend = simulation_end 

    # Number of replications
    nrepl = simulation_replications
    # Run up to 5 extra replications, in case of failures
    nextra = 1
    experiences = []
    experiences_without_tariff = []
    # Policy settings
    data.loc[simstart:simend, "dfpdbt"] = 0
    data.loc[simstart:simend, "dfpsrp"] = 1
    def get_state(data, current_quarter):
        """Extract state variables from data from the previous quarter"""   
        previous_quarter = pd.Period(current_quarter) - 1
        return data.loc[previous_quarter].values

    def apply_actions(data, current_quarter, actions):
        """Apply PPO agent's actions to the data""" 
        
        previous_quarter = pd.Period(current_quarter) - 1 
        # Map actions to specific policy variables
        # Convert actions tensor to numpy if it's a torch tensor
        if torch.is_tensor(actions):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions 
        for var, action in zip(policy_vars, actions_np):
            if var == 'rff':
                continue
            if var == 'egfe': 
                data.loc[current_quarter, var] = data.loc[previous_quarter, var] + action * 1000  
            else: 
                if data.loc[previous_quarter, var] + action > 0.1 and data.loc[previous_quarter, var] + action < 0.4:
                    data.loc[current_quarter, var] = data.loc[previous_quarter, var] + action
                else:
                    data.loc[current_quarter, var] = data.loc[previous_quarter, var] 
        return data

    def apply_tariff_enhanced(data, current_quarter, tariff_rate):
        """
        Apply tariff effects to the U.S. economy with more nuanced logic.
        
        Args:
            data: DataFrame containing economic variables (indexed by quarter).
            current_quarter: Current simulation quarter (string or datetime).
            tariff_rate: Base rate of tariff (e.g., 0.05 for 5%).
        
        Returns:
            Modified DataFrame with updated import/export data and govt. revenue.
        """
        if tariff_rate == 0:
            return data
        # 1) Access import and export levels
        imports = data.loc[current_quarter, 'emn']  # Nominal imports
        exports = data.loc[current_quarter, 'exn']  # Nominal exports
        
        # 2) Tiered or adaptive elasticity approach
        # For demonstration, let's do a logistic-based elasticity
        # That shifts from -0.7 to -0.3 over the range of tariff_rate [0, 0.2] as an example
        import_elasticity = -0.5 + 0.4 / (1 + np.exp(-(tariff_rate - 0.1)*10))
        
        # Optionally mix with a base elasticity from some table or categories
        # import_elasticity could be a weighted average of multiple sub-elasticities.
        
        # 3) Compute tariff revenue
        tariff_revenue = imports * tariff_rate
        data.loc[current_quarter, 'gfrecn'] += tariff_revenue
        
        # 4) Adjust imports with partial or immediate effect
        # immediate effect = imports * (1 + tariff_rate * import_elasticity)
        # partial approach = alpha * immediate + (1-alpha)*previous
        alpha = 0.5  # partial adjustment factor
        desired_import = imports * (1 + tariff_rate * import_elasticity)
        new_imports = alpha * desired_import + (1 - alpha) * imports
        
        data.loc[current_quarter, 'emn'] = max(new_imports, 0)  # ensure not negative
        
        # 5) More nuanced retaliation model
        # maybe the partner imposes a tariff that is 70% of ours once we exceed 5%:
        min_tariff_threshold = 0.05
        if tariff_rate >= min_tariff_threshold:
            partner_tariff = 0.7 * tariff_rate
        else:
            partner_tariff = 0.0

        # Then the elasticity of demand for U.S. exports might differ from our import elasticity
        # We do a simple approach for demonstration:
        export_elasticity = -0.4
        # If partner_tariff > 0, reduce exports
        desired_export = exports * (1 + partner_tariff * export_elasticity)
        new_exports = alpha * desired_export + (1 - alpha) * exports
        data.loc[current_quarter, 'exn'] = max(new_exports, 0)
        
        return data

    # Initialize variables for tracking best replication
    score_replications = {}
    the_best_replication = 0 
    score_replications_without_tariff = {}
    the_best_replication_without_tariff = 0
    highest_score_without_tariff = float('-inf')
    # Modified simulation loop
    for rep in range(nrepl):
        start_time = time.time()
        sim_data = data.copy()        
        simstart_year = int(simstart.split('q')[0])
        if 1970 <= simstart_year <= 2023 and not is_training:
            sim_data_without_tariff = history_data.copy()
        else: 
            sim_data_without_tariff = data.copy()
        sim_data_without_rl = data.copy()
        sim_data_without_rl_tariff = data.copy()
        initial_simulation = True
        total_reward = torch.tensor(0.0).clone().detach()
        annual_inflation_target = 2.0  # 2% annual inflation target
        quarterly_inflation_target = ((1 + annual_inflation_target/100)**(1/4) - 1) * 100
        # This gives approximately 0.495% per quarter
        # Define target values
        targets = {
            'pcpi': quarterly_inflation_target,    # 2% inflation target
            'lur': 4.0,     # 4% unemployment target
            'hggdp': 4.0,    # 4% GDP growth target
            'rff': 0.5,  # Limit on interest rate changes
            'gfdbtn': 80,  # Debt-to-GDP target
        } 
        # Quarterly policy decisions 
        break_the_loop = False
        reward_list = {}
        logger.info(f"Starting training simulation with effective relocation in iteration {rep}")
        while True: 
            if initial_simulation:
                # Policy settings
                # Standard configuration, use surplus ratio targeting 
                sim_data_without_rl.loc[simstart:simend, "dfpdbt"] = 0
                sim_data_without_rl.loc[simstart:simend, "dfpsrp"] = 1
                sim_data_without_rl_tariff = sim_data_without_rl.copy()

                sim_data = frbus.init_trac(residstart, simend, sim_data)
                sim_data_without_tariff = frbus.init_trac(residstart, simend, sim_data_without_tariff)
                sim_data_without_rl = frbus.init_trac(residstart, simend, sim_data_without_rl)
                sim_data_without_rl_tariff = frbus.init_trac(residstart, simend, sim_data_without_rl_tariff)

                # 100 bp monetary policy shock and solve
                sim_data_without_rl.loc[simstart, "rffintay_aerr"] += 1 
                sim_data_without_rl_tariff.loc[simstart, "rffintay_aerr"] += 1 

            if ppo_agent.current_quarter in ppo_agent.economic_states and not initial_simulation:
                sim_data = ppo_agent.economic_states[ppo_agent.current_quarter]['solutions'] 
                
            for quarter in pd.date_range(start=ppo_agent.current_quarter, end=simend, freq='Q'):
                q = (quarter.month - 1) // 3 + 1
                quarter_str = f"{quarter.year}q{q}".lower()  
                previous_quarter = pd.Period(quarter_str) - 1
                if quarter_str.lower() == "2022q4".lower() and is_training:
                    break_the_loop = True
                    logger.info(f"Reached end of simulation - breaking the loop")
                    

                # Apply actions to the data
                if tariff_rate > 0:
                    sim_data = apply_tariff_enhanced(sim_data, quarter_str, tariff_rate)
                    sim_data_without_rl_tariff = apply_tariff_enhanced(sim_data_without_rl_tariff, quarter_str, tariff_rate)

                # Get current state
                state = get_state(sim_data, quarter_str)
                # Standard PPO action selection
                actions, log_probs, state_value = ppo_agent.forward_with_effective_relocation(state) 
                # Apply PPO actions to the data
                sim_data = apply_actions(sim_data, quarter_str, actions)
                # Run one quarter of simulation
                try: 
                    sim_data = frbus.solve(quarter_str, quarter_str, sim_data)
                    if initial_simulation: 
                        sim_data_without_rl_tariff = frbus.solve(quarter_str, quarter_str, sim_data_without_rl_tariff)
                        sim_data_without_tariff = frbus.solve(quarter_str, quarter_str, sim_data_without_tariff)
                        sim_data_without_rl = frbus.solve(simstart, simend, sim_data_without_rl) 
                    initial_simulation = False  
                    # Send metrics update to API
                    if not is_training:
                        await broadcast_metrics_update(
                            solution=sim_data,
                            solution_without_tariff=sim_data_without_tariff,
                            solution_without_rl=sim_data_without_rl,
                            solution_without_rl_tariff=sim_data_without_rl_tariff,
                            quarter_str=quarter_str.upper(),
                            targets=targets
                        )
                        
                    
                    # Store experience for PPO update
                    if is_training: # Calculate reward based on economic outcomes 
                        reward = calculate_reward_policy_v1(sim_data, quarter_str, simend)
                        reward = torch.as_tensor(reward).clone().detach()    
                        # Standard experience storing
                        experience = {
                            'quarter': quarter_str,
                            'state': state,
                            'actions': torch.as_tensor(actions).float(),
                            'log_probs': log_probs,
                            'reward': torch.as_tensor(reward).float(),
                            'value': state_value,
                            'done': quarter_str.lower() == "2022q4".lower()
                        } 
                        ppo_agent.save_state(state, quarter_str, previous_quarter, sim_data)
                        ppo_agent.visited_quarters.append(quarter_str)
                        experiences.append(experience)  
                        reward_list[quarter_str] = total_reward.clone().detach()
                        total_reward += reward
                        logger.info(f"Visited quarters: {quarter_str} with total reward {total_reward} reward {reward} and action {actions}") 
                        should_relocate, target_quarter =  ppo_agent.should_relocate(previous_quarter, quarter_str)
                        if should_relocate and not(break_the_loop):
                            ppo_agent.relocate(target_quarter)
                            total_reward = reward_list[target_quarter] if target_quarter in reward_list else 0
                            logger.info(f"Relocated to {target_quarter} with total reward {total_reward}")
                            logger.info(f"Reward list: {reward_list}")
                            break 

                except Exception as e:
                    previous_quarter = pd.Period(quarter_str) - 1
                    logger.error(f"Actions of Decision Maker: {actions}") 
                    logger.error(f"Current quarter: {quarter_str}, Previous quarter: {previous_quarter}")
                    for var, action in zip(policy_vars, actions):
                        logger.error(f"Variable: {var}, Action: {action}, Previous Data: {sim_data.loc[previous_quarter, var]}, Current Data: {sim_data.loc[quarter_str, var]}")
                    logger.error(f"Simulation stochsim failed for quarter {quarter_str}: {e}")
                    raise 

            if break_the_loop and is_training:
                # Standard PPO update
                logger.info(f"Updating PPO for replication effective relocation - cleaning up state")
                ppo_agent.update_ppo_effective_relocation(experiences) 
                # Standard PPO update
                experiences = []  # Clear experiences after update
                reward_list = {}
                ppo_agent.clean_up_state()
                break
            
            # Break the loop if not training after running the simulation
            if not is_training:
                break
        if is_training: 
            end_time = time.time()                    
            if total_reward >= highest_score:
                highest_score = total_reward
                the_best_replication = rep + replication_restart
            if rep + replication_restart == the_best_replication:
                checkpoint_path = f'checkpoints_{key_checkpoint_path}/ppo_agent/ppo_agent_replication_{rep + replication_restart}.pt'
                os.makedirs(f'checkpoints_{key_checkpoint_path}/ppo_agent', exist_ok=True)
                checkpoint = {
                    'actor_state_dict': ppo_agent.actor.state_dict(),
                    'critic_state_dict': ppo_agent.critic.state_dict(),
                    'replication': rep + replication_restart,
                    'action_bounds': ppo_agent.action_bounds
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved the bestcheckpoint at replication {rep + replication_restart}")
                
            logger.info(f"Total reward for replication effective relocation {rep + replication_restart}: {total_reward}")
            logger.info(f"Time taken for replication effective relocation {rep + replication_restart}: {end_time - start_time} seconds")
            logger.info(f"Highest reward for replication effective relocation {the_best_replication} with tariff: {highest_score}")
            logger.info(f"ETA taken for remaining replications effective relocation {nrepl - rep - replication_restart}: {(end_time - start_time) * (nrepl - rep - replication_restart)} seconds")

    if is_training:
        logger.info(f"The best replication effective relocation is {the_best_replication} with score {highest_score}")
        # Save the score replications to a text file
        with open(f'checkpoints_{key_checkpoint_path}/score_replications_effective_relocation.txt', 'w') as f:
            f.write(str(score_replications)) 
        with open(f'checkpoints_{key_checkpoint_path}/the_best_replication_effective_relocation.txt', 'w') as f:
            f.write(str(the_best_replication)) 
        # Copy the best checkpoint to the best_checkpoint folder
        os.makedirs(f'checkpoints_{key_checkpoint_path}/best_checkpoint_effective_relocation', exist_ok=True)
        shutil.copy(f'checkpoints_{key_checkpoint_path}/ppo_agent/ppo_agent_replication_{the_best_replication}.pt', f'checkpoints_{key_checkpoint_path}/best_checkpoint_effective_relocation/ppo_agent_best_replication_effective_relocation.pt')

    return "Simulation effective relocation completed successfully"


def load_checkpoint(path, ppo_agent):
    try:
        checkpoint = torch.load(path)
        ppo_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return ppo_agent
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}") 
        return ppo_agent

# Add the main execution block
async def main_training(active_learning=False):
    # Your existing setup code - example values shown below
    key_checkpoint_path = "trump_historical_ppo" 
    
    # Keep the standard agent for the tariff case
    if not active_learning:
        ppo_agent = PPOAgent(state_dim=934, action_dim=len(policy_vars), hidden_dim=4096, seed=69)
    else:
        # Create an active learning PPO agent instead of standard PPO
        ppo_agent = ActiveLearningPPOAgent(
            state_dim=934, 
            action_dim=len(policy_vars), 
            hidden_dim=4096,
            uncertainty_model_dim=512  # Size of the uncertainty prediction model
        )
    
    # Keep the standard agent for the without_tariff case
    if not active_learning:
        ppo_agent_without_tariff = PPOAgent(state_dim=934, action_dim=len(policy_vars), hidden_dim=4096, seed=69)
    else:
        ppo_agent_without_tariff = ActiveLearningPPOAgent(
            state_dim=934, 
            action_dim=len(policy_vars), 
            hidden_dim=4096,
            uncertainty_model_dim=512  # Size of the uncertainty prediction model
        )
    simulation_start = "1970q1"
    simulation_end = "2023q1"
    simulation_replications = 25
    
    result = await run_the_simulation_function(
        ppo_agent, 
        ppo_agent_without_tariff,
        simulation_start, 
        simulation_end, 
        simulation_replications, 
        key_checkpoint_path,
        is_training=True
    )
    logger.info(result)

# Add the main execution block
async def main_training_effective_relocation():
    # Your existing setup code - example values shown below
    key_checkpoint_path = "trump_historical_active_learning_effective_relocation" 
    
    # Keep the standard agent for the tariff case
    simulation_start = "1970q1"
    simulation_end = "2023q1"
    simulation_replications = 25
    ppo_agent = ActiveLearningEffectiveRelocationPPOAgent(state_dim=934, action_dim=len(policy_vars), current_quarter=simulation_start, hidden_dim=4096, seed=69)
    logger.info("Starting simulation training with effective relocation")
    result = await run_the_simulation_effective_relocation_function(
        ppo_agent, 
        simulation_start, 
        simulation_end, 
        simulation_replications, 
        key_checkpoint_path,
        is_training=True
    )
    logger.info(result)

# Add the main execution block - trump_v16 - replication 211
async def main_training_resume():
    # Your existing setup code - example values shown below
    key_checkpoint_path = "trump_historical_active_2" 
    checkpoint_path = f"checkpoints_{key_checkpoint_path}/ppo_agent/ppo_agent_replication_34.pt"
    checkpoint_path_without_tariff = f"checkpoints_{key_checkpoint_path}/ppo_agent_without_tariff/ppo_agent_replication_0.pt"
    ppo_agent = load_checkpoint(checkpoint_path, PPOAgent(state_dim=934, action_dim=len(policy_vars), hidden_dim=4096))
    ppo_agent_without_tariff = load_checkpoint(checkpoint_path_without_tariff, PPOAgent(state_dim=934, action_dim=len(policy_vars), hidden_dim=4096, seed=69)) 

    simulation_start = "1970q1"
    simulation_end = "2023q1"
    simulation_replications = 25
    replication_restart = 211
    
    result = await run_the_simulation_function(
        ppo_agent, 
        ppo_agent_without_tariff,
        simulation_start, 
        simulation_end, 
        simulation_replications, 
        key_checkpoint_path,
        is_training=True,
        replication_restart=replication_restart,
        highest_score=278.0218200683594
    )
    logger.info(result)

# Add the main execution block
async def main_simulation(
    simstart: str,
    simend: str,
    tariff_rate: float,
    active_learning: bool = False
):
    """
    Main simulation function that orchestrates the economic simulation process.
    
    Parameters:
    - simstart: Start date for simulation in format 'YYYYqN'
    - simend: End date for simulation in format 'YYYYqN'
    - tariff_rate: Tariff rate as a decimal (e.g., 0.10 for 10%)
    """
    key_checkpoint_path = ""
    if not active_learning:
        key_checkpoint_path = "trump_historical_ppo"
    else:
        key_checkpoint_path = "trump_historical_active"
    checkpoint_path = f"checkpoints_{key_checkpoint_path}/best_checkpoint/ppo_agent_best_replication.pt"
    checkpoint_path_without_tariff = f"checkpoints_{key_checkpoint_path}/best_checkpoint/ppo_agent_best_replication_without_tariff.pt"
    # Keep the standard agent for the tariff case
    if not active_learning:
        ppo_agent = PPOAgent(state_dim=934, action_dim=len(policy_vars), hidden_dim=4096, seed=69)
    else:
        ppo_agent = ActiveLearningPPOAgent(
            state_dim=934, 
            action_dim=len(policy_vars), 
            hidden_dim=4096,
            uncertainty_model_dim=512  # Size of the uncertainty prediction model
        )
    
    # Keep the standard agent for the without_tariff case
    if not active_learning:
        ppo_agent_without_tariff = PPOAgent(state_dim=934, action_dim=len(policy_vars), hidden_dim=4096, seed=69)
    else:
        ppo_agent_without_tariff = ActiveLearningPPOAgent(
            state_dim=934, 
            action_dim=len(policy_vars), 
            hidden_dim=4096,
            uncertainty_model_dim=512  # Size of the uncertainty prediction model
        )
    ppo_agent = load_checkpoint(checkpoint_path, ppo_agent)
    ppo_agent_without_tariff = load_checkpoint(checkpoint_path_without_tariff, ppo_agent_without_tariff)
    simulation_replications = 1

    result = await run_the_simulation_function(
        ppo_agent, 
        ppo_agent_without_tariff,
        simstart, 
        simend, 
        simulation_replications,
        key_checkpoint_path,
        is_training=False,
        tariff_rate=tariff_rate,
        replication_restart=0,
        highest_score=0.0
    )
    logger.info(result)

# Add the main execution block
async def main_simulation_effective_relocation(
    simstart: str,
    simend: str,
    tariff_rate: float,
    active_learning: bool = True
):
    """
    Main simulation function that orchestrates the economic simulation process.
    
    Parameters:
    - simstart: Start date for simulation in format 'YYYYqN'
    - simend: End date for simulation in format 'YYYYqN'
    - tariff_rate: Tariff rate as a decimal (e.g., 0.10 for 10%)
    """
    key_checkpoint_path = "trump_historical_active_learning_effective_relocation"
    checkpoint_path = f"checkpoints_{key_checkpoint_path}/best_checkpoint_effective_relocation/ppo_agent_best_replication_effective_relocation.pt"

    ppo_agent = ActiveLearningEffectiveRelocationPPOAgent(state_dim=934, action_dim=len(policy_vars), current_quarter=simstart, hidden_dim=4096, seed=69)
    logger.info("Starting the simulation with effective relocation")
    ppo_agent = load_checkpoint(checkpoint_path, ppo_agent)
    simulation_replications = 1

    result = await run_the_simulation_effective_relocation_function(
        ppo_agent, 
        simstart, 
        simend, 
        simulation_replications,
        key_checkpoint_path,
        is_training=False,
        tariff_rate=tariff_rate,
        replication_restart=0,
        highest_score=0.0
    )
    logger.info(result)
@app.get("/run_simulation_training")
def run_simulation_training():
    asyncio.run(main_training())
    return {"message": "Simulation training completed successfully"}

@app.get("/run_simulation_training_effective_relocation")
def run_simulation_training_effective_relocation():
    asyncio.run(main_training_effective_relocation())
    return {"message": "Simulation training effective relocation completed successfully"}

@app.get("/run_simulation")
async def run_simulation(
    simulation_type: str = Query("hypothetical", description="Type of simulation: 'historical' or 'hypothetical'"),
    start_year: int = Query(2024, description="Start year for simulation"),
    end_year: int = Query(2030, description="End year for simulation"),
    tariff_rate: float = Query(10.0, description="Tariff rate percentage (for hypothetical simulation)"),
    active_learning: bool = Query(False, description="Active learning flag")
):
    """
    Run economic simulation with specified parameters.
    
    Parameters:
    - simulation_type: 'historical' for 1970-2024 data, 'hypothetical' for future projections
    - start_year: Starting year for simulation
    - end_year: Ending year for simulation
    - tariff_rate: Tariff rate percentage (0-50) for hypothetical simulation
    """
    # Validate parameters
    if simulation_type == "historical" and (start_year < 1970 or end_year > 2024):
        return {"error": "Historical simulation must be between 1970-2024"}
    
    if simulation_type == "hypothetical" and (start_year < 2024 or end_year > 2075):
        return {"error": "Hypothetical simulation must be between 2024-2075"}
    
    if end_year < start_year:
        return {"error": "End year must be after start year"}
    
    # Convert dates to the format expected by main_simulation
    simstart = f"{start_year}q1"
    simend = f"{end_year}q4" if end_year != 2024 else "2024q3"
    
    # Call the main simulation function with the parameters
    asyncio.create_task(main_simulation(
        simstart=simstart,
        simend=simend,
        tariff_rate=tariff_rate/100.0,
        active_learning=active_learning
    ))
    
    return {"message": f"Simulation started: {simulation_type} from {start_year} to {end_year} with tariff rate {tariff_rate}%"}

@app.get("/run_simulation_effective_relocation")
async def run_simulation_effective_relocation(
    simulation_type: str = Query("hypothetical", description="Type of simulation: 'historical' or 'hypothetical'"),
    start_year: int = Query(2024, description="Start year for simulation"),
    end_year: int = Query(2030, description="End year for simulation"),
    tariff_rate: float = Query(10.0, description="Tariff rate percentage (for hypothetical simulation)")
):
    """
    Run economic simulation with specified parameters.
    
    Parameters:
    - simulation_type: 'historical' for 1970-2024 data, 'hypothetical' for future projections
    - start_year: Starting year for simulation
    - end_year: Ending year for simulation
    - tariff_rate: Tariff rate percentage (0-50) for hypothetical simulation
    """
    # Validate parameters
    if simulation_type == "historical" and (start_year < 1970 or end_year > 2024):
        return {"error": "Historical simulation must be between 1970-2024"}
    
    if simulation_type == "hypothetical" and (start_year < 2024 or end_year > 2075):
        return {"error": "Hypothetical simulation must be between 2024-2075"}
    
    if end_year < start_year:
        return {"error": "End year must be after start year"}
    
    # Convert dates to the format expected by main_simulation
    simstart = f"{start_year}q1"
    simend = f"{end_year}q4" if end_year != 2024 else "2024q3"
    
    # Call the main simulation function with the parameters
    asyncio.create_task(main_simulation_effective_relocation(
        simstart=simstart,
        simend=simend,
        tariff_rate=tariff_rate/100.0
    ))
    
    return {"message": f"Simulation started: {simulation_type} from {start_year} to {end_year} with tariff rate {tariff_rate}%"}

@app.get("/run_simulation_resume")
def run_simulation_resume():
    asyncio.run(main_training_resume())
    return {"message": "Simulation resumed successfully"}



@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send latest metrics immediately upon connection
        if latest_metrics:
            logger.info(f"Sending latest metrics: {latest_metrics}")
            await websocket.send_json(latest_metrics)
        
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/simulation/update")
async def update_simulation(data: SimulationComparison):
    """
    Receive new simulation data and broadcast to all connected clients
    """
    try:
        # Store latest metrics
        latest_metrics[data.rl_decision.quarter] = data.dict()
        # Broadcast to all connected clients
        await manager.broadcast(data.dict())
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "clients": len(manager.active_connections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/latest")
async def get_latest_metrics():
    """
    Get the most recent metrics
    """
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics available")
    return latest_metrics

@app.get("/simulation/quarters/{quarter}")
async def get_quarter_metrics(quarter: str):
    """
    Get metrics for a specific quarter
    """
    if quarter not in latest_metrics:
        raise HTTPException(status_code=404, detail=f"No data for quarter {quarter}")
    return latest_metrics[quarter]