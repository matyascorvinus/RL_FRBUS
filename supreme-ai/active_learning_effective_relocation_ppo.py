import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ppo_agent import PPOAgent
from torch.distributions import MultivariateNormal
import copy
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class ActiveLearningEffectiveRelocationPPOAgent(PPOAgent):
    def __init__(self, state_dim, action_dim, current_quarter="1970q1", hidden_dim=256, uncertainty_model_dim=256, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, seed=42, phi=100):
        super(ActiveLearningEffectiveRelocationPPOAgent, self).__init__(
            state_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs, seed
        ) 
        
        # Exploration parameters
        self.exploration_factor = 1.0  # Start with high exploration
        self.min_exploration = 0.1    # Minimum exploration factor
        self.exploration_decay = 0.995  # Decay rate for exploration

        # Active learning parameters
        self.relocation_cost = 1
        self.economic_states = {}  # Dictionary to store economic states 
        self.state_values = {}     # Store value estimates for states
        self.visited_quarters = []  # Track quarters visited in current episode
        self.original_quarter = current_quarter
        self.current_quarter = current_quarter
        self.phi = phi
        self.max_delta_value = 1.0  # Max observed value change (will be updated during training)
        self.epsilon = 0.1  # Starting probability for "In Trouble"
        self.tau = 0.2  # Parameter for "In Trouble" method
        self.pi = self.epsilon     # Current relocation probability
        # Method selection
        self.method = "InTrouble"  # Can be "Bored" or "InTrouble"
    def clean_up_state(self):
        self.relocation_cost = 1
        self.economic_states = {}  # Dictionary to store economic states 
        self.state_values = {}     # Store value estimates for states
        self.visited_quarters = []  # Track quarters visited in current episode
        self.current_quarter = self.original_quarter

    def save_state(self, state_data, current_quarter, previous_quarter, solutions):
        """Save complete economic snapshot for potential relocation"""
        self.economic_states[current_quarter] = {}  # Deep copy all economic indicators
        self.economic_states[current_quarter]['state_data'] = state_data.copy()
        self.economic_states[current_quarter]['solutions'] = solutions.copy()
        # Calculate and store value estimate
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_data).float()
            _, _, value = self.forward(state_tensor.numpy())
            self.state_values[current_quarter] = value.item()

    def relocate(self, target_quarter):
        """Reset the FRB/US model to a previous economic state"""
        if target_quarter in self.economic_states:
            # Call FRB/US model API to reset economic conditions
            self.current_quarter = target_quarter 
    
    def calculate_uncertainty(self, state):
        """Calculate uncertainty for a state using policy distribution"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float()
            action_mean_std = self.actor(state_tensor)
            means, log_stds = torch.chunk(action_mean_std, 2, dim=-1)
            stds = torch.exp(torch.clamp(log_stds, min=-5.0, max=2.0))
            
            # Higher standard deviation means higher uncertainty
            uncertainty = stds.mean().item()
            
            # You could also add value estimate volatility if you had ensemble critics
            return uncertainty

    def calculate_relocation_probability_bored(self, current_value, previous_value):
        """Calculate relocation probability using the 'Bored' method"""
        delta = abs(current_value - previous_value)
        
        # Update max observed delta
        if delta > self.max_delta_value:
            self.max_delta_value = delta
        
        # Calculate probability (highest when delta is 0)
        if delta == 0:
            pi = 0.5
        else:
            pi = 0.5 * np.exp(-self.phi * delta / self.max_delta_value)
        
        # Adjust for relocation cost
        return pi / (1 + self.relocation_cost)

    def increase_relocation_cost(self):
        self.relocation_cost = self.relocation_cost + 2

    def calculate_relocation_probability_in_trouble(self, current_value, previous_value):
        """Calculate relocation probability using the 'In Trouble' method"""
        # If value is decreasing, might be in trouble
        if current_value < previous_value:
            self.pi = self.pi + self.tau * (previous_value - current_value)
            self.pi = min(0.9, self.pi)  # Cap at 0.9 as in the paper
        else:
            # Reset probability if not in trouble
            self.pi = self.epsilon
        
        # Adjust for relocation cost
        return self.pi / (1 + self.relocation_cost)
    

    def choose_relocation_quarter(self):
        """Choose which quarter to relocate to based on uncertainty"""
        if not self.visited_quarters:
            return None
        
        # Calculate uncertainty for each visited quarter
        uncertainties = {q: self.calculate_uncertainty(self.economic_states[q]['state_data']) 
                         for q in self.visited_quarters}
        
        # Choose quarter with highest uncertainty
        return max(uncertainties, key=uncertainties.get)
    

    def should_relocate(self, previous_quarter, current_quarter):
        """Determine if agent should relocate based on current method"""
        if current_quarter == 0 or len(self.visited_quarters) <= 1:
            return False, None
            
        current_value = self.state_values[current_quarter]
        previous_quarter = self.visited_quarters[-2]
        previous_value = self.state_values[previous_quarter]
        
        # Calculate relocation probability based on selected method
        if self.method == "Bored":
            reloc_prob = self.calculate_relocation_probability_bored(current_value, previous_value)
        else:  # "InTrouble"
            reloc_prob = self.calculate_relocation_probability_in_trouble(current_value, previous_value)
        
        # Decide whether to relocate
        if np.random.random() < reloc_prob:
            self.increase_relocation_cost()
            target_quarter = self.choose_relocation_quarter()
            logger.info(f"Relocating to {target_quarter} from {current_quarter} with relocation cost {self.relocation_cost}")
            return True, target_quarter
            
        return False, None

    def forward_with_effective_relocation(self, state):
        """
        Forward pass that using effective relocation for active learning.
        Returns actions, log_probs, value.
        """
        # Convert state if needed and keep original for parent call 
        action, log_prob, value = super().forward(state)
        return action, log_prob, value
    
    def update_ppo_effective_relocation(self, experiences):
        """
        Update PPO agent with collected experiences.
        
        Args:
            experiences (list): List of dictionaries containing experience data
        """
        trajectories = []
        trajectory = []
        index = 0
        for experience in experiences:
            if index != 0 and pd.Period(experience['quarter']) < pd.Period(experiences[index - 1]['quarter']):
                logger.info(f"Current trajectory with current quarter: {experience['quarter']} and previous quarter: {experiences[index - 1]['quarter']}")
                trajectories.append(trajectory)
                trajectory = []
            trajectory.append(experience)
            index += 1 
        logger.info(f"Trajectories: {len(trajectories)}")
        for trajectory in trajectories:
            super().update_ppo(trajectory) 

        