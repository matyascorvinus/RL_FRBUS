import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 
ACTION_BOUNDS = { 
    'egfe': (-0.5, 0.5),    # Trend level of federal government expenditures. (trillions of dollars)
    'trptx': (-0.1, 0.1),    # Personal tax revenues rates (percentage)
    'trcit': (-0.1, 0.1),    # Corporate tax revenues rates (percentage)
}        

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=10, seed=42):
        super(PPOAgent, self).__init__()
        torch.manual_seed(seed)
        self.K_epochs = K_epochs
        self.device = torch.device("cpu")

        # Larger network for complex economic relationships
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # Output mean and std for each action
            nn.Linear(hidden_dim, action_dim * 2)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.gamma = gamma  # Quarterly discount factor
        self.eps_clip = eps_clip
        self.action_dim = action_dim 

        # Action bounds for different policy tools (Addition - Subtraction)
        self.action_bounds = ACTION_BOUNDS
        
    def forward(self, state):
        state = torch.from_numpy(state).float()
        raw_output = self.actor(state)
        # Split into means and log_stds
        means, log_stds = torch.chunk(raw_output, 2, dim=-1)
        # Optionally clamp the log_stds to avoid extreme values
        log_stds = torch.clamp(log_stds, min=-5.0, max=2.0)
        stds = log_stds.exp()

        # Create multivariate distribution
        dist = Normal(means, stds)

        # Pass through critic
        value = self.critic(state)

        # 1) Sample raw action from the Normal distribution
        raw_action = dist.rsample()  # shape: [action_dim] (assuming a single state)

        # 2) Tanh transform to keep values in [-1, 1]
        action_tanh = torch.tanh(raw_action)  # shape: [action_dim]

        # 3) Adjust log_prob for Tanh
        #    log_prob of raw_action minus the log of the Jacobian determinant
        #    = dist.log_prob(raw_action) - log(1 - tanh(raw_action)^2)
        # We sum across dimensions for the final log prob of the action vector.
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # shape: [1]

        # 4) Scale each dimension from [-1, 1] to [low, high]
        #    We'll rely on a consistent ordering of the keys in `self.action_bounds`.
        actions = []
        tool_keys = list(self.action_bounds.keys())  # e.g. ['rff','gtrt','egfe','trptx','trcit']

        # Make sure this ordering matches the order in which your network outputs each action dimension!
        for i, tool_name in enumerate(tool_keys):
            low, high = self.action_bounds[tool_name]
            # Rescale from [-1, 1] to [low, high]
            # scaled = low + (high - low)*( (action_tanh[i]+1)/2 )
            scaled_action = low + (high - low) * (action_tanh[i] + 1) / 2.0
            actions.append(scaled_action)

        # 5) Combine into a single tensor
        actions = torch.stack(actions)  # shape: [action_dim]
 
        return actions.detach().numpy(), log_prob, value
    
    def evaluate(self, state, action):
        # Get mean and std from actor network
        action_mean_std = self.actor(state)
        
        # Split into means and log_stds (same as forward)
        means, log_stds = torch.chunk(action_mean_std, 2, dim=-1)
        log_stds = torch.clamp(log_stds, min=-5.0, max=2.0)
        stds = log_stds.exp()
        
        # Create distribution
        dist = Normal(means, stds)
        
        # Convert provided actions back to raw space (inverse of tanh)
        # First rescale actions from [low, high] to [-1, 1]
        normalized_actions = []
        tool_keys = list(self.action_bounds.keys())
        
        for i, tool_name in enumerate(tool_keys):
            low, high = self.action_bounds[tool_name]
            # Inverse of scaling from forward method
            normalized = 2.0 * (action[:, i] - low) / (high - low) - 1.0
            normalized_actions.append(normalized)
        
        normalized_actions = torch.stack(normalized_actions, dim=1)
        
        # Inverse of tanh (atanh)
        raw_actions = torch.atanh(torch.clamp(normalized_actions, -0.999, 0.999))
        
        # Calculate log probabilities (same as forward)
        log_probs = dist.log_prob(raw_actions) - torch.log(1 - normalized_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        dist_entropy = dist.entropy()
        
        # Get state value from critic
        state_value = self.critic(state)
        
        return log_probs, state_value, dist_entropy
        
    def _update(self, states, actions, logprobs, rewards, values):
        """
        Update policy and value function using PPO algorithm.
        
        Args:
            states: Tensor of states (already stacked)
            actions: Tensor of actions (already stacked)
            logprobs: Tensor of log probabilities (already stacked)
            rewards: Tensor of rewards (already stacked)
            values: Tensor of state values (already stacked)
        """
        # Calculate advantages using GAE
        advantages = []
        gae = 0
        
        # Calculate GAE with proper discounting
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * gae  # 0.95 is GAE lambda
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages)
        
        # Normalize advantages for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Get current policy and value predictions
            logger.info(f"Evaluating policy and value predictions for {len(states)} states and actions {actions}")
            logprobs_new, state_values, dist_entropy = self.evaluate(states, actions)
            
            # Calculate KL divergence
            kl_div = (logprobs - logprobs_new).mean()
            if kl_div > 0.015:  # Early stopping threshold
                break
            
            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs_new - logprobs.detach())
            
            # Finding Surrogate Loss with clipped objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Calculate policy and value losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * torch.mean(torch.pow(state_values - rewards, 2))
            # Sum entropy across dimensions since it's per-dimension
            entropy_loss = -0.01 * dist_entropy.sum(dim=-1).mean()  # Changed this line
            
            # Combined loss
            loss = policy_loss + value_loss + entropy_loss
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
        
    def update_ppo(self, experiences):
        """
        Update PPO agent with collected experiences.
        
        Args:
            experiences (list): List of dictionaries containing experience data
        """
        # Convert experiences to torch tensors, handling numpy arrays
        states = torch.stack([torch.from_numpy(e['state']).float() for e in experiences])
        
        # For actions, we need to handle the case where it's already a tensor
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        
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
                
        
        actions = torch.stack(actions_list)
        log_probs = torch.stack(log_probs_list).detach()
        rewards = torch.stack(rewards_list)
        values = torch.stack(values_list)
        
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
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
                batch_rewards = rewards[batch_indices]
                batch_values = values[batch_indices]

                # Update the policy and value function
                self._update(
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_rewards,
                    batch_values
                )
        