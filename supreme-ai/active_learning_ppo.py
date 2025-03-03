import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ppo_agent import PPOAgent

class UncertaintyModel(nn.Module):
    """
    Model that predicts uncertainty for active learning.
    Higher uncertainty = more informative for learning.
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(UncertaintyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Uncertainty between 0 and 1
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class ActiveLearningPPOAgent(PPOAgent):
    def __init__(self, state_dim, action_dim, hidden_dim=256, uncertainty_model_dim=256, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=10, seed=42):
        super(ActiveLearningPPOAgent, self).__init__(
            state_dim, action_dim, hidden_dim, lr, gamma, eps_clip, K_epochs, seed
        )
        
        # Uncertainty model for active learning
        self.uncertainty_model = UncertaintyModel(state_dim, action_dim, uncertainty_model_dim)
        self.uncertainty_optimizer = optim.Adam(self.uncertainty_model.parameters(), lr=lr)
        
        # Exploration parameters
        self.exploration_factor = 1.0  # Start with high exploration
        self.min_exploration = 0.1    # Minimum exploration factor
        self.exploration_decay = 0.995  # Decay rate for exploration
        
        # Active learning parameters
        self.information_gain_weight = 1.0  # Weight for information gain in action selection
        
    def initialize_uncertainty_model(self):
        """Initialize the uncertainty model with random experiences"""
        # This could be improved with actual data, but for now we'll just initialize the model
        pass
        
    def decay_exploration(self):
        """Decay the exploration factor over time"""
        self.exploration_factor = max(self.min_exploration, 
                                      self.exploration_factor * self.exploration_decay)
        
    def forward_with_uncertainty(self, state):
        """
        Forward pass that considers uncertainty for active learning.
        Returns actions, log_probs, value, and uncertainty.
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state = state.unsqueeze(0).to(self.device)
            
        # Get the action distribution
        action_mean, action_std = self.actor(state)
        action_var = action_std.pow(2)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sample multiple candidate actions
        num_candidates = 5
        candidate_actions = []
        
        for _ in range(num_candidates):
            action = action_dist.sample()
            action = torch.clamp(action, -1, 1)  # Clamp to action space
            candidate_actions.append(action)
            
        # Evaluate uncertainty for each candidate action
        uncertainties = []
        for action in candidate_actions:
            uncertainty = self.uncertainty_model(state, action)
            uncertainties.append(uncertainty.item())
            
        # Select action based on uncertainty (information gain)
        # With probability exploration_factor, choose the most uncertain action
        # Otherwise, choose the action with highest expected reward
        if np.random.rand() < self.exploration_factor:
            # Choose action with highest uncertainty
            selected_idx = np.argmax(uncertainties)
        else:
            # Choose action based on policy (default PPO behavior)
            selected_idx = 0  # The first sampled action
            
        action = candidate_actions[selected_idx]
        uncertainty = uncertainties[selected_idx]
        
        # Get value and log_prob for the selected action
        value = self.critic(state)
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        
        return action.detach().cpu().numpy()[0], log_prob.detach(), value.detach(), uncertainty
    
    def update_uncertainty_model(self, experiences):
        """Update the uncertainty model based on prediction errors"""
        # Extract data from experiences
        states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
        actions = torch.FloatTensor([exp['actions'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        
        # Get value predictions
        values = self.critic(states).squeeze()
        
        # Calculate TD errors (reward - value) as a proxy for prediction error
        td_errors = torch.abs(rewards - values).detach()
        
        # Normalize TD errors to 0-1 range for target uncertainty
        target_uncertainty = td_errors / (td_errors.max() + 1e-8)
        
        # Update uncertainty model to predict high uncertainty for high TD error states/actions
        for _ in range(5):  # Multiple optimization steps
            predicted_uncertainty = self.uncertainty_model(states, actions).squeeze()
            loss = nn.MSELoss()(predicted_uncertainty, target_uncertainty)
            
            self.uncertainty_optimizer.zero_grad()
            loss.backward()
            self.uncertainty_optimizer.step()
            
        # Decay exploration factor
        self.decay_exploration()
        
    def update_ppo_active_learning(self, experiences):
        """
        PPO update that weights experiences by their uncertainty/information gain.
        Higher uncertainty experiences get more weight in the update.
        """
        # Extract data from experiences
        states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
        actions = torch.FloatTensor([exp['actions'] for exp in experiences]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_probs'] for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in experiences]).to(self.device)
        values = torch.FloatTensor([exp['value'] for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in experiences]).to(self.device)
        uncertainties = torch.FloatTensor([exp['uncertainty'] for exp in experiences]).to(self.device)
        
        # Normalize uncertainties for weighting
        weights = uncertainties / (uncertainties.mean() + 1e-8)
        
        # Calculate returns using weighted rewards
        returns = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_reward = 0
            discounted_reward = rewards[i] + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # Update for K epochs
        for _ in range(self.K_epochs):
            # Get current policy and value
            action_mean, action_std = self.actor(states)
            action_var = action_std.pow(2)
            dist = torch.distributions.Normal(action_mean, action_std)
            curr_log_probs = dist.log_prob(actions).sum(1)
            entropy = dist.entropy().mean()
            curr_values = self.critic(states).squeeze()
            
            # Calculate ratios
            ratios = torch.exp(curr_log_probs - old_log_probs)
            
            # Calculate surrogate losses with importance weights
            advantages = returns - values.detach()
            surr1 = ratios * advantages * weights  # Apply uncertainty weights
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages * weights
            
            # Loss calculation with weighted components
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((returns - curr_values) * weights).pow(2).mean()
            total_loss = actor_loss + critic_loss - 0.01 * entropy
            
            # Update actor-critic
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        return 