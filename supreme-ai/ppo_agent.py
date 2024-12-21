import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.0003, gamma=0.99, eps_clip=0.2):
        super(PPOAgent, self).__init__()
        
        # Larger network for complex economic relationships
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output mean and std for each action
            nn.Linear(hidden_dim, action_dim * 2)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.gamma = gamma  # Quarterly discount factor
        self.eps_clip = eps_clip
        self.action_dim = action_dim
        
        # Action bounds for different policy tools
        self.action_bounds = {
            'rff': (0.125, 8.0),    # Quarterly interest rate change limits (percentage points)
            'gtrt': (0.1, 0.3),    # Trend ratio of transfer payments to GDP (percentage)
            'egfen': (0.1, 0.5),    # Trend level of federal government expenditures. (billions of dollars)
            'trp': (0.1, 0.4),    # Personal tax revenues rates (percentage)
            'trci': (0.1, 0.4),    # Corporate tax revenues rates (percentage)
        }
        
    def forward(self, state):
        state = torch.from_numpy(state).float()
        raw_output = self.actor(state)
        # Scale the means to be within action bounds
        means = torch.tanh(raw_output[:self.action_dim]) # Will give values between -1 and 1
        # Then scale to your desired range
        means = means * 0.5 # Or whatever scaling factor matches your bounds
        log_stds = raw_output[self.action_dim:]
        
        # Bound the standard deviations
        stds = log_stds.exp().clamp(min=1e-3, max=1)
        
        # Create normal distributions for each action
        dists = [Normal(mean, std) for mean, std in zip(means, stds)]
        
        # Sample actions and get their log probs
        actions = []
        log_probs = []
        
        for i, dist in enumerate(dists):
            action = dist.sample()
            # Apply action bounds
            bound_low, bound_high = self.action_bounds[list(self.action_bounds.keys())[i]]
            action = torch.clamp(action, bound_low, bound_high)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
        
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        
        state_value = self.critic(state)
        
        return actions.detach().numpy(), log_probs, state_value
    
    def evaluate(self, state, action):
        # Get mean and std from actor network
        action_mean_std = self.actor(state)
        action_mean, action_std = torch.chunk(action_mean_std, 2, dim=-1)
        # Use softplus to ensure positive std and add small value for stability
        action_std = torch.nn.functional.softplus(action_std) + 1e-3
        
        # Create normal distribution
        dist = Normal(action_mean, action_std)
        
        # Get log probability of the taken action
        action_logprobs = dist.log_prob(action)
        
        # Get entropy for exploration
        dist_entropy = dist.entropy().mean()
        
        # Get state value
        state_value = self.critic(state)
        
        return action_logprobs, state_value, dist_entropy
    
    def update(self, states, actions, logprobs, rewards, values, dones):
        """
        Update policy and value function using PPO algorithm.
        
        Args:
            states: Tensor of states (already stacked)
            actions: Tensor of actions (already stacked)
            logprobs: Tensor of log probabilities (already stacked)
            rewards: Tensor of rewards (already stacked)
            values: Tensor of state values (already stacked)
            dones: Tensor of done flags (already stacked)
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
        for _ in range(10):
            # Get current policy and value predictions
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
            entropy_loss = -0.01 * dist_entropy  # Encourage exploration
            
            # Combined loss
            loss = policy_loss + value_loss + entropy_loss
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
