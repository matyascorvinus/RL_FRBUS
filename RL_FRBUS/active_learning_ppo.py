import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ppo_agent import PPOAgent
from torch.distributions import MultivariateNormal
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(ActorCritic, self).__init__()
        
        # Actor network - produces mean of action distribution
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Action standard deviation (learnable parameter)
        self.action_std = nn.Parameter(torch.full((action_dim,), action_std_init))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        action_mean = self.actor(state)
        
        # Create normal distribution with learned std
        cov_mat = torch.diag(self.action_std.pow(2))
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # Sample action from distribution
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action, action_logprob
        
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        # Calculate action probabilities
        action_var = self.action_std.pow(2)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # Get log probabilities
        action_logprobs = dist.log_prob(action)
        
        # Get entropy
        dist_entropy = dist.entropy()
        
        # Get state value
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
    def update_action_std(self, action_std_decay_rate=0.995, min_action_std=0.1):
        """Decay action std for better exploitation over time"""
        self.action_std.data.mul_(action_std_decay_rate)
        self.action_std.data.clamp_(min=min_action_std)
        return self.action_std.mean().item()
    
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
        
    def train_step(self, states, actions, targets):
        # Forward pass
        predictions = self.forward(states, actions)
        
        # Calculate loss
        loss = nn.MSELoss()(predictions.squeeze(), targets)
        
        # Backward pass and optimization
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

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
        self.uncertainty_init_samples = 100  # Number of random samples for initialization
        self.uncertainty_init_epochs = 5     # Training epochs for initialization
        
        # FRB/US specific components
        self.frbus_model = None            # FRB/US model reference
        self.apply_actions_fn = None       # Function to apply actions to FRB/US
        self.get_state_fn = None           # Function to get state from FRB/US
        self.calculate_reward_fn = None    # Function to calculate reward
        
    def set_frbus_components(self, frbus_model, apply_actions_fn, get_state_fn, calculate_reward_fn):
        """Set the FRB/US model and related functions"""
        self.frbus_model = frbus_model
        self.apply_actions_fn = apply_actions_fn
        self.get_state_fn = get_state_fn
        self.calculate_reward_fn = calculate_reward_fn
        logger.info("FRB/US components set in ActiveLearningPPOAgent")
        
    def initialize_uncertainty_model(self, solutions, quarter_str, solution_without_rl, simend, tariff_rate=None):
        """Initialize the uncertainty model with random experiences from FRB/US model"""
        if self.frbus_model is None or self.apply_actions_fn is None or self.get_state_fn is None:
            logger.warning("Warning: FRB/US components not set - skipping uncertainty model initialization")
            return
            
        logger.info(f"Initializing uncertainty model with {self.uncertainty_init_samples} samples...")
        
        # Create storage for random samples
        states = []
        actions = []
        rewards = []
        
        # Make a copy of solutions to avoid modifying the original
        init_solutions = solutions.copy()
                    
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
        # Generate random samples
        for _ in range(self.uncertainty_init_samples):
            try:
                # Get current state
                state = self.get_state_fn(init_solutions, quarter_str)
                
                # Generate random action within bounds
                random_action = []
                tool_keys = list(self.action_bounds.keys())
                for tool_name in tool_keys:
                    low, high = self.action_bounds[tool_name]
                    random_val = np.random.uniform(low, high)
                    random_action.append(random_val)
                random_action = np.array(random_action)
                
                # Apply actions to the model
                temp_solutions = self.apply_actions_fn(init_solutions.copy(), quarter_str, random_action)
                
                # Run one quarter of simulation
                temp_solutions = self.frbus_model.solve(quarter_str, quarter_str, temp_solutions)
                
                # Calculate reward
                if tariff_rate is not None:
                    # Apply tariff if needed
            
                    temp_solutions = apply_tariff_enhanced(temp_solutions, quarter_str, tariff_rate)
                
                reward = self.calculate_reward_fn(temp_solutions, quarter_str, simend)
                
                # Store experience
                states.append(state)
                actions.append(random_action)
                rewards.append(reward)
                
            except Exception as e:
                logger.error(f"Error during random sampling: {e}")
                continue
                
            if len(states) >= self.uncertainty_init_samples:
                break
                
        if len(states) == 0:
            logger.error("Failed to collect any valid samples for uncertainty model")
            return
            
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Calculate values using critic
        with torch.no_grad():
            values = self.critic(states).squeeze()
            
        # Calculate TD errors (reward - value) as proxy for prediction error
        td_errors = torch.abs(rewards - values)
            
        # Normalize TD errors to 0-1 range for target uncertainty
        targets = td_errors / (td_errors.max() + 1e-8)
        
        # Train uncertainty model
        logger.info(f"Training uncertainty model with {len(states)} samples...")
        for epoch in range(self.uncertainty_init_epochs):
            loss = self.uncertainty_model.train_step(states, actions, targets)
            if epoch % 1 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss:.6f}")
                
        logger.info("Uncertainty model initialization complete")
        
    def forward_with_uncertainty(self, state):
        """
        Forward pass that considers uncertainty for active learning.
        Returns actions, log_probs, value, and uncertainty.
        """
        # Convert state if needed and keep original for parent call
        original_state = state
        if torch.is_tensor(state):
            state_tensor = state
        else:
            state_tensor = torch.from_numpy(state).float().to(self.device)
            
        # Create multiple candidate actions
        num_candidates = 5
        candidate_actions = []
        candidate_log_probs = []
        candidate_values = []
        
        # Sample multiple actions using the parent class forward method
        for _ in range(num_candidates):
            # Pass original state to avoid double conversion in parent forward method
            action, log_prob, value = super().forward(original_state)
            candidate_actions.append(torch.FloatTensor(action))
            candidate_log_probs.append(log_prob)
            candidate_values.append(value)
            
        # Calculate uncertainty for each candidate action
        uncertainties = []
        for action in candidate_actions:
            # Evaluate uncertainty
            uncertainty = self.uncertainty_model(
                state_tensor.unsqueeze(0) if state_tensor.dim() == 1 else state_tensor,
                action.unsqueeze(0).to(self.device) if action.dim() == 1 else action.to(self.device)
            )
            uncertainties.append(uncertainty.item())
            
        # Select action based on uncertainty and exploration factor
        if np.random.rand() < self.exploration_factor:
            # Choose action with highest uncertainty when exploring
            selected_idx = np.argmax(uncertainties)
        else:
            # Choose action based on policy (first action) when exploiting
            selected_idx = 0
            
        # Get the selected action
        action = candidate_actions[selected_idx].cpu().numpy()
        log_prob = candidate_log_probs[selected_idx]
        value = candidate_values[selected_idx]
        uncertainty = uncertainties[selected_idx]
        logger.info(f"Selected action: {action}, log_prob: {log_prob}, value: {value}, uncertainty: {uncertainty}")
        
        return action, log_prob, value, uncertainty
        
    def decay_exploration(self):
        """Decay the exploration factor over time"""
        self.exploration_factor = max(self.min_exploration, 
                                      self.exploration_factor * self.exploration_decay)
                                      
    def update_uncertainty_model(self, experiences):
        """Update the uncertainty model based on prediction errors"""
        # Extract data from experiences
        states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
        
        # Handle actions correctly
        actions_list = []
        for exp in experiences:
            # If actions is already a tensor or numpy array
            if isinstance(exp['actions'], (np.ndarray, list, tuple)):
                actions_list.append(torch.FloatTensor(exp['actions']))
            else:
                actions_list.append(exp['actions'])
        actions = torch.stack(actions_list).to(self.device)
        
        # Handle rewards
        rewards_list = []
        for exp in experiences:
            if isinstance(exp['reward'], (np.ndarray, float, int)):
                rewards_list.append(torch.tensor(float(exp['reward'])))
            else:
                rewards_list.append(exp['reward'])
        rewards = torch.stack(rewards_list).to(self.device)
        
        # Get value predictions
        with torch.no_grad():
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
        
        return loss.item()
        
    def update_ppo_active_learning(self, experiences):
        """
        PPO update that weights experiences by their uncertainty/information gain.
        Higher uncertainty experiences get more weight in the update.
        """
        # First update the uncertainty model
        uncertainty_loss = self.update_uncertainty_model(experiences)
        
        # Extract data from experiences with uncertainty weighting
        states = torch.FloatTensor([exp['state'] for exp in experiences]).to(self.device)
        
        # Handle actions correctly
        actions_list = []
        for exp in experiences:
            if isinstance(exp['actions'], (np.ndarray, list, tuple)):
                actions_list.append(torch.FloatTensor(exp['actions']))
            else:
                actions_list.append(exp['actions'])
        actions = torch.stack(actions_list).to(self.device)
        
        # Handle log_probs
        log_probs_list = []
        for exp in experiences:
            if isinstance(exp['log_probs'], (np.ndarray, list, tuple)):
                log_probs_list.append(torch.FloatTensor(exp['log_probs']))
            else:
                log_probs_list.append(exp['log_probs'])
        old_log_probs = torch.stack(log_probs_list).to(self.device)
        
        # Handle rewards
        rewards_list = []
        for exp in experiences:
            if isinstance(exp['reward'], (np.ndarray, float, int)):
                rewards_list.append(torch.tensor(float(exp['reward'])))
            else:
                rewards_list.append(exp['reward'])
        rewards = torch.stack(rewards_list).to(self.device)
        
        # Handle values
        values_list = []
        for exp in experiences:
            if isinstance(exp['value'], (np.ndarray, float, int)):
                values_list.append(torch.tensor(float(exp['value'])))
            else:
                values_list.append(exp['value'])
        values = torch.stack(values_list).to(self.device)
        
        # Handle dones
        dones = torch.FloatTensor([exp['done'] for exp in experiences]).to(self.device)
        
        # Get uncertainties or use default if not available
        uncertainties = []
        for exp in experiences:
            if 'uncertainty' in exp:
                uncertainties.append(exp['uncertainty'])
            else:
                # If uncertainty not stored, get it from model
                with torch.no_grad():
                    state = torch.FloatTensor(exp['state']).to(self.device)
                    action = torch.FloatTensor(exp['actions']).to(self.device)
                    uncertainty = self.uncertainty_model(state.unsqueeze(0), action.unsqueeze(0)).item()
                uncertainties.append(uncertainty)
                
        uncertainties = torch.FloatTensor(uncertainties).to(self.device)
        
        # Normalize uncertainties for weighting (higher uncertainty = higher weight)
        weights = uncertainties / (uncertainties.mean() + 1e-8)
        
        # Create weighted experiences for the parent update method
        weighted_experiences = []
        for i, exp in enumerate(experiences):
            weighted_exp = exp.copy()
            weighted_exp['reward'] = rewards[i].item() * weights[i].item()
            weighted_experiences.append(weighted_exp)
            
        # Call parent class update method
        super().update_ppo(weighted_experiences)
        
        return uncertainty_loss