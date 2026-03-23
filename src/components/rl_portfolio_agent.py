import numpy as np

class DeepRLPortfolioAgent:
    """
    Deep Reinforcement Learning Agent for Portfolio Allocation.
    Inspired by TradeMaster (NTU) - PPO/EIIE concepts.
    
    This agent learns to allocate portfolio weights dynamically based on 
    market states (e.g., recent returns, volatility, momentum), replacing
    static heuristic rules.
    """
    def __init__(self, num_assets, state_dim=3):
        self.num_assets = num_assets
        self.state_dim = state_dim
        # Simplified Neural Network: Linear layer -> Softmax
        # In production, this would be a PyTorch nn.Module (e.g., LSTM or Transformer)
        self.actor_weights = np.random.randn(state_dim, num_assets)
        self.learning_rate = 0.05
        
    def get_action(self, state):
        """
        Forward pass. State shape: (num_assets, state_dim)
        Outputs allocation weights summing to 1.
        """
        # State processing: Dot product of state features and actor weights
        # For simplicity, we aggregate the state into logits
        logits = np.zeros(self.num_assets)
        for i in range(self.num_assets):
            logits[i] = np.dot(state[i], self.actor_weights[:, i])
            
        # Softmax activation for portfolio weights
        exp_preds = np.exp(logits - np.max(logits))
        action_weights = exp_preds / np.sum(exp_preds)
        return action_weights
        
    def update_policy(self, state, action, reward):
        """
        Simplified Policy Gradient update (REINFORCE-like).
        If reward is positive, increase probability of this action.
        """
        # Calculate gradients (simplified pseudo-gradient)
        for i in range(self.num_assets):
            # Move weights in direction of state features scaled by reward and action probability
            gradient = state[i] * (action[i] * reward)
            self.actor_weights[:, i] += self.learning_rate * gradient

# Singleton instance for the loop
rl_agent = DeepRLPortfolioAgent(num_assets=4)
