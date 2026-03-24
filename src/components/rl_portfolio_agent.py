import numpy as np
import os

class DeepRLPortfolioAgent:
    """
    Deep Reinforcement Learning Agent for Portfolio Allocation.
    Inspired by TradeMaster (NTU) - PPO/EIIE concepts.
    
    This agent learns to allocate portfolio weights dynamically based on 
    market states (e.g., recent returns, volatility, momentum), replacing
    static heuristic rules.
    """
    def __init__(self, num_assets, state_dim=3, model_path="models/rl_weights.npy"):
        self.num_assets = num_assets
        self.state_dim = state_dim
        self.model_path = os.path.join(os.path.dirname(__file__), "../../", model_path)
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        if os.path.exists(self.model_path):
            self.actor_weights = np.load(self.model_path)
            print(f"✅ RL Agent: Loaded weights from {self.model_path}")
        else:
            self.actor_weights = np.random.randn(state_dim, num_assets)
            print("🆕 RL Agent: Initialized with random weights")
            
        self.learning_rate = 0.05
        
    def save_weights(self):
        """Persist weights to disk."""
        np.save(self.model_path, self.actor_weights)
        print(f"💾 RL Agent: Saved weights to {self.model_path}")
        
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
