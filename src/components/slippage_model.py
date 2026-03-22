import numpy as np

class InstitutionalSlippageModel:
    """
    Simulates institutional execution costs based on volume participation.
    Inspired by TradeMaster's high-fidelity market simulators.
    """
    def __init__(self, participation_rate=0.05, spread_cost=0.0001):
        self.participation_rate = participation_rate  # 0.05 means we won't trade more than 5% of daily volume
        self.spread_cost = spread_cost # Fixed spread impact

    def estimate_impact(self, order_size, daily_volume, daily_volatility, current_price):
        """
        Calculates price impact as a percentage of current price.
        Impact_Pct = (Sigma_Pct * sqrt(Size / Volume)) / Participation_Const
        """
        if daily_volume <= 0 or current_price <= 0:
            return 0.005 # Default penalty
        
        relative_size = order_size / daily_volume
        # Normalize volatility to percentage
        vol_pct = daily_volatility / current_price
        
        impact_pct = vol_pct * np.sqrt(relative_size / self.participation_rate)
        return self.spread_cost + impact_pct
