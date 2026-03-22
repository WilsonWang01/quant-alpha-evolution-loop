import numpy as np

class InstitutionalSlippageModel:
    """
    Simulates institutional execution costs based on volume participation.
    Inspired by TradeMaster's high-fidelity market simulators.
    """
    def __init__(self, participation_rate=0.05, spread_cost=0.0001):
        self.participation_rate = participation_rate  # 0.05 means we won't trade more than 5% of daily volume
        self.spread_cost = spread_cost # Fixed spread impact

    def estimate_impact(self, order_size, daily_volume, daily_volatility):
        """
        Calculates price impact using Square Root Model (Almgren-Chriss logic).
        Impact = Sigma * sqrt(Size / Volume)
        """
        if daily_volume == 0:
            return 0.01 # Severe penalty for zero liquidity
        
        relative_size = order_size / daily_volume
        impact = daily_volatility * np.sqrt(relative_size / self.participation_rate)
        return self.spread_cost + impact
