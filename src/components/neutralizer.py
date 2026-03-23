import pandas as pd
import numpy as np

class FactorNeutralizer:
    """
    Implements factor neutralization techniques.
    Benchmarking WorldQuant/Institutional standards.
    """
    @staticmethod
    def neutralize_beta(returns, market_returns):
        """
        Removes market beta from asset returns via linear regression.
        Alpha = Returns - (Beta * Market_Returns)
        """
        if returns.empty or market_returns.empty:
            return returns
            
        common_idx = returns.index.intersection(market_returns.index)
        y = returns.loc[common_idx]
        x = market_returns.loc[common_idx]
        
        # Calculate Beta = Cov(r, m) / Var(m)
        covariance = np.cov(y, x)[0][1]
        market_variance = np.var(x)
        
        if market_variance == 0:
            return y
            
        beta = covariance / market_variance
        alpha_returns = y - (beta * x)
        return alpha_returns, beta

    @staticmethod
    def scale_to_unit_exposure(weights):
        """
        Ensures weights sum to 1 (long-only) or neutralizes dollar exposure.
        """
        total = np.sum(np.abs(weights))
        if total == 0:
            return weights
        return weights / total
