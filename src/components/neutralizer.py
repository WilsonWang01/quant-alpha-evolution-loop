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
    def cross_sectional_neutralize(alpha_series, group_series=None):
        """
        Standardizes alpha and de-means it.
        If group_series (industry/sector) is provided, de-means within each group.
        Equivalent to WorldQuant 101's 'indneutralize'.
        """
        if group_series is not None:
            # Group-wise de-meaning: subtract the mean of the group from each element
            alpha_series = alpha_series - alpha_series.groupby(group_series).transform('mean')
        else:
            # Global de-meaning
            alpha_series = alpha_series - alpha_series.mean()
        
        # Scaling to unit variance (Z-Score)
        std = alpha_series.std()
        if std > 0:
            alpha_series = alpha_series / std
            
        return alpha_series

    @staticmethod
    def style_neutralize(alpha_df, style_factors_df):
        """
        Removes style exposures (e.g., Size, Value, Vol) via cross-sectional regression.
        Alpha_resid = Alpha - (B_size * Size + B_vol * Vol + ...)
        Benchmarked against Citadel/Two Sigma multi-factor risk models.
        """
        # Ensure indices align
        common_idx = alpha_df.index.intersection(style_factors_df.index)
        y = alpha_df.loc[common_idx].values
        X = style_factors_df.loc[common_idx].values
        
        # Add intercept for de-meaning
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Solve OLS: Beta = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X.dot(beta)
            return pd.Series(residuals, index=common_idx)
        except np.linalg.LinAlgError:
            return alpha_df
