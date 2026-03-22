import pandas as pd
import numpy as np

class FormulaicAlphaGen:
    """
    Generator for formulaic Alphas based on basic operators.
    Inspired by AlphaGen and WorldQuant 101 Alphas.
    """
    @staticmethod
    def rank(series):
        return series.rank(pct=True)

    @staticmethod
    def delay(series, period=1):
        return series.shift(period)

    @staticmethod
    def std(series, period=20):
        return series.rolling(period).std()

    @staticmethod
    def delta(series, period=1):
        return series.diff(period)

    def generate_liquidity_alpha(self, data):
        """
        Alpha = Rank(Turnover / Volatility) -> Rewards high volume but low relative volatility
        """
        turnover = data['Volume'] * data['Close']
        volatility = data['Close'].rolling(20).std()
        alpha = self.rank(turnover / volatility)
        return alpha
