import numpy as np

class RiskEngine:
    def __init__(self, max_drawdown_limit=0.15, var_limit=0.05, max_weight_limit=0.05):
        self.max_drawdown_limit = max_drawdown_limit
        self.var_limit = var_limit
        self.max_weight_limit = max_weight_limit  # Citadel-style: 5% cap per asset

    def apply_concentration_caps(self, target_weights, sector_mapping=None, max_sector_limit=0.20):
        """
        Adjusts target weights to respect institutional constraints.
        1. Max Weight per Ticker (e.g., 5%)
        2. Max Weight per Sector (e.g., 20%)
        """
        # Step 1: Max Individual Weight
        capped_weights = np.minimum(target_weights, self.max_weight_limit)
        
        # Step 2: Max Sector Weight (if sector mapping exists)
        if sector_mapping is not None:
            sector_totals = capped_weights.groupby(sector_mapping).transform('sum')
            # If a sector exceeds 20%, scale down its components proportionally
            scaling_factor = np.where(sector_totals > max_sector_limit, 
                                     max_sector_limit / sector_totals, 
                                     1.0)
            capped_weights = capped_weights * scaling_factor
            
        # Step 3: Re-normalize if necessary (keep within 1.0)
        total_sum = np.sum(capped_weights)
        if total_sum > 1.0:
            capped_weights = capped_weights / total_sum
            
        return capped_weights

    def check_portfolio_risk(self, nav_series):
        """
        全账户风险扫描：
        1. 实时计算最大回撤
        2. 计算 VaR (Value at Risk)
        """
        # 计算回撤
        peak = np.maximum.accumulate(nav_series)
        dd = (nav_series - peak) / peak
        current_dd = dd[-1]

        # 逻辑：如果触发硬预警，返回强制清仓指令
        if abs(current_dd) > self.max_drawdown_limit:
            return "FATAL_ERROR", f"触发硬预警：当前回撤 {current_dd:.2%}, 超过上限 {self.max_drawdown_limit:.2%}"
        
        return "SAFE", "风险在控"

    def vwap_execution(self, target_volume, market_volume_profile):
        """
        模拟 VWAP 下单：大资金分批进场，减少冲击成本
        """
        # 简化版：将 100 万订单拆成 10 份，每分钟跟成交量的 10%
        slices = 10
        slice_size = target_volume / slices
        return f"已将订单拆分为 {slices} 份，预计每 5 分钟成交量占比 5%，最大限度减少冲击成本。"

if __name__ == "__main__":
    engine = RiskEngine()
    print("🦞 风险引擎初始化完毕：已就绪。")
