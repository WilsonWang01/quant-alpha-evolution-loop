import pandas as pd
import numpy as np
import sys
import os

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.neutralizer import FactorNeutralizer
from quant_risk_engine import RiskEngine

def test_neutralization():
    print("Testing FactorNeutralizer...")
    # Test Cross-Sectional Neutralization
    alpha = pd.Series([0.1, 0.5, 0.9, -0.2], index=['A', 'B', 'C', 'D'])
    sector = pd.Series(['IT', 'IT', 'Energy', 'Energy'], index=['A', 'B', 'C', 'D'])
    
    neutralized = FactorNeutralizer.cross_sectional_neutralize(alpha, sector)
    print(f"Neutralized Alpha:\n{neutralized}")
    
    # IT (0.1, 0.5) Mean = 0.3. Residuals = (-0.2, 0.2)
    # Energy (0.9, -0.2) Mean = 0.35. Residuals = (0.55, -0.55)
    # Standardized should mean sum within group = 0
    assert np.isclose(neutralized.groupby(sector).mean().sum(), 0)
    print("✅ FactorNeutralizer (Cross-Sectional) passed.")

def test_risk_caps():
    print("Testing RiskEngine Concentration Caps...")
    engine = RiskEngine(max_weight_limit=0.05)
    
    # Create target weights where one stock has 25% (exceeds 5% limit)
    target_weights = pd.Series([0.25, 0.04, 0.02, 0.01], index=['A', 'B', 'C', 'D'])
    sector = pd.Series(['IT', 'IT', 'Energy', 'Finance'], index=['A', 'B', 'C', 'D'])
    
    capped = engine.apply_concentration_caps(target_weights)
    print(f"Capped Weights:\n{capped}")
    
    assert capped['A'] <= 0.05
    assert np.sum(capped) <= 1.0
    print("✅ RiskEngine (Concentration Caps) passed.")

if __name__ == "__main__":
    try:
        test_neutralization()
        test_risk_caps()
        print("\n🦞 Engine Upgrade V2.5: Infrastructure tests passed.")
    except Exception as e:
        print(f"❌ Tests failed: {str(e)}")
        sys.exit(1)
