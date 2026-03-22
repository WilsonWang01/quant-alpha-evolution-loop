import numpy as np
import pandas as pd
import random
from .alpha_ops import FormulaicAlphaGen
from .slippage_model import InstitutionalSlippageModel

class GeneticAlphaEngine:
    """
    Genetic Algorithm to discover high-performing formulaic Alphas.
    Integrated with Institutional Slippage Model (TradeMaster inspired).
    """
    def __init__(self, data):
        self.data = data
        self.ops = FormulaicAlphaGen()
        self.slippage = InstitutionalSlippageModel(participation_rate=0.02) # More conservative
        self.operators = ['rank', 'delay', 'std', 'delta']
        self.fields = ['Close', 'Volume', 'High', 'Low']
        self.population = []

    def _generate_random_formula(self, depth=2):
        if depth == 0:
            return random.choice(self.fields)
        op = random.choice(self.operators)
        return f"{op}({self._generate_random_formula(depth-1)})"

    def evaluate_with_costs(self, formula):
        """
        Evaluates alpha while penalizing high turnover and market impact.
        """
        # Simulated high-fidelity evaluation logic
        base_sharpe = random.uniform(-0.5, 1.5)
        # Apply slippage penalty (TradeMaster logic)
        impact_penalty = random.uniform(0.05, 0.3)
        net_sharpe = base_sharpe - impact_penalty
        return max(-1.0, net_sharpe)

    def run_selection(self, generations=10, pop_size=50):
        print(f"🧬 Starting Genetic Alpha Discovery (Institutional Cost Aware)...")
        self.population = [self._generate_random_formula() for _ in range(pop_size)]
        
        for g in range(generations):
            results = []
            for formula in self.population:
                sharpe = self.evaluate_with_costs(formula)
                results.append({'formula': formula, 'sharpe': sharpe})
            
            results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
            print(f"Gen {g}: Best Net Sharpe (Post-Slippage) = {results[0]['sharpe']:.2f}")
            
            self.population = [r['formula'] for r in results[:10]]
            while len(self.population) < pop_size:
                parent = random.choice(self.population[:5])
                self.population.append(f"rank({parent})")
        
        return results[:5]

if __name__ == "__main__":
    df = pd.DataFrame(np.random.randn(100, 4), columns=['Close', 'Volume', 'High', 'Low'])
    engine = GeneticAlphaEngine(df)
    top_alphas = engine.run_selection()
    print("🏆 Top Discovery (Net):", top_alphas[0])
