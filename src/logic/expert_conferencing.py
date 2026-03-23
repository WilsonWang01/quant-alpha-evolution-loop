import json

class ExpertConferencing:
    """
    Simulates a Multi-Agent decision-making process for strategy validation.
    Inspired by lobster-quant-company skill.
    """
    def __init__(self):
        self.experts = ["researcher_a", "researcher_crypto", "risk_officer"]
        self.consensus_log = "logs/expert_consensus.log"

    def audit_strategy(self, strategy_name, metrics, weights=None):
        """
        Simulated expert会诊 logic.
        """
        audit_report = {
            "strategy": strategy_name,
            "status": "In Review",
            "expert_feedback": []
        }
        
        # Risk Officer Feedback: MDD Limit
        if metrics['mdd'] > 0.12:  # Tightened from 0.15 to 0.12 (Wilson's preference)
            audit_report['expert_feedback'].append({"expert": "risk_officer", "opinion": "REJECT: MDD exceeds professional redline (12%)"})
        else:
            audit_report['expert_feedback'].append({"expert": "risk_officer", "opinion": "PASS: Drawdown is within professional boundaries"})

        # Citadel-style Concentration Limit
        if weights is not None:
            max_weight = max(weights.values()) if isinstance(weights, dict) else max(weights)
            if max_weight > 0.30:
                audit_report['expert_feedback'].append({"expert": "risk_officer", "opinion": f"REJECT: Concentration risk too high ({max_weight:.2%}). Limit is 30%."})

        # A-Share Specialist Feedback
        if "A-Share" in strategy_name and metrics['sharpe'] < 0.3:
            audit_report['expert_feedback'].append({"expert": "researcher_a", "opinion": "REVISE: A-Share alpha decay detected. Suggest liquidity filtering."})
            
        with open(self.consensus_log, "a") as f:
            f.write(json.dumps(audit_report) + "\n")
            
        return audit_report

if __name__ == "__main__":
    conf = ExpertConferencing()
    test_metrics = {'sharpe': 0.2, 'mdd': 0.18}
    print(conf.audit_strategy("Alpha-V24+ A-Share Core", test_metrics))
