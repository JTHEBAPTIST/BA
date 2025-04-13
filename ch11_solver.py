
import numpy as np
from scipy.optimize import minimize

def solve_ch11(mu, sigma, dividends, tax_rate=0.2, risk_aversion=2):
    n = len(mu)

    def objective(w):
        return -((mu - tax_rate * dividends) @ w - risk_aversion * w @ sigma @ w)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(objective, w0, bounds=bounds, constraints=constraints)
    return {f"Stock{i+1}": w for i, w in enumerate(result.x) if w > 0.001}, -result.fun
