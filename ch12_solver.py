
import numpy as np
from scipy.optimize import minimize

def solve_ch12(mu, sigma, beta, risk_aversion, target_beta, V, q, S, beta_f):
    n = len(mu)

    def objective(w):
        return risk_aversion * w @ sigma @ w - mu @ w

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(objective, w0, bounds=bounds, constraints=constraints)
    w_opt = result.x

    beta_p = np.dot(w_opt, beta)
    num_contracts = (target_beta - beta_p) * V / (q * S * beta_f)

    expected_return = np.dot(mu, w_opt) + num_contracts * beta_f * (mu @ beta / np.sum(beta**2))
    volatility = np.sqrt(w_opt @ sigma @ w_opt + 2 * num_contracts * np.dot(w_opt, sigma @ beta) + (num_contracts**2) * np.dot(beta, sigma @ beta))

    result_summary = {
        "weights": {f"Stock{i+1}": w for i, w in enumerate(w_opt) if w > 0.001},
        "portfolio_beta": beta_p,
        "required_contracts": num_contracts,
        "expected_return": expected_return,
        "volatility": volatility
    }

    return result_summary
