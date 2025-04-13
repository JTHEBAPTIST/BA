
import numpy as np
import pandas as pd

def generate_simple_portfolios(tickers, mu, sigma, market_cap):
    n = len(tickers)
    today = pd.Timestamp.today().strftime('%m/%d/%Y')

    def format_blotter(weights, name):
        return pd.DataFrame({
            'CoName': [f"Company {i+1}" for i in range(n)],
            'Exchange': ['NYSE'] * n,
            'Dates': [today] * n,
            'CUSIP': [f"CUSIP{i:03d}" for i in range(n)],
            'Sector': ['Tech'] * n,
            'Industry': ['Software'] * n,
            'GICS Sector': ['Information Technology'] * n,
            'AggZScore': np.random.normal(0, 1, size=n),
            'NBeta': np.random.uniform(0.5, 1.5, size=n),
            'Ticker': tickers,
            'Stock Weight': weights,
            'Latest Price': [np.nan] * n
        }).loc[weights > 0].sort_values(by="Stock Weight", ascending=False).reset_index(drop=True)

    # Equal-weight portfolio
    eq_weights = np.ones(n) / n
    equal_weight_df = format_blotter(eq_weights, "EqualWeight")

    # Market-cap weighted
    mc_weights = market_cap / np.sum(market_cap)
    market_cap_df = format_blotter(mc_weights, "MarketCapWeighted")

    # Random weights
    rand = np.random.rand(n)
    rand_weights = rand / np.sum(rand)
    random_df = format_blotter(rand_weights, "RandomWeight")

    # Max Sharpe (naive unconstrained)
    def neg_sharpe(w):
        return -(mu @ w) / (np.sqrt(w @ sigma @ w) + 1e-8)
    from scipy.optimize import minimize
    w0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    maxsharpe_df = format_blotter(res.x, "MaxSharpe")

    return {
        'EqualWeight': equal_weight_df,
        'MarketCapWeighted': market_cap_df,
        'RandomWeight': random_df,
        'MaxSharpe': maxsharpe_df
    }
