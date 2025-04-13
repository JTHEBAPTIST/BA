
def solve_ch15(pret, ret_sp500):
    import numpy as np

    T = len(pret)

    # Sharpe Ratio
    SR = pret.mean() / pret.std()

    # Information Ratio via regression residuals
    X = np.vstack([np.ones(T), ret_sp500]).T
    beta = np.linalg.inv(X.T @ X) @ X.T @ pret
    residuals = pret - X @ beta
    IR = beta[0] / residuals.std()

    # Tracking Error IR
    IR2 = (pret - ret_sp500).mean() / (pret - ret_sp500).std()

    return SR, IR, IR2
