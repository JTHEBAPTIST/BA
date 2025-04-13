
import numpy as np

def solve_ch14(sigma, w0, risk_aversion):
    mu_bl = 0.1 * risk_aversion * sigma @ w0
    return mu_bl
