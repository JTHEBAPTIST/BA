
print("\n==== Simple Portfolio Generators ====")
from simple_portfolios import generate_simple_portfolios

market_cap = np.random.uniform(1e8, 5e9, size=data["n"])
simple_portfolios = generate_simple_portfolios(data["tickers"], data["mu"], data["sigma"], market_cap)

for name, df in simple_portfolios.items():
    print(f"\n{name} Portfolio (Top 5 Holdings):")
    print(df[['Ticker', 'Stock Weight']].head(5).to_string(index=False))
