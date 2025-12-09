import yfinance as yf

def get_latest_risk_free_rate() -> float:
    irx = yf.Ticker('^IRX')
    hist = irx.history(period='5d')
    if hist.empty:
        return 0.05

    last = hist['Close'].iloc[-1]
    return float(last) / 100.0

def get_closest_expiry():
    ...