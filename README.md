# Event Surface Mispricing Identifier

This project scans Polymarket single-stock price events, builds a risk-neutral terminal price distribution from listed options, and flags mispricings between:

- **Option-implied probabilities** (via Heston + Breeden–Litzenberger or Black–Scholes), and  
- **Prediction-market prices** (Polymarket YES/NO contracts).

When an edge is large enough, it computes how many shares you could buy at Polymarket before the **VWAP** crosses a target price, and logs the trade in a local portfolio.

> Educational / research tool only. Not investment advice.

---

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/Event-Surface-Mispricing-Identifier.git
cd Event-Surface-Mispricing-Identifier
pip install -r requirements.txt
pip install -e .
```

1. **Load securities from Polymarket**
   - `polymarket.load_secs(...)` finds relevant events by keyword (e.g. “close”), filters by volume, and extracts:
     - Underlying ticker (MSFT, NVDA, etc.)
     - Bucket/label (e.g. `$520–540`, `>540`, `<520`)
     - Event URL
     - Current Polymarket YES price  
   - Results are stored in a SQLite DB (`secs.db`) via the `Securities` class.

2. **Build risk-neutral distribution from options**
   - For a given security:
     - The event’s expiry is obtained from Polymarket (`get_event_expiry`).
     - `get_heston_inputs` pulls option chains from `yfinance`, builds mid-prices, and derives inputs for a Heston model (spot, τ, r, strikes, call/put mids, ATM IV).
     - `Heston` plus a Breeden–Litzenberger second derivative gives an approximate **risk-neutral PDF** over terminal prices.

3. **Compute mispricing (“edge”)**
   - `Pricer` maps the Polymarket bucket label (like `$520–540` or `<520`) into a numeric price range.
   - It integrates the risk-neutral PDF over that range to get the option-implied probability.
   - It compares that to the Polymarket probability for YES and produces an edge.

4. **Size a bet and log it**
   - `Better`:
     - Takes the option-implied prob and Polymarket prob for a security.
     - If the absolute edge ≥ `min_edge`, decides which side to take (YES/NO).
     - Uses `max_invest_for_side_at_price` to walk the Polymarket order book and find:
       - Maximum shares you can buy on that side
       - Total cost
       - VWAP, constrained to stay below a target price (default: best-mid + small bump)
     - Stores the resulting position (sec_id, ticker, label, side, qty, vwap) in a `positions` SQLite DB via the `Portfolio` class.

5. **Engine**
   - `engine.main()` wires it all together:
     - Instantiates `Securities`, `Pricer`, and `Better`
     - Optionally seeds the `secs` table from Polymarket
     - Iterates over all stored securities, computes edges, and attempts bets.

---

## Project Structure

```text
Event-Surface-Mispricing-Identifier/
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── src/
│   └── esmi/
│       ├── __init__.py
│       ├── engine.py          # orchestrates the full pipeline
│       ├── polymarket.py      # Polymarket/Gamma/CLOB API helpers
│       ├── pricer.py          # Heston/BL + Black–Scholes probability engine
│       ├── heston.py          # Heston model + call pricer
│       ├── heston_inputs.py   # pulls market data from yfinance
│       ├── black_scholes.py   # lognormal terminal distribution from ATM vol
│       ├── secs.py            # SQLite-backed Securities store (secs.db)
│       └── portfolio.py       # SQLite-backed Portfolio store (positions.db)
└── data/
    ├── secs.db
    └── positions.db
