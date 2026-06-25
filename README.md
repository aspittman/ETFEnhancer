## Setup

```bash
git clone https://github.com/aspittman/ETFEnhancer.git
cd ETFEnhancer

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Live Bot Controls

Key risk and selection settings live in `config.py`:

- `BLOCKED_SYMBOLS`: symbols blocked from new entries. `XLY` is blocked by default.
- `MARKET_REGIME_SYMBOL`: benchmark used to decide whether new buys are allowed.
- `MIN_CANDIDATE_SCORE`: minimum ranked score required before a symbol can be bought.
- `MAX_BUYS_PER_CYCLE`: maximum new buys per scan cycle.
- `ATR_WINDOW` and `ATR_TRAILING_MULTIPLIER`: ATR trailing stop settings.

The bot still manages existing positions even when a symbol is blocked or the market
regime is weak.

## Backtesting

Run the strategy offline against historical Yahoo Finance data:

```bash
python backtest.py
```

Useful options:

```bash
python backtest.py --symbols SPY,QQQ,NVDA --period 2y --interval 1h
python backtest.py --min-score 3 --max-buys-per-bar 1
```

The backtester uses the same entry signal, scoring, market regime filter, fixed stop
loss, and ATR trailing stop logic as the live bot.

## Trade Analytics

Analyze paired live trades from `logs/trades.csv`:

```bash
python analytics.py logs/trades.csv
```

The report includes win rate, expectancy, total P/L, profit factor, best/worst
symbols, and per-symbol stats.
