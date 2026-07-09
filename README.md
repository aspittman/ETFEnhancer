## Setup

```bash
git clone https://github.com/aspittman/ETFEnhancer.git
cd ETFEnhancer

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file with Alpaca credentials:

```bash
API_KEY=your_alpaca_key
SECRET_KEY=your_alpaca_secret
ALPACA_PAPER=true
```

Use `ALPACA_PAPER=true` with Alpaca paper keys and `ALPACA_PAPER=false` with
live keys. Mixing paper/live keys and mode will make Alpaca return
`request is not authorized`.

## Live Bot Controls

Key risk and selection settings live in `config.py`:

- `ETF_ONLY_MODE`: defaults to `True`; the live universe is ETF-only by default.
- `BLOCKED_SYMBOLS`: symbols blocked from new entries. Empty by default.
- `MARKET_REGIME_SYMBOL`: benchmark used to decide whether new buys are allowed.
  New long entries require SPY to be above its 200-day moving average and its
  50-day moving average to be above its 200-day moving average.
- `MIN_CANDIDATE_SCORE`: optional minimum ranked score. `None` disables the
  minimum-score gate.
- `MAX_NEW_BUYS_PER_CYCLE`: maximum new buys per scan cycle.
- `ATR_WINDOW` and `ATR_TRAILING_MULTIPLIER`: ATR trailing stop settings.
  ATR trailing stops are available but disabled by default.
- `ENABLE_MARKET_REGIME_FILTER`, `ENABLE_MA_ALIGNMENT_FILTER`,
  `ENABLE_MACD_FILTER`, `ENABLE_RELATIVE_STRENGTH_SCORE`,
  `ENABLE_MOMENTUM_SCORE`, `ENABLE_VOLUME_FILTER`, `ENABLE_ATR_TREND_FILTER`,
  `ENABLE_MIN_SCORE_FILTER`, and `ENABLE_TOP_CANDIDATE_SELECTION`:
  independently enable or disable entry and ranking behavior. Volume, ATR trend,
  and minimum-score gates are disabled by default.
- `ENABLE_ATR_TRAILING_STOP` and `ENABLE_FIXED_STOP_LOSS`: independently
  enable or disable exit protection rules.

The bot still manages existing positions even when a symbol is blocked or the market
regime is weak.

## Backtesting

Run the strategy offline against historical Yahoo Finance data:

```bash
python backtest.py
```

By default this runs the ETF-only universe. Use `--universe` for other reports:

```bash
python backtest.py --universe etf
python backtest.py --universe stock
python backtest.py --universe combined
python backtest.py --universe all
```

Useful options:

```bash
python backtest.py --symbols SPY,QQQ,XLK --period 2y --interval 1h
python backtest.py --min-score 3 --max-buys-per-bar 1
python backtest.py --filter-impact --universe etf --period 2y
```

The backtester uses the same entry signal, relative-strength scoring versus SPY,
market regime filter, fixed stop loss, and optional ATR trailing stop logic as the live bot.
`--filter-impact` prints total P/L, expectancy, win rate, profit factor, and deltas
for each filter by toggling one switch at a time.

## Trade Analytics

Analyze paired live trades from `logs/trades.csv`:

```bash
python analytics.py logs/trades.csv
```

The report includes win rate, expectancy, total P/L, profit factor, best/worst
symbols, and per-symbol stats.
