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
- `ENABLE_SPY_CORE` and `SPY_CORE_ALLOCATION_PERCENT`: keep 60% of the bot's
  capital in SPY while the market regime is healthy. The core exits when the
  regime turns weak and is not managed by tactical midpoint or ATR exits.
- `REQUIRE_POSITIVE_RELATIVE_STRENGTH`: tactical ETFs must outperform SPY over
  the strategy's relative-strength window before they can be purchased.
- `RANK_TACTICAL_BY_RELATIVE_STRENGTH`: ranks qualifying tactical candidates by
  excess return versus SPY before structural distance and blended score.
- `BLOCKED_SYMBOLS`: symbols blocked from new entries. Empty by default.
- `MARKET_REGIME_SYMBOL`: benchmark used to decide whether new buys are allowed.
  New long entries require SPY to be above its 200-day moving average and its
  50-day moving average to be above its 200-day moving average.
- `MIN_CANDIDATE_SCORE`: optional minimum ranked score. `None` disables the
  minimum-score gate.
- `MAX_NEW_BUYS_PER_CYCLE`: maximum new buys per scan cycle.
- `HARD_STOP_PERCENT`: independent catastrophic stop, defaulting to 8%.
- `ENABLE_STRUCTURAL_MIDPOINT_STOP`: primary exit. Weekly closing-price pivots
  must be confirmed by an opposite reversal before they can raise the stop.
- `PIVOT_REVERSAL_PERCENT`, `PIVOT_LOOKBACK_WEEKS`, and
  `MIN_WEEKS_BETWEEN_PIVOTS`: default to 6%, 16 weeks, and 3 weeks.
- `PIVOT_TIMEFRAME="1wk"` and `PIVOT_PRICE_SOURCE="close"`: define the
  completed-bar series used by the ZigZag state machine.
- `STRUCTURAL_STOP_EXIT_TIMEFRAME="1d"`: a completed daily close below the
  structural midpoint triggers the normal exit.
- `USE_TENTATIVE_HIGH_FOR_STOP=False`: candidate highs do not move the stop
  until a qualifying reversal confirms them.
- `ENABLE_DYNAMIC_MIDPOINT_STOP`: legacy per-price midpoint logic, disabled
  live and retained for backtest comparison only.
- `ENABLE_ATR_STOP`: optional legacy ATR comparison exit, disabled by default.
- `PULLBACK_LOOKBACK`, `PULLBACK_EMA_TOLERANCE`, and
  `PULLBACK_SMA_TOLERANCE`: control what qualifies as a pullback.
- `ENABLE_MARKET_REGIME_FILTER`, `ENABLE_MA_ALIGNMENT_FILTER`,
  `ENABLE_MACD_FILTER`, `ENABLE_RELATIVE_STRENGTH_SCORE`,
  `ENABLE_MOMENTUM_SCORE`, `ENABLE_VOLUME_FILTER`, `ENABLE_ATR_TREND_FILTER`,
  `ENABLE_MIN_SCORE_FILTER`, and `ENABLE_TOP_CANDIDATE_SELECTION`:
  independently enable or disable entry and ranking behavior. Volume, ATR trend,
  and minimum-score gates are disabled by default.
- `ENABLE_EMA_EXIT` and `ENABLE_MACD_EXIT` are disabled; EMA and MACD weakness
  do not close a live position.

The bot still manages existing positions even when a symbol is blocked or the market
regime is weak.

Live broker orders are restricted to `LIVE_TRADING_UNIVERSE`, which is an immutable
copy of the ETF-only universe. The final order function rejects any stock, crypto,
option, or other symbol outside that allowlist, regardless of which caller requested
the order. No stock universe is defined by this project.

## Backtesting

Run the strategy offline against historical Yahoo Finance data:

```bash
python backtest.py
```

The backtester uses the same ETF-only universe:

```bash
python backtest.py
```

Useful options:

```bash
python backtest.py --symbols SPY,QQQ,XLK --period 2y --interval 1h
python backtest.py --min-score 3 --max-buys-per-bar 1
python backtest.py --filter-impact --period 2y
python backtest.py --compare-exits --period 2y
python backtest.py --pivot-grid --period 2y
python backtest.py --walk-forward --window-months 6 --warmup-months 6 --cost-bps 5 --period 2y
```

The backtester uses the same pullback entry, relative-strength scoring versus SPY,
market regime filter, emergency stop, and dynamic midpoint stop as the live bot.
`--compare-exits` runs legacy per-price midpoint, confirmed weekly structural
midpoint, and ATR exits over one shared market-data set with
identical entry rules and reports profit factor, expectancy, P/L, average winner,
average loser, average holding period, win rate, drawdown, and trade count.
`--pivot-grid` tests 4%, 6%, 8%, and 10% reversals over 12-, 16-, and 26-week
lookbacks. Pivot confirmation is advanced sequentially after completed weekly
bars, without using a future reversal before it occurs.
`--filter-impact` prints total P/L, expectancy, win rate, profit factor, and deltas
for each filter by toggling one switch at a time.
`--walk-forward` runs independent sequential windows and reports strategy versus
SPY return, excess return, capital exposure, genuine strategy exits, forced
window-end exits, and estimated transaction costs. `--cost-bps 5` applies five
basis points on both entry and exit. The initial `--warmup-months 6` is used to
seed indicators and pivots and is excluded from scored windows.

## Trade Analytics

Analyze confirmed paper fills:

```bash
python analytics.py logs/trades_paper.csv
```

By default the report downloads hourly SPY prices and compares every completed
trade with SPY over the same entry-to-exit window. It also reports SPY's
buy-and-hold return from the first entry through the final exit. Use
`--benchmark QQQ` to select another benchmark or `--no-benchmark` for an
offline report.

Live and paper executions are deliberately kept separate in
`logs/trades_live.csv` and `logs/trades_paper.csv`. Each row also includes the
environment and Alpaca order ID. The older `logs/trades.csv` file is preserved
as a legacy record and is not appended to by new bot runs.

To analyze confirmed live fills instead:

```bash
python analytics.py logs/trades_live.csv
```

The report includes win rate, expectancy, total P/L, profit factor, best/worst
symbols, and per-symbol stats.
