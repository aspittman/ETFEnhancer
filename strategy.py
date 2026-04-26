import yfinance as yf
import ta
import time

def wait_for_market_open(trading_client):
    print("Checking if market is open...")

    while True:
        clock = trading_client.get_clock()

        if clock.is_open:
            print("Market is OPEN! Starting strategy...")
            break
        else:
            print("Market closed. Waiting 60 seconds...")
            time.sleep(60)
            
def check_signal(symbol, ma_short, ma_long, macd_fast, macd_slow, macd_signal):
    try:
        data = yf.download(symbol, period="1y", interval="1h", progress=False)

        if data is None or data.empty or "Close" not in data.columns:
            return None

        close = data["Close"].squeeze()

        if close is None or len(close) < ma_long + 5:
            return None

        ma_short_series = close.rolling(window=ma_short).mean()
        ma_long_series = close.rolling(window=ma_long).mean()

        macd = ta.trend.MACD(
            close=close,
            window_slow=macd_slow,
            window_fast=macd_fast,
            window_sign=macd_signal
        )

        macd_line = macd.macd()
        macd_signal_line = macd.macd_signal()
        macd_hist = macd.macd_diff()

        latest_close = float(close.iloc[-1])
        latest_ma_short = float(ma_short_series.iloc[-1])
        latest_ma_long = float(ma_long_series.iloc[-1])
        prev_ma_short = float(ma_short_series.iloc[-2])

        latest_macd = float(macd_line.iloc[-1])
        latest_macd_signal = float(macd_signal_line.iloc[-1])
        latest_macd_hist = float(macd_hist.iloc[-1])

        values = [
            latest_close, latest_ma_short, latest_ma_long,
            prev_ma_short, latest_macd, latest_macd_signal, latest_macd_hist
        ]

        if any(v != v for v in values):  # NaN check
            return None

        in_uptrend = latest_close > latest_ma_short and latest_ma_short > latest_ma_long
        ma_rising = latest_ma_short > prev_ma_short
        macd_confirmed = latest_macd > latest_macd_signal and latest_macd_hist > 0

        if not (in_uptrend and ma_rising and macd_confirmed):
            return None

        score = (
            ((latest_close - latest_ma_short) / latest_ma_short) * 100
            + ((latest_ma_short - latest_ma_long) / latest_ma_long) * 100
            + latest_macd_hist
        )

        return {
            "symbol": symbol,
            "price": latest_close,
            "score": score,
            "ma_short": latest_ma_short,
            "ma_long": latest_ma_long,
            "macd": latest_macd,
            "macd_signal": latest_macd_signal,
            "macd_hist": latest_macd_hist
        }

    except Exception as e:
        print(f"Error checking {symbol}: {e}")
        return None


def scan_universe(universe, ma_short, ma_long, macd_fast, macd_slow, macd_signal):
    candidates = []

    for symbol in universe:
        print(f"Scanning {symbol}...")
        result = check_signal(symbol, ma_short, ma_long, macd_fast, macd_slow, macd_signal)

        if result:
            candidates.append(result)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates
