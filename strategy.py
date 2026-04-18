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

def check_signal(symbol, rsi_period, oversold, overbought):
    data = yf.download(symbol, period="5d", interval="5m")

    close = data["Close"].squeeze()

    rsi = ta.momentum.RSIIndicator(
        close=close,
        window=rsi_period
    ).rsi().iloc[-1]

    print(f"Current RSI: {rsi:.2f}")

    if rsi < oversold:
        return "buy"
    elif rsi > overbought:
        return "sell"
    else:
        return "hold"
