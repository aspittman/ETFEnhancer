CORE_ETFS = [
    "SPY", "QQQ", "DIA", "IWM"
]

SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"
]

OPTIONAL_ETFS = [
    "SMH", "SOXX", "VGT", "VOO", "SCHD", "IYW", "IYT"
]

ETF_UNIVERSE = CORE_ETFS + SECTOR_ETFS + OPTIONAL_ETFS

UNIVERSE = ETF_UNIVERSE

# This is the final live-trading allowlist.
LIVE_TRADING_UNIVERSE = frozenset(UNIVERSE)


def is_live_trading_symbol(symbol):
    return isinstance(symbol, str) and symbol.strip().upper() in LIVE_TRADING_UNIVERSE
