"""
data_fetcher.py
Fetches live and historical market data for GOLD (XAUUSD) and DXY from Alpha Vantage.
Falls back to synthetic data if API key is demo or unavailable.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

INTERVAL_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1hour": "60min",
    "daily": "daily",
}


def fetch_intraday(symbol: str, interval: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame:
    """Fetch intraday OHLCV data from Alpha Vantage."""
    av_interval = INTERVAL_MAP.get(interval, "5min")
    
    if av_interval == "daily":
        function = "TIME_SERIES_DAILY"
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": api_key,
        }
    else:
        function = "TIME_SERIES_INTRADAY"
        params = {
            "function": function,
            "symbol": symbol,
            "interval": av_interval,
            "outputsize": outputsize,
            "apikey": api_key,
        }

    try:
        resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Parse response
        ts_key = [k for k in data.keys() if "Time Series" in k]
        if not ts_key:
            logger.warning(f"No time series data for {symbol} {interval}. Using synthetic data.")
            return generate_synthetic_data(symbol, interval)

        ts = data[ts_key[0]]
        records = []
        for dt_str, ohlcv in ts.items():
            records.append({
                "datetime": pd.to_datetime(dt_str),
                "open": float(ohlcv.get("1. open", ohlcv.get("1. open", 0))),
                "high": float(ohlcv.get("2. high", 0)),
                "low": float(ohlcv.get("3. low", 0)),
                "close": float(ohlcv.get("4. close", 0)),
                "volume": float(ohlcv.get("5. volume", ohlcv.get("6. volume", 1000))),
            })

        df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
        df.set_index("datetime", inplace=True)
        return df

    except Exception as e:
        logger.error(f"Error fetching {symbol} {interval}: {e}")
        return generate_synthetic_data(symbol, interval)


def generate_synthetic_data(symbol: str, interval: str, n_bars: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for testing/demo."""
    np.random.seed(42 + hash(symbol + interval) % 100)
    
    # Base price
    if "XAU" in symbol or "GOLD" in symbol.upper():
        base_price = 2350.0
        volatility = 0.003
        volume_base = 5000
    elif "DXY" in symbol.upper() or symbol == "USDX":
        base_price = 104.5
        volatility = 0.0008
        volume_base = 100000
    else:
        base_price = 100.0
        volatility = 0.002
        volume_base = 10000

    interval_minutes = {"1min": 1, "5min": 5, "15min": 15, "1hour": 60, "daily": 1440}.get(interval, 5)
    
    end_time = datetime.now().replace(second=0, microsecond=0)
    timestamps = [end_time - timedelta(minutes=i * interval_minutes) for i in range(n_bars, 0, -1)]

    prices = [base_price]
    for _ in range(n_bars - 1):
        # Add trend + mean reversion + noise
        trend = 0.0001 * np.sin(len(prices) / 50)
        shock = np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + trend + shock)
        prices.append(max(new_price, base_price * 0.8))

    records = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        spread = close * volatility * 0.5
        high = close + abs(np.random.normal(0, spread))
        low = close - abs(np.random.normal(0, spread))
        open_ = prices[i - 1] if i > 0 else close
        volume = volume_base * (1 + abs(np.random.normal(0, 0.5)))
        records.append({
            "datetime": ts,
            "open": round(open_, 3),
            "high": round(max(open_, close, high), 3),
            "low": round(min(open_, close, low), 3),
            "close": round(close, 3),
            "volume": round(volume, 0),
        })

    df = pd.DataFrame(records)
    df.set_index("datetime", inplace=True)
    return df


def fetch_gold_data(api_key: str, interval: str = "5min") -> pd.DataFrame:
    """Fetch XAUUSD data."""
    # Alpha Vantage uses XAUUSD for forex-like pair
    df = fetch_intraday("XAUUSD", interval, api_key)
    if df.empty or len(df) < 10:
        df = generate_synthetic_data("XAUUSD", interval)
    return df


def fetch_dxy_data(api_key: str, interval: str = "5min") -> pd.DataFrame:
    """Fetch DXY (US Dollar Index) data."""
    # Try DXY as forex or use synthetic
    df = fetch_intraday("DXY", interval, api_key)
    if df.empty or len(df) < 10:
        df = generate_synthetic_data("DXY", interval)
    return df


def fetch_all_timeframes(api_key: str) -> dict:
    """Fetch gold data for 1min, 5min, 15min timeframes."""
    timeframes = {}
    for tf in ["1min", "5min", "15min"]:
        timeframes[tf] = fetch_gold_data(api_key, tf)
        time.sleep(0.5)  # Rate limiting
    timeframes["dxy"] = fetch_dxy_data(api_key, "5min")
    return timeframes


def get_current_price(api_key: str) -> dict:
    """Get the latest gold price."""
    try:
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": "XAU",
            "to_currency": "USD",
            "apikey": api_key,
        }
        resp = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
        data = resp.json()
        rate_info = data.get("Realtime Currency Exchange Rate", {})
        if rate_info:
            return {
                "price": float(rate_info.get("5. Exchange Rate", 2350.0)),
                "bid": float(rate_info.get("8. Bid Price", 2349.5)),
                "ask": float(rate_info.get("9. Ask Price", 2350.5)),
                "timestamp": rate_info.get("6. Last Refreshed", datetime.now().isoformat()),
            }
    except Exception as e:
        logger.warning(f"Could not get live price: {e}")
    
    # Fallback
    base = 2350.0
    noise = np.random.normal(0, 2)
    return {
        "price": round(base + noise, 2),
        "bid": round(base + noise - 0.5, 2),
        "ask": round(base + noise + 0.5, 2),
        "timestamp": datetime.now().isoformat(),
    }
