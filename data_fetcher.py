"""
data_fetcher.py
Fetches live GOLD (XAUUSD) and DXY data.

Source priority:
  1. Alpha Vantage  — FX_INTRADAY with from_symbol=XAU, to_symbol=USD
  2. Yahoo Finance  — GC=F (Gold Futures) via yfinance — FREE, no key needed
  3. Synthetic data — realistic random walk anchored to real current price

Run diagnose_api(api_key) to see exactly what is failing and why.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

# Current real-world base prices — update if way off
GOLD_BASE_PRICE = 3100.0
DXY_BASE_PRICE  = 104.0


# ════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC — call this to see exactly what is happening
# ════════════════════════════════════════════════════════════════════════════

def diagnose_api(api_key: str) -> dict:
    """Tests every endpoint. Returns dict with status of each source."""
    results = {}

    # 1. Alpha Vantage spot price
    try:
        r = requests.get(ALPHA_VANTAGE_BASE, params={
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": "XAU",
            "to_currency": "USD",
            "apikey": api_key,
        }, timeout=10)
        d = r.json()
        if "Realtime Currency Exchange Rate" in d:
            price = float(d["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
            results["av_spot"] = {"status": "OK", "price": price}
        elif "Note" in d:
            results["av_spot"] = {"status": "RATE_LIMITED", "msg": d["Note"][:100]}
        elif "Information" in d:
            results["av_spot"] = {"status": "INVALID_KEY", "msg": d["Information"][:100]}
        else:
            results["av_spot"] = {"status": "UNKNOWN", "raw": str(d)[:150]}
    except Exception as e:
        results["av_spot"] = {"status": "ERROR", "msg": str(e)}

    # 2. Alpha Vantage FX_INTRADAY
    try:
        r = requests.get(ALPHA_VANTAGE_BASE, params={
            "function": "FX_INTRADAY",
            "from_symbol": "XAU",
            "to_symbol": "USD",
            "interval": "5min",
            "outputsize": "compact",
            "apikey": api_key,
        }, timeout=10)
        d = r.json()
        ts_key = [k for k in d.keys() if "Time Series" in k]
        if ts_key:
            bars = len(d[ts_key[0]])
            price = float(list(d[ts_key[0]].values())[0].get("4. close", 0))
            results["av_fx_intraday"] = {"status": "OK", "bars": bars, "price": price}
        elif "Note" in d:
            results["av_fx_intraday"] = {"status": "RATE_LIMITED", "msg": d["Note"][:100]}
        elif "Information" in d:
            results["av_fx_intraday"] = {"status": "INVALID_KEY", "msg": d["Information"][:100]}
        else:
            results["av_fx_intraday"] = {"status": "NO_DATA", "raw": str(d)[:150]}
    except Exception as e:
        results["av_fx_intraday"] = {"status": "ERROR", "msg": str(e)}

    # 3. Yahoo Finance
    try:
        import yfinance as yf
        hist = yf.Ticker("GC=F").history(period="1d", interval="5m")
        if not hist.empty:
            results["yahoo"] = {"status": "OK", "bars": len(hist),
                                "price": round(float(hist["Close"].iloc[-1]), 2)}
        else:
            results["yahoo"] = {"status": "EMPTY"}
    except ImportError:
        results["yahoo"] = {"status": "NOT_INSTALLED",
                            "msg": "pip install yfinance"}
    except Exception as e:
        results["yahoo"] = {"status": "ERROR", "msg": str(e)}

    working = [k for k, v in results.items() if v.get("status") == "OK"]
    results["summary"] = {
        "working": working,
        "key_valid": any(results.get(k, {}).get("status") == "OK"
                         for k in ["av_spot", "av_fx_intraday"]),
        "has_live_data": len(working) > 0,
    }
    return results


# ════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — Alpha Vantage FX_INTRADAY
# ════════════════════════════════════════════════════════════════════════════

def _fetch_av_fx_intraday(from_sym, to_sym, interval, api_key):
    av_interval = {"1min": "1min", "5min": "5min", "15min": "15min",
                   "1hour": "60min"}.get(interval, "5min")
    try:
        r = requests.get(ALPHA_VANTAGE_BASE, params={
            "function": "FX_INTRADAY",
            "from_symbol": from_sym,
            "to_symbol": to_sym,
            "interval": av_interval,
            "outputsize": "compact",
            "apikey": api_key,
        }, timeout=15)
        d = r.json()

        if "Note" in d or "Information" in d:
            return pd.DataFrame()

        ts_key = [k for k in d.keys() if "Time Series" in k]
        if not ts_key:
            return pd.DataFrame()

        rows = []
        for dt_str, bar in d[ts_key[0]].items():
            rows.append({
                "datetime": pd.to_datetime(dt_str),
                "open":   float(bar.get("1. open",  0)),
                "high":   float(bar.get("2. high",  0)),
                "low":    float(bar.get("3. low",   0)),
                "close":  float(bar.get("4. close", 0)),
                "volume": float(bar.get("5. volume", 1000)),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("datetime").set_index("datetime")

        # Sanity check price range for gold
        latest = df["close"].iloc[-1]
        if from_sym == "XAU" and (latest < 500 or latest > 10000):
            logger.warning(f"AV suspicious XAU price {latest} — discarding")
            return pd.DataFrame()

        return df
    except Exception as e:
        logger.error(f"AV FX_INTRADAY error: {e}")
        return pd.DataFrame()


def _fetch_av_spot(api_key):
    try:
        r = requests.get(ALPHA_VANTAGE_BASE, params={
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": "XAU",
            "to_currency": "USD",
            "apikey": api_key,
        }, timeout=10)
        d = r.json()
        rate = d.get("Realtime Currency Exchange Rate", {})
        if rate:
            p = float(rate["5. Exchange Rate"])
            return p if 500 < p < 10000 else None
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — Yahoo Finance (FREE, no key needed)
# ════════════════════════════════════════════════════════════════════════════

def _fetch_yahoo(ticker_sym, interval):
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    yf_map = {"1min": "1m", "5min": "5m", "15min": "15m", "1hour": "1h", "daily": "1d"}
    yfi = yf_map.get(interval, "5m")
    period_map = {"1m": "5d", "5m": "30d", "15m": "30d", "1h": "60d", "1d": "1y"}
    period = period_map.get(yfi, "30d")

    try:
        hist = yf.Ticker(ticker_sym).history(period=period, interval=yfi, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()

        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index.name = "datetime"
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)

        # Sanity check
        latest = df["close"].iloc[-1]
        if "GC" in ticker_sym and (latest < 500 or latest > 10000):
            return pd.DataFrame()

        return df
    except Exception as e:
        logger.error(f"yfinance {ticker_sym} error: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — Synthetic fallback
# ════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(symbol, interval, n_bars=500, base_price=None):
    """Synthetic OHLCV anchored to a realistic current price."""
    if base_price is None:
        if any(x in symbol.upper() for x in ["XAU", "GOLD", "GC"]):
            base_price = GOLD_BASE_PRICE
            vol = 0.003
        elif any(x in symbol.upper() for x in ["DXY", "DX", "USD"]):
            base_price = DXY_BASE_PRICE
            vol = 0.0008
        else:
            base_price = 100.0
            vol = 0.002
    else:
        vol = 0.003 if base_price > 1000 else 0.0008

    vol_base = 5000

    mins = {"1min": 1, "5min": 5, "15min": 15, "1hour": 60, "daily": 1440}.get(interval, 5)
    end = datetime.now().replace(second=0, microsecond=0)
    times = [end - timedelta(minutes=i * mins) for i in range(n_bars, 0, -1)]

    prices = [base_price]
    rng = np.random.default_rng(int(time.time()) % 99999)
    for i in range(n_bars - 1):
        drift = 0.00005 * np.sin(i / 80)
        shock = rng.normal(0, vol)
        prices.append(max(prices[-1] * (1 + drift + shock), base_price * 0.85))

    rows = []
    for i, (ts, close) in enumerate(zip(times, prices)):
        spread = close * vol * 0.6
        high  = close + abs(rng.normal(0, spread))
        low   = close - abs(rng.normal(0, spread))
        op    = prices[i - 1] if i > 0 else close
        vol_v = vol_base * max(0.1, 1 + rng.normal(0, 0.5))
        rows.append({
            "datetime": ts,
            "open":   round(op, 3),
            "high":   round(max(op, close, high), 3),
            "low":    round(min(op, close, low), 3),
            "close":  round(close, 3),
            "volume": round(vol_v, 0),
        })

    df = pd.DataFrame(rows).set_index("datetime")
    return df


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def fetch_gold_data(api_key: str, interval: str = "5min"):
    """Returns (DataFrame, source_label)."""
    is_demo = api_key.lower().strip() in ("demo", "", "your_api_key_here")

    # 1. Alpha Vantage (real key only)
    if not is_demo:
        df = _fetch_av_fx_intraday("XAU", "USD", interval, api_key)
        if not df.empty and len(df) >= 10:
            return df, "Alpha Vantage Live"
        time.sleep(12)   # respect 5 req/min limit

    # 2. Yahoo Finance (always available)
    df = _fetch_yahoo("GC=F", interval)
    if not df.empty and len(df) >= 10:
        return df, "Yahoo Finance Live"

    # 3. Synthetic — try to anchor to real spot price
    spot = None
    if not is_demo:
        spot = _fetch_av_spot(api_key)
    if spot is None:
        try:
            import yfinance as yf
            h = yf.Ticker("GC=F").history(period="1d", interval="1m")
            if not h.empty:
                spot = float(h["Close"].iloc[-1])
        except Exception:
            pass

    base = spot if spot else GOLD_BASE_PRICE
    df = generate_synthetic_data("XAUUSD", interval, base_price=base)
    label = f"Synthetic (anchored ${base:,.0f})" if spot else f"Synthetic (${base:,.0f})"
    return df, label


def fetch_dxy_data(api_key: str, interval: str = "5min"):
    """Returns (DataFrame, source_label)."""
    df = _fetch_yahoo("DX-Y.NYB", interval)
    if not df.empty and len(df) >= 10:
        return df, "Yahoo Finance Live (DXY)"

    df = generate_synthetic_data("DXY", interval, base_price=DXY_BASE_PRICE)
    return df, "Synthetic DXY"


def fetch_all_timeframes(api_key: str):
    """
    Returns (timeframes_dict, sources_dict).
    sources_dict has a key for each timeframe + 'dxy', value is the source label.
    """
    timeframes = {}
    sources = {}

    for tf in ["1min", "5min", "15min"]:
        df, src = fetch_gold_data(api_key, tf)
        timeframes[tf] = df
        sources[tf] = src
        if "Alpha Vantage" in src:
            time.sleep(13)
        else:
            time.sleep(0.3)

    dxy_df, dxy_src = fetch_dxy_data(api_key, "5min")
    timeframes["dxy"] = dxy_df
    sources["dxy"] = dxy_src

    return timeframes, sources


def get_current_price(api_key: str) -> dict:
    """Best available current gold price."""
    is_demo = api_key.lower().strip() in ("demo", "", "your_api_key_here")

    # Alpha Vantage spot
    if not is_demo:
        p = _fetch_av_spot(api_key)
        if p:
            return {"price": p, "bid": round(p - 0.3, 2),
                    "ask": round(p + 0.3, 2), "source": "Alpha Vantage",
                    "timestamp": datetime.now().isoformat()}

    # Yahoo Finance
    try:
        import yfinance as yf
        h = yf.Ticker("GC=F").history(period="1d", interval="1m")
        if not h.empty:
            p = round(float(h["Close"].iloc[-1]), 2)
            return {"price": p, "bid": round(p - 0.3, 2),
                    "ask": round(p + 0.3, 2), "source": "Yahoo Finance",
                    "timestamp": datetime.now().isoformat()}
    except Exception:
        pass

    # Fallback
    p = round(GOLD_BASE_PRICE + np.random.normal(0, 3), 2)
    return {"price": p, "bid": round(p - 0.3, 2),
            "ask": round(p + 0.3, 2), "source": "Synthetic",
            "timestamp": datetime.now().isoformat()}
