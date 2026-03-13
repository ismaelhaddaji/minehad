"""
liquidity_analysis.py
Advanced liquidity sweep detection and order flow analysis.
"""

import pandas as pd
import numpy as np


def detect_equal_highs_lows(df: pd.DataFrame, tolerance: float = 0.001, lookback: int = 50) -> dict:
    """
    Detect equal highs and equal lows (liquidity pools).
    These represent stop-loss clusters that are targets for sweeps.
    """
    recent = df.iloc[-lookback:]
    highs = recent['high'].values
    lows = recent['low'].values
    
    equal_highs = []
    equal_lows = []
    
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / highs[i] < tolerance:
                equal_highs.append(round((highs[i] + highs[j]) / 2, 2))
            if abs(lows[i] - lows[j]) / lows[i] < tolerance:
                equal_lows.append(round((lows[i] + lows[j]) / 2, 2))
    
    # Remove duplicates
    equal_highs = sorted(set([round(h, 1) for h in equal_highs]))[-5:]
    equal_lows = sorted(set([round(l, 1) for l in equal_lows]))[:5]
    
    return {
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
        "liquidity_above": equal_highs,
        "liquidity_below": equal_lows,
    }


def analyze_volume_profile(df: pd.DataFrame, n_bins: int = 20) -> dict:
    """
    Simplified volume profile analysis.
    Identifies high-volume price levels (point of control).
    """
    if len(df) < 10:
        return {"poc": df['close'].mean() if len(df) > 0 else 0, "value_area_high": 0, "value_area_low": 0}
    
    price_min = df['low'].min()
    price_max = df['high'].max()
    
    if price_max <= price_min:
        return {"poc": price_min, "value_area_high": price_max, "value_area_low": price_min}
    
    bins = np.linspace(price_min, price_max, n_bins + 1)
    bin_volumes = np.zeros(n_bins)
    
    for _, row in df.iterrows():
        # Distribute bar volume across its range
        bar_bins = np.where((bins[:-1] <= row['close']) & (bins[1:] >= row['close']))[0]
        if len(bar_bins) > 0:
            bin_volumes[bar_bins[0]] += row.get('volume', 1)
    
    poc_idx = np.argmax(bin_volumes)
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
    
    # Value area = 70% of volume
    total_vol = bin_volumes.sum()
    target = total_vol * 0.70
    
    sorted_idx = np.argsort(bin_volumes)[::-1]
    cumvol = 0
    va_indices = []
    for idx in sorted_idx:
        cumvol += bin_volumes[idx]
        va_indices.append(idx)
        if cumvol >= target:
            break
    
    va_high = (bins[max(va_indices)] + bins[max(va_indices) + 1]) / 2
    va_low = (bins[min(va_indices)] + bins[min(va_indices) + 1]) / 2
    
    return {
        "poc": round(poc, 2),
        "value_area_high": round(va_high, 2),
        "value_area_low": round(va_low, 2),
        "bin_prices": [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)],
        "bin_volumes": bin_volumes.tolist(),
    }


def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Detect order blocks (institutional entry zones).
    Bullish order block: last bearish candle before bullish impulse.
    Bearish order block: last bullish candle before bearish impulse.
    """
    recent = df.iloc[-lookback:]
    order_blocks = {"bullish": [], "bearish": []}
    
    closes = recent['close'].values
    opens = recent['open'].values
    highs = recent['high'].values
    lows = recent['low'].values
    
    for i in range(1, len(closes) - 3):
        # Look for strong move after candle
        is_bearish_candle = closes[i] < opens[i]
        is_bullish_candle = closes[i] > opens[i]
        
        # Next 3 candles move
        future_move = (closes[i+1] + closes[i+2] + closes[i+3]) / 3 - closes[i]
        
        if is_bearish_candle and future_move > closes[i] * 0.002:
            order_blocks["bullish"].append({
                "high": round(highs[i], 2),
                "low": round(lows[i], 2),
                "mid": round((highs[i] + lows[i]) / 2, 2),
            })
        
        if is_bullish_candle and future_move < -closes[i] * 0.002:
            order_blocks["bearish"].append({
                "high": round(highs[i], 2),
                "low": round(lows[i], 2),
                "mid": round((highs[i] + lows[i]) / 2, 2),
            })
    
    # Keep most recent 3 of each
    order_blocks["bullish"] = order_blocks["bullish"][-3:]
    order_blocks["bearish"] = order_blocks["bearish"][-3:]
    
    return order_blocks


def get_liquidity_analysis(df: pd.DataFrame) -> dict:
    """Full liquidity analysis."""
    eq_levels = detect_equal_highs_lows(df)
    vol_profile = analyze_volume_profile(df)
    order_blocks = detect_order_blocks(df)
    
    current_price = df['close'].iloc[-1] if len(df) > 0 else 0
    
    # Nearest liquidity targets
    liq_above = [l for l in eq_levels.get('liquidity_above', []) if l > current_price]
    liq_below = [l for l in eq_levels.get('liquidity_below', []) if l < current_price]
    
    return {
        "equal_highs": eq_levels.get('equal_highs', []),
        "equal_lows": eq_levels.get('equal_lows', []),
        "nearest_liquidity_above": min(liq_above) if liq_above else current_price * 1.01,
        "nearest_liquidity_below": max(liq_below) if liq_below else current_price * 0.99,
        "volume_profile": vol_profile,
        "order_blocks": order_blocks,
        "poc": vol_profile.get("poc", current_price),
        "value_area_high": vol_profile.get("value_area_high", current_price * 1.005),
        "value_area_low": vol_profile.get("value_area_low", current_price * 0.995),
    }
