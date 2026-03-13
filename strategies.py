"""
strategies.py
Implements multiple trading strategies for GOLD day trading.
"""

import pandas as pd
import numpy as np
from feature_engineering import get_session_info


def strategy_london_kill_zone_breakout(features: pd.DataFrame, asian_range: dict) -> dict:
    """
    London Kill Zone Breakout Strategy.
    Break of Asian session range during London Kill Zone (07:00-10:00).
    """
    result = {"name": "London Kill Zone Breakout", "signal": 0, "score": 0, "reason": []}
    
    if not features.index[-1:].shape[0]:
        return result
    
    latest = features.iloc[-1]
    asian_high = asian_range.get("asian_high")
    asian_low = asian_range.get("asian_low")
    
    if asian_high is None or asian_low is None:
        result["reason"].append("No Asian range available")
        return result
    
    # Must be in London Kill Zone
    hour = latest.get('hour', 12)
    if not (7 <= hour < 10):
        result["reason"].append("Not in London Kill Zone")
        return result
    
    price = latest['close']
    asian_range_size = asian_high - asian_low
    
    # Bullish breakout: close above Asian high
    if price > asian_high and latest.get('volume_spike', 0):
        if latest.get('above_vwap', 0) and latest.get('macd_bullish', 0):
            result["signal"] = 1
            result["score"] = 3
            result["reason"].append(f"Bullish breakout above Asian high {asian_high:.2f}")
            result["reason"].append("Volume confirmed + VWAP aligned")
            result["entry"] = price
            result["stop_loss"] = asian_high - latest.get('atr', asian_range_size * 0.5)
            result["take_profit"] = price + (price - result["stop_loss"]) * 2.5
            
    # Bearish breakout: close below Asian low
    elif price < asian_low and latest.get('volume_spike', 0):
        if not latest.get('above_vwap', 1) and not latest.get('macd_bullish', 0):
            result["signal"] = -1
            result["score"] = 3
            result["reason"].append(f"Bearish breakout below Asian low {asian_low:.2f}")
            result["reason"].append("Volume confirmed + VWAP aligned")
            result["entry"] = price
            result["stop_loss"] = asian_low + latest.get('atr', asian_range_size * 0.5)
            result["take_profit"] = price - (result["stop_loss"] - price) * 2.5
    
    return result


def strategy_ny_reversal(features: pd.DataFrame, london_range: dict) -> dict:
    """
    New York Reversal Strategy.
    Liquidity sweep of London range during NY session reversal.
    """
    result = {"name": "New York Reversal", "signal": 0, "score": 0, "reason": []}
    
    latest = features.iloc[-1]
    hour = latest.get('hour', 12)
    
    # Must be in NY session / NY kill zone
    if not (13 <= hour < 18):
        result["reason"].append("Not in NY session")
        return result
    
    london_high = london_range.get("london_high")
    london_low = london_range.get("london_low")
    
    if london_high is None:
        # Use session high/low as proxy
        london_high = latest.get('session_high', latest['close'] * 1.005)
        london_low = latest.get('session_low', latest['close'] * 0.995)
    
    price = latest['close']
    sweep = latest.get('liquidity_sweep', 0)
    
    # Bullish reversal after sweep below London low
    if sweep == -1 and price > london_low:
        if latest.get('rsi', 50) < 40 or latest.get('rsi_divergence', 0) == 1:
            result["signal"] = 1
            result["score"] = 2
            result["reason"].append(f"Bullish reversal after sweep below London low {london_low:.2f}")
            result["entry"] = price
            atr = latest.get('atr', price * 0.002)
            result["stop_loss"] = price - atr * 1.5
            result["take_profit"] = price + atr * 3
            
    # Bearish reversal after sweep above London high
    elif sweep == 1 and price < london_high:
        if latest.get('rsi', 50) > 60 or latest.get('rsi_divergence', 0) == -1:
            result["signal"] = -1
            result["score"] = 2
            result["reason"].append(f"Bearish reversal after sweep above London high {london_high:.2f}")
            result["entry"] = price
            atr = latest.get('atr', price * 0.002)
            result["stop_loss"] = price + atr * 1.5
            result["take_profit"] = price - atr * 3
    
    return result


def strategy_vwap_liquidity_sweep_reversal(features: pd.DataFrame) -> dict:
    """
    Prop Trader High Win Rate Strategy: VWAP Liquidity Sweep Reversal.
    1. Identify liquidity sweep of recent high/low.
    2. Price returns inside range quickly.
    3. Confirm rejection near VWAP.
    4. Enter in opposite direction.
    """
    result = {"name": "VWAP Liquidity Sweep Reversal", "signal": 0, "score": 0, "reason": []}
    
    latest = features.iloc[-1]
    sweep = latest.get('liquidity_sweep', 0)
    price = latest['close']
    vwap = latest.get('vwap', price)
    
    if abs(sweep) == 0:
        return result
    
    atr = latest.get('atr', price * 0.002)
    dist_from_vwap = abs(price - vwap)
    vwap_proximity = dist_from_vwap < atr * 1.5  # Near VWAP
    
    # Bearish sweep reversal (sweep above → short)
    if sweep == 1:
        score = 2
        if latest.get('rsi_divergence', 0) == -1:
            score += 1
        if latest.get('volume_spike', 0):
            score += 1
        if vwap_proximity:
            score += 1
        if latest.get('rsi', 50) > 65:
            score += 1
            
        if score >= 3:
            result["signal"] = -1
            result["score"] = score
            result["reason"].append("Sweep above resistance + quick reversal")
            if latest.get('rsi_divergence', 0) == -1:
                result["reason"].append("RSI bearish divergence confirmed")
            if latest.get('volume_spike', 0):
                result["reason"].append("Volume spike on sweep")
            if vwap_proximity:
                result["reason"].append("VWAP rejection confirmed")
            result["entry"] = price
            result["stop_loss"] = price + atr * 1.5
            result["take_profit"] = price - atr * 3
    
    # Bullish sweep reversal (sweep below → long)
    elif sweep == -1:
        score = 2
        if latest.get('rsi_divergence', 0) == 1:
            score += 1
        if latest.get('volume_spike', 0):
            score += 1
        if vwap_proximity:
            score += 1
        if latest.get('rsi', 50) < 35:
            score += 1
            
        if score >= 3:
            result["signal"] = 1
            result["score"] = score
            result["reason"].append("Sweep below support + quick reversal")
            if latest.get('rsi_divergence', 0) == 1:
                result["reason"].append("RSI bullish divergence confirmed")
            if latest.get('volume_spike', 0):
                result["reason"].append("Volume spike on sweep")
            if vwap_proximity:
                result["reason"].append("VWAP bounce confirmed")
            result["entry"] = price
            result["stop_loss"] = price - atr * 1.5
            result["take_profit"] = price + atr * 3
    
    return result


def strategy_ema_trend_following(features: pd.DataFrame) -> dict:
    """
    EMA Trend Following Strategy.
    Full EMA alignment with pullback entry.
    """
    result = {"name": "EMA Trend Following", "signal": 0, "score": 0, "reason": []}
    
    latest = features.iloc[-1]
    price = latest['close']
    atr = latest.get('atr', price * 0.002)
    
    ema9 = latest.get('ema_9', price)
    ema21 = latest.get('ema_21', price)
    ema50 = latest.get('ema_50', price)
    ema200 = latest.get('ema_200', price)
    
    # Strong uptrend: 9 > 21 > 50 > 200
    if ema9 > ema21 > ema50 > ema200:
        score = 2
        # Pullback to EMA21 entry
        dist_to_ema21 = abs(price - ema21) / price * 100
        if dist_to_ema21 < 0.3:  # Price near EMA21
            score += 1
            result["reason"].append(f"Price pulled back to EMA21 ({ema21:.2f})")
        if latest.get('macd_bullish', 0):
            score += 1
        if latest.get('above_vwap', 0):
            score += 1
        if latest.get('trending', 0) and latest.get('adx', 0) > 30:
            score += 1
        
        result["signal"] = 1
        result["score"] = score
        result["reason"].insert(0, "Strong uptrend: EMA9 > EMA21 > EMA50 > EMA200")
        result["entry"] = price
        result["stop_loss"] = ema21 - atr
        result["take_profit"] = price + (price - result["stop_loss"]) * 2.5
    
    # Strong downtrend: 9 < 21 < 50 < 200
    elif ema9 < ema21 < ema50 < ema200:
        score = 2
        dist_to_ema21 = abs(price - ema21) / price * 100
        if dist_to_ema21 < 0.3:
            score += 1
            result["reason"].append(f"Price pulled back to EMA21 ({ema21:.2f})")
        if not latest.get('macd_bullish', 1):
            score += 1
        if not latest.get('above_vwap', 0):
            score += 1
        if latest.get('trending', 0) and latest.get('adx', 0) > 30:
            score += 1
        
        result["signal"] = -1
        result["score"] = score
        result["reason"].insert(0, "Strong downtrend: EMA9 < EMA21 < EMA50 < EMA200")
        result["entry"] = price
        result["stop_loss"] = ema21 + atr
        result["take_profit"] = price - (result["stop_loss"] - price) * 2.5
    
    return result


def strategy_rsi_momentum(features: pd.DataFrame) -> dict:
    """RSI Momentum + Divergence Strategy."""
    result = {"name": "RSI Momentum", "signal": 0, "score": 0, "reason": []}
    
    latest = features.iloc[-1]
    price = latest['close']
    rsi = latest.get('rsi', 50)
    atr = latest.get('atr', price * 0.002)
    
    score = 0
    
    # Oversold bounce
    if rsi < 30:
        score = 2
        result["reason"].append(f"RSI oversold at {rsi:.1f}")
        if latest.get('rsi_divergence', 0) == 1:
            score += 2
            result["reason"].append("Bullish RSI divergence")
        if latest.get('macd_bullish', 0):
            score += 1
        if result["reason"]:
            result["signal"] = 1
            result["score"] = score
            result["entry"] = price
            result["stop_loss"] = price - atr * 2
            result["take_profit"] = price + atr * 4
    
    # Overbought rejection
    elif rsi > 70:
        score = 2
        result["reason"].append(f"RSI overbought at {rsi:.1f}")
        if latest.get('rsi_divergence', 0) == -1:
            score += 2
            result["reason"].append("Bearish RSI divergence")
        if not latest.get('macd_bullish', 1):
            score += 1
        if result["reason"]:
            result["signal"] = -1
            result["score"] = score
            result["entry"] = price
            result["stop_loss"] = price + atr * 2
            result["take_profit"] = price - atr * 4
    
    return result


def strategy_macd_crossover(features: pd.DataFrame) -> dict:
    """MACD Crossover Strategy."""
    result = {"name": "MACD Crossover", "signal": 0, "score": 0, "reason": []}
    
    if len(features) < 3:
        return result
    
    latest = features.iloc[-1]
    prev = features.iloc[-2]
    price = latest['close']
    atr = latest.get('atr', price * 0.002)
    
    macd = latest.get('macd', 0)
    macd_sig = latest.get('macd_signal', 0)
    prev_macd = prev.get('macd', 0)
    prev_sig = prev.get('macd_signal', 0)
    
    # Bullish crossover
    if prev_macd < prev_sig and macd > macd_sig:
        score = 2
        result["reason"].append("MACD bullish crossover")
        if macd < 0:  # Still negative = confirmed reversal
            score += 1
        if latest.get('volume_spike', 0):
            score += 1
        result["signal"] = 1
        result["score"] = score
        result["entry"] = price
        result["stop_loss"] = price - atr * 2
        result["take_profit"] = price + atr * 3
    
    # Bearish crossover
    elif prev_macd > prev_sig and macd < macd_sig:
        score = 2
        result["reason"].append("MACD bearish crossover")
        if macd > 0:
            score += 1
        if latest.get('volume_spike', 0):
            score += 1
        result["signal"] = -1
        result["score"] = score
        result["entry"] = price
        result["stop_loss"] = price + atr * 2
        result["take_profit"] = price - atr * 3
    
    return result


def strategy_bollinger_breakout(features: pd.DataFrame) -> dict:
    """Bollinger Band Squeeze Breakout Strategy."""
    result = {"name": "Bollinger Band Breakout", "signal": 0, "score": 0, "reason": []}
    
    latest = features.iloc[-1]
    price = latest['close']
    atr = latest.get('atr', price * 0.002)
    
    bb_pct = latest.get('bb_pct', 0.5)
    bb_width = latest.get('bb_width', 0.02)
    squeeze = latest.get('bb_squeeze', 0)
    
    # Check for recent squeeze followed by expansion
    if squeeze and latest.get('vol_expansion', 0):
        if price > latest.get('bb_upper', price * 1.002):
            result["signal"] = 1
            result["score"] = 3
            result["reason"].append("BB squeeze breakout - bullish expansion")
            result["entry"] = price
            result["stop_loss"] = latest.get('bb_mid', price) - atr
            result["take_profit"] = price + (price - result["stop_loss"]) * 2.5
        elif price < latest.get('bb_lower', price * 0.998):
            result["signal"] = -1
            result["score"] = 3
            result["reason"].append("BB squeeze breakout - bearish expansion")
            result["entry"] = price
            result["stop_loss"] = latest.get('bb_mid', price) + atr
            result["take_profit"] = price - (result["stop_loss"] - price) * 2.5
    
    return result


def strategy_support_resistance_breakout(features: pd.DataFrame, supports: list, resistances: list) -> dict:
    """Support/Resistance Breakout Strategy."""
    result = {"name": "S/R Breakout", "signal": 0, "score": 0, "reason": []}
    
    latest = features.iloc[-1]
    price = latest['close']
    atr = latest.get('atr', price * 0.002)
    
    if not supports or not resistances:
        return result
    
    nearest_resistance = min([r for r in resistances if r > price * 0.999], default=None)
    nearest_support = max([s for s in supports if s < price * 1.001], default=None)
    
    # Resistance breakout
    if nearest_resistance and price > nearest_resistance and latest.get('volume_spike', 0):
        score = 2
        result["reason"].append(f"Resistance breakout at {nearest_resistance:.2f}")
        if latest.get('structure_break', 0) == 1:
            score += 1
        if latest.get('macd_bullish', 0):
            score += 1
        result["signal"] = 1
        result["score"] = score
        result["entry"] = price
        result["stop_loss"] = nearest_resistance - atr
        result["take_profit"] = price + (price - result["stop_loss"]) * 2.5
    
    # Support breakdown
    elif nearest_support and price < nearest_support and latest.get('volume_spike', 0):
        score = 2
        result["reason"].append(f"Support breakdown at {nearest_support:.2f}")
        if latest.get('structure_break', 0) == -1:
            score += 1
        if not latest.get('macd_bullish', 1):
            score += 1
        result["signal"] = -1
        result["score"] = score
        result["entry"] = price
        result["stop_loss"] = nearest_support + atr
        result["take_profit"] = price - (result["stop_loss"] - price) * 2.5
    
    return result


def run_all_strategies(features: pd.DataFrame, asian_range: dict, london_range: dict,
                       supports: list, resistances: list) -> list:
    """Run all strategies and return list of signals."""
    strategies = [
        strategy_london_kill_zone_breakout(features, asian_range),
        strategy_ny_reversal(features, london_range),
        strategy_vwap_liquidity_sweep_reversal(features),
        strategy_ema_trend_following(features),
        strategy_rsi_momentum(features),
        strategy_macd_crossover(features),
        strategy_bollinger_breakout(features),
        strategy_support_resistance_breakout(features, supports, resistances),
    ]
    
    # Filter for active signals only
    active = [s for s in strategies if s.get('signal', 0) != 0]
    return active, strategies  # active signals + all strategies
