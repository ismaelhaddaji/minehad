"""
regime_detection.py
Detects market regimes: Trending, Ranging, High Volatility, Compression.
"""

import pandas as pd
import numpy as np


REGIMES = {
    "TRENDING_UP": "Trending Up",
    "TRENDING_DOWN": "Trending Down",
    "RANGING": "Ranging",
    "HIGH_VOLATILITY": "High Volatility",
    "COMPRESSION": "Low Volatility Compression",
}


def detect_regime(features: pd.DataFrame) -> dict:
    """
    Detect current market regime using ADX, ATR, moving average slopes.
    Returns regime classification with confidence and strategy recommendation.
    """
    if len(features) < 20:
        return {
            "regime": REGIMES["RANGING"],
            "confidence": 0.5,
            "adx": 20,
            "atr_expansion": False,
            "strategy_focus": "mean_reversion",
            "description": "Insufficient data"
        }
    
    latest = features.iloc[-1]
    
    adx = latest.get('adx', 20)
    atr = latest.get('atr', 0)
    atr_mean = features['atr'].rolling(20).mean().iloc[-1] if 'atr' in features.columns else atr
    atr_expansion = atr > atr_mean * 1.3 if atr_mean > 0 else False
    
    ema9 = latest.get('ema_9', 0)
    ema21 = latest.get('ema_21', 0)
    ema50 = latest.get('ema_50', 0)
    ema200 = latest.get('ema_200', 0)
    
    # EMA slope
    ema21_slope = latest.get('ema_21_slope', 0)
    ema50_slope = latest.get('ema_50_slope', 0)
    
    # BB width for compression
    bb_width = latest.get('bb_width', 0.02)
    bb_width_mean = features['bb_width'].rolling(20).mean().iloc[-1] if 'bb_width' in features.columns else bb_width
    is_compressed = bb_width < bb_width_mean * 0.7 if bb_width_mean > 0 else False
    
    # Volatility
    atr_pct = latest.get('atr_pct', 0.5)
    is_high_vol = atr_pct > 0.8 or (atr_expansion and adx < 25)
    
    # Regime classification
    if is_compressed and not atr_expansion:
        regime = REGIMES["COMPRESSION"]
        strategy_focus = "volatility_expansion"
        confidence = 0.75 if bb_width < bb_width_mean * 0.6 else 0.6
        description = "Price coiling — breakout imminent"
    
    elif is_high_vol and adx < 25:
        regime = REGIMES["HIGH_VOLATILITY"]
        strategy_focus = "breakout"
        confidence = 0.7
        description = "High volatility with no clear direction — caution"
    
    elif adx > 25:
        # Trending regime
        if ema9 > ema21 > ema50 and ema21_slope > 0:
            regime = REGIMES["TRENDING_UP"]
            strategy_focus = "trend_following"
            strength = min(adx / 50, 1.0)
            confidence = 0.6 + strength * 0.3
            description = f"Strong uptrend — ADX {adx:.1f}"
        elif ema9 < ema21 < ema50 and ema21_slope < 0:
            regime = REGIMES["TRENDING_DOWN"]
            strategy_focus = "trend_following"
            strength = min(adx / 50, 1.0)
            confidence = 0.6 + strength * 0.3
            description = f"Strong downtrend — ADX {adx:.1f}"
        else:
            regime = REGIMES["RANGING"]
            strategy_focus = "mean_reversion"
            confidence = 0.55
            description = "Mixed trend signals"
    
    else:
        # Low ADX = ranging
        regime = REGIMES["RANGING"]
        strategy_focus = "mean_reversion"
        # Confidence based on how clearly ranging it is
        confidence = 0.6 if adx < 20 else 0.5
        description = f"Consolidation/ranging — ADX {adx:.1f}"
    
    # Recommended strategies based on regime
    strategy_map = {
        "trend_following": ["EMA Trend Following", "MACD Crossover", "London Kill Zone Breakout"],
        "mean_reversion": ["VWAP Liquidity Sweep Reversal", "RSI Momentum", "New York Reversal"],
        "breakout": ["Bollinger Band Breakout", "S/R Breakout", "London Kill Zone Breakout"],
        "volatility_expansion": ["Bollinger Band Breakout", "S/R Breakout"],
    }
    
    return {
        "regime": regime,
        "confidence": round(confidence, 2),
        "adx": round(adx, 1),
        "atr_expansion": atr_expansion,
        "is_high_vol": is_high_vol,
        "is_compressed": is_compressed,
        "strategy_focus": strategy_focus,
        "recommended_strategies": strategy_map.get(strategy_focus, []),
        "description": description,
        "ema_alignment": "bullish" if ema9 > ema21 > ema50 else ("bearish" if ema9 < ema21 < ema50 else "mixed"),
        "trend_strength": "strong" if adx > 30 else ("moderate" if adx > 20 else "weak"),
    }


def get_regime_color(regime: str) -> str:
    """Return color for regime display."""
    colors = {
        REGIMES["TRENDING_UP"]: "#00ff88",
        REGIMES["TRENDING_DOWN"]: "#ff4444",
        REGIMES["RANGING"]: "#ffaa00",
        REGIMES["HIGH_VOLATILITY"]: "#ff6600",
        REGIMES["COMPRESSION"]: "#8888ff",
    }
    return colors.get(regime, "#ffffff")


def get_regime_history(features: pd.DataFrame, window: int = 100) -> pd.Series:
    """Compute rolling regime classifications."""
    regimes = []
    for i in range(len(features)):
        if i < 20:
            regimes.append(REGIMES["RANGING"])
        else:
            subset = features.iloc[max(0, i-50):i+1]
            regime_info = detect_regime(subset)
            regimes.append(regime_info["regime"])
    return pd.Series(regimes, index=features.index)
