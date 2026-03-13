"""
ensemble_engine.py
Combines strategy signals with scoring system to generate high-probability trade setups.
"""

import pandas as pd
import numpy as np


SCORE_WEIGHTS = {
    "trend_confirmation": 2,
    "momentum_confirmation": 2,
    "volatility_breakout": 2,
    "structure_breakout": 2,
    "volume_confirmation": 2,
    "london_kill_zone": 2,
    "liquidity_sweep": 2,
    "usd_correlation": 1,
}

MAX_SCORE = sum(SCORE_WEIGHTS.values())  # 15
MIN_SIGNAL_SCORE = 9


def compute_ensemble_score(features: pd.DataFrame, active_strategies: list,
                            regime_info: dict, ml_probability: float = 0.5) -> dict:
    """
    Compute ensemble score combining all signals.
    Returns trade recommendation if score >= MIN_SIGNAL_SCORE.
    """
    latest = features.iloc[-1]
    price = latest['close']
    
    score = 0
    score_breakdown = {}
    signal_directions = []
    
    # Collect strategy signals
    for strat in active_strategies:
        if strat.get('signal', 0) != 0:
            signal_directions.append(strat['signal'])
            score += min(strat.get('score', 1), 3)
    
    # Determine dominant direction
    if not signal_directions:
        return {
            "signal": 0,
            "direction": "",
            "score": 0,
            "max_score": MAX_SCORE,
            "score_pct": 0.0,
            "grade": "F",
            "score_breakdown": {},
            "triggered_strategies": [],
            "reasons": [],
            "recommendation": "No strategy triggered — waiting for setup",
            "ml_probability": ml_probability,
            "regime": regime_info.get("regime", "Unknown"),
        }
    
    bull_count = sum(1 for s in signal_directions if s > 0)
    bear_count = sum(1 for s in signal_directions if s < 0)
    dominant_signal = 1 if bull_count > bear_count else -1
    
    # Score individual components
    # Trend confirmation
    ema_bullish = latest.get('ema_bullish', 0)
    ema_bearish = latest.get('ema_bearish', 0)
    if (dominant_signal == 1 and ema_bullish) or (dominant_signal == -1 and ema_bearish):
        score += SCORE_WEIGHTS["trend_confirmation"]
        score_breakdown["Trend"] = f"+{SCORE_WEIGHTS['trend_confirmation']}"
    
    # Momentum confirmation
    rsi = latest.get('rsi', 50)
    macd_bullish = latest.get('macd_bullish', 0)
    momentum_aligned = (dominant_signal == 1 and macd_bullish and rsi > 45) or \
                       (dominant_signal == -1 and not macd_bullish and rsi < 55)
    if momentum_aligned:
        score += SCORE_WEIGHTS["momentum_confirmation"]
        score_breakdown["Momentum"] = f"+{SCORE_WEIGHTS['momentum_confirmation']}"
    
    # Volatility breakout
    vol_expansion = latest.get('vol_expansion', 0)
    bb_squeeze = latest.get('bb_squeeze', 0)
    if vol_expansion or bb_squeeze:
        score += SCORE_WEIGHTS["volatility_breakout"]
        score_breakdown["Volatility"] = f"+{SCORE_WEIGHTS['volatility_breakout']}"
    
    # Structure break
    structure = latest.get('structure_break', 0)
    if (dominant_signal == 1 and structure == 1) or (dominant_signal == -1 and structure == -1):
        score += SCORE_WEIGHTS["structure_breakout"]
        score_breakdown["Structure"] = f"+{SCORE_WEIGHTS['structure_breakout']}"
    
    # Volume confirmation
    if latest.get('volume_spike', 0):
        score += SCORE_WEIGHTS["volume_confirmation"]
        score_breakdown["Volume"] = f"+{SCORE_WEIGHTS['volume_confirmation']}"
    
    # London/NY Kill Zone
    hour = latest.get('hour', 12)
    in_kill_zone = (7 <= hour < 10) or (13 <= hour < 16)
    if in_kill_zone:
        score += SCORE_WEIGHTS["london_kill_zone"]
        score_breakdown["Kill Zone"] = f"+{SCORE_WEIGHTS['london_kill_zone']}"
    
    # Liquidity sweep
    sweep = latest.get('liquidity_sweep', 0)
    if abs(sweep) > 0:
        score += SCORE_WEIGHTS["liquidity_sweep"]
        score_breakdown["Liquidity Sweep"] = f"+{SCORE_WEIGHTS['liquidity_sweep']}"
    
    # USD correlation
    dxy_trend = latest.get('dxy_trend', 0)
    usd_confirms = (dominant_signal == 1 and dxy_trend == 0) or \
                   (dominant_signal == -1 and dxy_trend == 1)
    if usd_confirms:
        score += SCORE_WEIGHTS["usd_correlation"]
        score_breakdown["DXY Macro"] = f"+{SCORE_WEIGHTS['usd_correlation']}"
    
    # Cap score at MAX
    score = min(score, MAX_SCORE)
    
    # Grade
    score_pct = score / MAX_SCORE
    if score_pct >= 0.87:
        grade = "A+"
    elif score_pct >= 0.73:
        grade = "A"
    elif score_pct >= 0.6:
        grade = "B+"
    elif score_pct >= 0.47:
        grade = "B"
    else:
        grade = "C"
    
    # Only recommend if score meets threshold
    if score < MIN_SIGNAL_SCORE:
        return {
            "signal": 0,
            "direction": "",
            "dominant_signal": dominant_signal,
            "score": score,
            "max_score": MAX_SCORE,
            "score_pct": round(score_pct, 3),
            "grade": grade,
            "score_breakdown": score_breakdown,
            "triggered_strategies": [],
            "reasons": [],
            "recommendation": f"Score {score}/{MAX_SCORE} ({grade}) — threshold is {MIN_SIGNAL_SCORE}",
            "ml_probability": ml_probability,
            "regime": regime_info.get("regime", "Unknown"),
        }
    
    # Aggregate trade levels from active strategies
    atr = latest.get('atr', price * 0.002)
    relevant_strategies = [s for s in active_strategies if s.get('signal') == dominant_signal]
    
    entries = [s.get('entry', price) for s in relevant_strategies if 'entry' in s]
    stops = [s.get('stop_loss') for s in relevant_strategies if 'stop_loss' in s]
    targets = [s.get('take_profit') for s in relevant_strategies if 'take_profit' in s]
    
    entry = np.mean(entries) if entries else price
    
    if dominant_signal == 1:
        stop_loss = min(stops) if stops else entry - atr * 2
        take_profit = np.mean(targets) if targets else entry + atr * 4
    else:
        stop_loss = max(stops) if stops else entry + atr * 2
        take_profit = np.mean(targets) if targets else entry - atr * 4
    
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    rr_ratio = reward / risk if risk > 0 else 0
    
    # Ensure minimum 2:1 RR
    if rr_ratio < 2.0:
        if dominant_signal == 1:
            take_profit = entry + risk * 2.5
        else:
            take_profit = entry - risk * 2.5
        rr_ratio = 2.5
    
    confidence_pct = round(score_pct * 100 * (0.7 + ml_probability * 0.3), 1)
    
    triggered_strategy_names = [s['name'] for s in active_strategies if s.get('signal') == dominant_signal]
    all_reasons = []
    for s in relevant_strategies:
        all_reasons.extend(s.get('reason', []))
    
    return {
        "signal": dominant_signal,
        "score": score,
        "max_score": MAX_SCORE,
        "score_pct": round(score_pct, 3),
        "grade": grade,
        "score_breakdown": score_breakdown,
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "risk_reward": round(rr_ratio, 2),
        "risk_pips": round(abs(entry - stop_loss), 2),
        "confidence_pct": confidence_pct,
        "ml_probability": ml_probability,
        "direction": "LONG" if dominant_signal == 1 else "SHORT",
        "triggered_strategies": triggered_strategy_names,
        "reasons": list(set(all_reasons)),
        "recommendation": f"{'🟢 LONG' if dominant_signal == 1 else '🔴 SHORT'} — Score {score}/{MAX_SCORE} ({grade})",
        "regime": regime_info.get("regime", "Unknown"),
    }
