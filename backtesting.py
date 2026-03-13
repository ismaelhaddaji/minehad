"""
backtesting.py
Event-driven backtesting engine for strategy evaluation.
"""

import pandas as pd
import numpy as np
from feature_engineering import compute_all_features, detect_support_resistance, detect_asian_session_range
from strategies import run_all_strategies
from ensemble_engine import compute_ensemble_score
from regime_detection import detect_regime


def run_backtest(df: pd.DataFrame, initial_capital: float = 10000,
                 risk_pct: float = 0.01, commission: float = 0.0,
                 min_score: int = 9) -> dict:
    """
    Simplified event-driven backtest.
    
    Returns performance metrics and equity curve.
    """
    if len(df) < 100:
        return _empty_backtest_result()
    
    features = compute_all_features(df)
    
    capital = initial_capital
    equity_curve = [capital]
    trades = []
    
    in_trade = False
    trade_entry = 0
    trade_stop = 0
    trade_target = 0
    trade_direction = 0
    trade_open_idx = 0
    
    window = 50  # Rolling window for analysis
    
    for i in range(window, len(features) - 1):
        subset = features.iloc[:i+1]
        latest = subset.iloc[-1]
        price = latest['close']
        
        # Check if in trade — exit conditions
        if in_trade:
            high = latest.get('high', price)
            low = latest.get('low', price)
            
            hit_stop = (trade_direction == 1 and low <= trade_stop) or \
                       (trade_direction == -1 and high >= trade_stop)
            hit_target = (trade_direction == 1 and high >= trade_target) or \
                         (trade_direction == -1 and low <= trade_target)
            
            if hit_target or hit_stop:
                if hit_target:
                    pnl_pct = abs(trade_target - trade_entry) / trade_entry
                    result = "WIN"
                else:
                    pnl_pct = -abs(trade_stop - trade_entry) / trade_entry
                    result = "LOSS"
                
                trade_capital_used = capital * risk_pct
                actual_pnl = trade_capital_used * (pnl_pct / abs(trade_stop - trade_entry) * trade_entry) if abs(trade_stop - trade_entry) > 0 else 0
                
                # Simplified PnL
                if result == "WIN":
                    actual_pnl = capital * risk_pct * 2.5  # 2.5x R
                else:
                    actual_pnl = -capital * risk_pct
                
                capital += actual_pnl
                trades.append({
                    "idx": i,
                    "direction": trade_direction,
                    "entry": trade_entry,
                    "exit": trade_target if hit_target else trade_stop,
                    "result": result,
                    "pnl": round(actual_pnl, 2),
                    "capital": round(capital, 2),
                })
                in_trade = False
                equity_curve.append(capital)
            else:
                equity_curve.append(capital)
            continue
        
        # Look for new signal (every N bars to reduce computation)
        if i % 5 != 0:
            equity_curve.append(capital)
            continue
        
        try:
            asian_range = detect_asian_session_range(subset)
            london_range = {"london_high": latest.get('session_high'), "london_low": latest.get('session_low')}
            supports, resistances = detect_support_resistance(subset)
            regime_info = detect_regime(subset)
            active_strategies, all_strategies = run_all_strategies(subset, asian_range, london_range, supports, resistances)
            
            ensemble = compute_ensemble_score(subset, active_strategies, regime_info)
            
            if ensemble.get('signal', 0) != 0 and ensemble.get('score', 0) >= min_score:
                in_trade = True
                trade_direction = ensemble['signal']
                trade_entry = ensemble.get('entry', price)
                trade_stop = ensemble.get('stop_loss', price * (0.998 if trade_direction == 1 else 1.002))
                trade_target = ensemble.get('take_profit', price * (1.005 if trade_direction == 1 else 0.995))
                trade_open_idx = i
        except:
            pass
        
        equity_curve.append(capital)
    
    return _compute_metrics(trades, equity_curve, initial_capital)


def _compute_metrics(trades: list, equity_curve: list, initial_capital: float) -> dict:
    """Compute backtest performance metrics."""
    if not trades:
        return _empty_backtest_result()
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    
    win_rate = len(wins) / len(trades) if trades else 0
    
    total_profit = sum(t['pnl'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    total_return = (equity_curve[-1] - initial_capital) / initial_capital if initial_capital > 0 else 0
    
    # Sharpe ratio (simplified, annualized)
    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 78)) if returns.std() > 0 else 0  # 78 5min bars/day
    
    # Max drawdown
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()
    
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
    
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "total_return": round(total_return * 100, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "equity_curve": equity_curve,
        "trades": trades,
        "final_capital": round(equity_curve[-1] if equity_curve else initial_capital, 2),
    }


def _empty_backtest_result() -> dict:
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "total_return": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "expectancy": 0,
        "equity_curve": [10000],
        "trades": [],
        "final_capital": 10000,
    }
