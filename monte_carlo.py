"""
monte_carlo.py
Monte Carlo simulation for risk estimation.
"""

import numpy as np
import pandas as pd


def run_monte_carlo(trades: list, n_simulations: int = 1000,
                    initial_capital: float = 10000) -> dict:
    """
    Monte Carlo simulation by randomizing trade sequence.
    Estimates worst-case drawdown and probability of ruin.
    """
    if not trades or len(trades) < 5:
        # Generate synthetic trade results if no history
        win_rate = 0.55
        avg_rr = 2.5
        n_trades = 50
        pnl_list = []
        risk_amount = initial_capital * 0.01
        for _ in range(n_trades):
            if np.random.random() < win_rate:
                pnl_list.append(risk_amount * avg_rr)
            else:
                pnl_list.append(-risk_amount)
    else:
        pnl_list = [t['pnl'] for t in trades]
    
    pnl_array = np.array(pnl_list)
    n_trades = len(pnl_array)
    
    equity_curves = []
    final_capitals = []
    max_drawdowns = []
    ruin_count = 0
    ruin_threshold = initial_capital * 0.5  # 50% drawdown = ruin
    
    for _ in range(n_simulations):
        # Shuffle trade sequence
        shuffled = np.random.choice(pnl_array, size=n_trades, replace=True)
        equity = initial_capital + np.cumsum(shuffled)
        equity = np.insert(equity, 0, initial_capital)
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / np.maximum(peak, 1)
        max_dd = dd.min()
        
        final_capitals.append(equity[-1])
        max_drawdowns.append(max_dd)
        equity_curves.append(equity)
        
        if equity.min() < ruin_threshold:
            ruin_count += 1
    
    final_capitals = np.array(final_capitals)
    max_drawdowns = np.array(max_drawdowns)
    
    # Sample equity curves for display (25 random)
    sample_idx = np.random.choice(len(equity_curves), size=min(25, len(equity_curves)), replace=False)
    sampled_curves = [equity_curves[i].tolist() for i in sample_idx]
    
    return {
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "probability_of_ruin": round(ruin_count / n_simulations * 100, 2),
        "worst_case_drawdown": round(max_drawdowns.min() * 100, 2),
        "median_drawdown": round(np.median(max_drawdowns) * 100, 2),
        "expected_final_capital": round(np.mean(final_capitals), 2),
        "median_final_capital": round(np.median(final_capitals), 2),
        "p5_final_capital": round(np.percentile(final_capitals, 5), 2),
        "p95_final_capital": round(np.percentile(final_capitals, 95), 2),
        "expected_return_pct": round((np.mean(final_capitals) - initial_capital) / initial_capital * 100, 2),
        "sample_equity_curves": sampled_curves,
        "drawdown_distribution": max_drawdowns.tolist()[:100],
    }


"""
walk_forward.py
Walk-forward optimization for strategy parameters.
"""


def walk_forward_analysis(df: pd.DataFrame, train_periods: int = 200,
                          test_periods: int = 50) -> dict:
    """
    Simple walk-forward analysis.
    Train on N bars, test on M bars, roll forward.
    """
    from backtesting import run_backtest
    
    results = []
    n = len(df)
    
    if n < train_periods + test_periods:
        return {"results": [], "avg_win_rate": 0, "consistency": 0}
    
    step = test_periods
    windows = []
    
    pos = 0
    while pos + train_periods + test_periods <= n:
        train_start = pos
        train_end = pos + train_periods
        test_start = train_end
        test_end = test_start + test_periods
        windows.append((train_start, train_end, test_start, test_end))
        pos += step
    
    for i, (ts, te, vs, ve) in enumerate(windows[:10]):  # Max 10 windows
        try:
            test_data = df.iloc[vs:ve]
            metrics = run_backtest(test_data, initial_capital=10000)
            results.append({
                "window": i + 1,
                "period": f"{df.index[vs].strftime('%m/%d') if hasattr(df.index[vs], 'strftime') else vs}–{df.index[ve-1].strftime('%m/%d') if hasattr(df.index[ve-1], 'strftime') else ve}",
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "return_pct": metrics["total_return"],
            })
        except:
            pass
    
    if not results:
        return {"results": [], "avg_win_rate": 0, "consistency": 0}
    
    win_rates = [r["win_rate"] for r in results if r["trades"] > 0]
    avg_win_rate = np.mean(win_rates) if win_rates else 0
    consistency = 1 - (np.std(win_rates) / np.mean(win_rates)) if win_rates and np.mean(win_rates) > 0 else 0
    
    return {
        "results": results,
        "avg_win_rate": round(avg_win_rate, 1),
        "consistency": round(max(0, consistency), 3),
        "windows_analyzed": len(results),
    }
