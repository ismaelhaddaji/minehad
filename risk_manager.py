"""
risk_manager.py
Computes position sizing, risk metrics, and validates trade setups.
"""

import numpy as np
import pandas as pd


def compute_position_size(account_balance: float, risk_pct: float, entry: float,
                          stop_loss: float, contract_size: float = 100) -> dict:
    """
    Compute recommended position size based on account risk.
    
    Args:
        account_balance: Total account balance in USD
        risk_pct: Risk per trade as decimal (e.g., 0.01 = 1%)
        entry: Entry price
        stop_loss: Stop loss price
        contract_size: Contract size (oz per lot for gold)
    
    Returns:
        Position sizing details
    """
    risk_amount = account_balance * risk_pct
    price_risk = abs(entry - stop_loss)
    
    if price_risk <= 0:
        return {"error": "Invalid stop loss", "lots": 0, "units": 0}
    
    # For gold: 1 oz = $1 per pip ($1 move)
    # 1 standard lot of gold = 100 oz
    # Risk per lot = price_risk * contract_size
    risk_per_lot = price_risk * contract_size
    lots = risk_amount / risk_per_lot if risk_per_lot > 0 else 0
    lots = round(lots, 2)
    
    # Cap at reasonable maximum
    lots = min(lots, 10.0)
    
    units = lots * contract_size
    dollar_value = units * entry
    
    return {
        "account_balance": account_balance,
        "risk_pct": risk_pct * 100,
        "risk_amount": round(risk_amount, 2),
        "price_risk": round(price_risk, 2),
        "lots": lots,
        "units": round(units, 0),
        "dollar_value": round(dollar_value, 0),
        "max_loss": round(risk_amount, 2),
        "max_gain": round(risk_amount * (abs(entry - stop_loss) * 2.5 / price_risk), 2) if price_risk > 0 else 0,
    }


def validate_trade(entry: float, stop_loss: float, take_profit: float,
                   min_rr: float = 2.0) -> dict:
    """Validate trade setup and reject if risk profile is poor."""
    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        return {"valid": False, "reason": "Invalid price levels"}
    
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk <= 0:
        return {"valid": False, "reason": "Zero risk (stop = entry)"}
    
    rr = reward / risk
    
    if rr < min_rr:
        return {
            "valid": False,
            "reason": f"R:R {rr:.2f} below minimum {min_rr}",
            "rr": rr
        }
    
    return {
        "valid": True,
        "risk": round(risk, 2),
        "reward": round(reward, 2),
        "rr": round(rr, 2),
        "reason": f"Valid setup — R:R {rr:.2f}"
    }


def compute_atr_stop(features: pd.DataFrame, direction: int, atr_multiplier: float = 1.5) -> float:
    """Compute ATR-based stop loss."""
    latest = features.iloc[-1]
    price = latest['close']
    atr = latest.get('atr', price * 0.002)
    
    if direction == 1:  # Long
        return price - atr * atr_multiplier
    else:  # Short
        return price + atr * atr_multiplier


def compute_swing_stop(features: pd.DataFrame, direction: int, lookback: int = 10) -> float:
    """Compute stop based on recent swing high/low."""
    recent = features.iloc[-lookback:]
    price = features['close'].iloc[-1]
    
    if direction == 1:  # Long — stop below recent swing low
        swing_low = recent['low'].min()
        return swing_low * 0.999  # Tiny buffer
    else:  # Short — stop above recent swing high
        swing_high = recent['high'].max()
        return swing_high * 1.001
