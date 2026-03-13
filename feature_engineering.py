"""
feature_engineering.py
Computes technical indicators and features for quantitative analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute session VWAP (resets each trading day)."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    # Group by date for daily reset
    if isinstance(df.index, pd.DatetimeIndex):
        date_groups = df.index.date
        vwap_values = pd.Series(index=df.index, dtype=float)
        for date in pd.unique(date_groups):
            mask = date_groups == date
            tp = typical_price[mask]
            vol = df['volume'][mask]
            cum_tp_vol = (tp * vol).cumsum()
            cum_vol = vol.cumsum()
            vwap_values[mask] = cum_tp_vol / cum_vol.replace(0, np.nan)
        return vwap_values
    else:
        tp_vol = (typical_price * df['volume']).cumsum()
        vol_cum = df['volume'].cumsum()
        return tp_vol / vol_cum.replace(0, np.nan)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0
    
    atr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx


def detect_volume_spikes(df: pd.DataFrame, multiplier: float = 2.0) -> pd.Series:
    avg_vol = df['volume'].rolling(20).mean()
    return (df['volume'] > avg_vol * multiplier).astype(int)


def detect_support_resistance(df: pd.DataFrame, lookback: int = 50, n_levels: int = 5) -> tuple:
    """Detect key support and resistance levels using pivot points."""
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    resistance_levels = []
    support_levels = []
    
    window = min(5, len(closes) // 10)
    for i in range(window, len(closes) - window):
        # Local maxima = resistance
        if highs[i] == max(highs[max(0, i-window):i+window+1]):
            resistance_levels.append(highs[i])
        # Local minima = support
        if lows[i] == min(lows[max(0, i-window):i+window+1]):
            support_levels.append(lows[i])
    
    # Cluster nearby levels
    def cluster_levels(levels, tolerance=0.001):
        if not levels:
            return []
        levels = sorted(set(levels))
        clustered = [levels[0]]
        for level in levels[1:]:
            if abs(level - clustered[-1]) / clustered[-1] > tolerance:
                clustered.append(level)
        return clustered
    
    resistance_levels = cluster_levels(resistance_levels)[-n_levels:]
    support_levels = cluster_levels(support_levels)[:n_levels]
    
    return support_levels, resistance_levels


def detect_asian_session_range(df: pd.DataFrame) -> dict:
    """Detect Asian session high/low (00:00-06:00 London time)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return {"asian_high": None, "asian_low": None}
    
    # Convert to UTC (approximate London time)
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    
    today = df_copy.index[-1].date()
    asian_mask = (
        (df_copy.index.date == today) &
        (df_copy.index.hour >= 0) &
        (df_copy.index.hour < 6)
    )
    
    asian_data = df_copy[asian_mask]
    if len(asian_data) > 0:
        return {
            "asian_high": asian_data['high'].max(),
            "asian_low": asian_data['low'].min(),
        }
    return {"asian_high": None, "asian_low": None}


def get_session_info(timestamp=None) -> dict:
    """Determine which trading session is currently active."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    hour_utc = timestamp.hour
    # London = UTC+0 (winter), UTC+1 (summer) - approximate as UTC
    london_hour = hour_utc  # Simplification
    
    sessions = {
        "asian": 0 <= london_hour < 7,
        "london": 7 <= london_hour < 16,
        "new_york": 13 <= london_hour < 22,
        "london_kill_zone": 7 <= london_hour < 10,
        "ny_kill_zone": 13 <= london_hour < 16,
    }
    
    active_sessions = [k for k, v in sessions.items() if v]
    return {**sessions, "active_sessions": active_sessions}


def detect_market_structure_breaks(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Detect breaks of market structure (BoS)."""
    highs = df['high'].rolling(lookback).max()
    lows = df['low'].rolling(lookback).min()
    
    bullish_bos = (df['close'] > highs.shift(1)).astype(int)
    bearish_bos = (df['close'] < lows.shift(1)).astype(int)
    
    return bullish_bos - bearish_bos  # +1 bullish break, -1 bearish break


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Detect liquidity sweeps (stop hunts)."""
    rolling_high = df['high'].rolling(lookback).max()
    rolling_low = df['low'].rolling(lookback).min()
    
    # Price briefly breaks level but closes back inside
    sweep_up = (
        (df['high'] > rolling_high.shift(1)) &
        (df['close'] < rolling_high.shift(1))
    ).astype(int)
    
    sweep_down = (
        (df['low'] < rolling_low.shift(1)) &
        (df['close'] > rolling_low.shift(1))
    ).astype(int)
    
    return sweep_up - sweep_down  # +1 = sweep above (bearish), -1 = sweep below (bullish)


def compute_price_acceleration(df: pd.DataFrame) -> pd.Series:
    """Measure price acceleration (second derivative of price)."""
    velocity = df['close'].diff()
    acceleration = velocity.diff()
    return acceleration / df['close'] * 1000  # Normalize


def compute_all_features(df: pd.DataFrame, dxy_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute all technical features and indicators."""
    features = df.copy()
    
    # EMAs
    for period in [9, 21, 50, 200]:
        features[f'ema_{period}'] = compute_ema(features['close'], period)
    
    # EMA trend alignment
    features['ema_bullish'] = (
        (features['ema_9'] > features['ema_21']) &
        (features['ema_21'] > features['ema_50']) &
        (features['ema_50'] > features['ema_200'])
    ).astype(int)
    
    features['ema_bearish'] = (
        (features['ema_9'] < features['ema_21']) &
        (features['ema_21'] < features['ema_50']) &
        (features['ema_50'] < features['ema_200'])
    ).astype(int)
    
    # RSI
    features['rsi'] = compute_rsi(features['close'])
    features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
    features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
    
    # MACD
    features['macd'], features['macd_signal'], features['macd_hist'] = compute_macd(features['close'])
    features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(int)
    
    # ATR
    features['atr'] = compute_atr(features)
    features['atr_pct'] = features['atr'] / features['close'] * 100
    
    # Bollinger Bands
    features['bb_upper'], features['bb_mid'], features['bb_lower'] = compute_bollinger_bands(features['close'])
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_mid']
    features['bb_pct'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).mean() * 0.8).astype(int)
    
    # VWAP
    features['vwap'] = compute_vwap(features)
    features['price_vs_vwap'] = (features['close'] - features['vwap']) / features['vwap'] * 100
    features['above_vwap'] = (features['close'] > features['vwap']).astype(int)
    
    # Volume
    features['volume_spike'] = detect_volume_spikes(features)
    features['volume_ratio'] = features['volume'] / features['volume'].rolling(20).mean()
    
    # ADX
    features['adx'] = compute_adx(features)
    features['trending'] = (features['adx'] > 25).astype(int)
    
    # Trend strength
    features['trend_strength'] = features['adx'] / 100
    
    # Momentum
    features['momentum_5'] = features['close'].pct_change(5) * 100
    features['momentum_10'] = features['close'].pct_change(10) * 100
    
    # Price acceleration
    features['price_acceleration'] = compute_price_acceleration(features)
    
    # Session highs/lows
    if isinstance(features.index, pd.DatetimeIndex):
        today = features.index[-1].date() if len(features) > 0 else None
        if today:
            today_mask = features.index.date == today
            today_data = features[today_mask]
            if len(today_data) > 0:
                session_high = today_data['high'].cummax()
                session_low = today_data['low'].cummin()
                features.loc[today_mask, 'session_high'] = session_high
                features.loc[today_mask, 'session_low'] = session_low
    
    if 'session_high' not in features.columns:
        features['session_high'] = features['high'].rolling(50).max()
        features['session_low'] = features['low'].rolling(50).min()
    
    # Market structure
    features['structure_break'] = detect_market_structure_breaks(features)
    
    # Liquidity sweep
    features['liquidity_sweep'] = detect_liquidity_sweep(features)
    
    # Volatility expansion
    features['vol_expansion'] = (
        features['atr'] > features['atr'].rolling(20).mean() * 1.5
    ).astype(int)
    
    # EMA slopes (trend direction)
    features['ema_21_slope'] = features['ema_21'].diff(3) / features['close'] * 1000
    features['ema_50_slope'] = features['ema_50'].diff(5) / features['close'] * 1000
    
    # DXY correlation (inverse relationship with gold)
    if dxy_df is not None and len(dxy_df) > 0:
        try:
            dxy_aligned = dxy_df['close'].reindex(features.index, method='ffill')
            features['dxy_price'] = dxy_aligned
            features['dxy_trend'] = (dxy_aligned > compute_ema(dxy_aligned, 21)).astype(int)
            # Correlation over 20 periods
            gold_ret = features['close'].pct_change()
            dxy_ret = dxy_aligned.pct_change()
            features['dxy_correlation'] = gold_ret.rolling(20).corr(dxy_ret)
        except:
            features['dxy_trend'] = 0
            features['dxy_correlation'] = -0.7  # Expected inverse correlation
    else:
        features['dxy_trend'] = 0
        features['dxy_correlation'] = -0.7
    
    # Time features
    if isinstance(features.index, pd.DatetimeIndex):
        features['hour'] = features.index.hour
        features['minute'] = features.index.minute
        features['day_of_week'] = features.index.dayofweek
        session = get_session_info()
        features['london_kill_zone'] = features['hour'].apply(lambda h: int(7 <= h < 10))
        features['ny_kill_zone'] = features['hour'].apply(lambda h: int(13 <= h < 16))
    else:
        features['hour'] = 12
        features['london_kill_zone'] = 0
        features['ny_kill_zone'] = 0
    
    # RSI divergence (simplified)
    features['rsi_divergence'] = 0
    rsi_slope = features['rsi'].diff(5)
    price_slope = features['close'].diff(5)
    features.loc[(price_slope > 0) & (rsi_slope < 0), 'rsi_divergence'] = -1  # Bearish divergence
    features.loc[(price_slope < 0) & (rsi_slope > 0), 'rsi_divergence'] = 1   # Bullish divergence
    
    return features


def get_support_resistance_levels(df: pd.DataFrame) -> tuple:
    """Return current support and resistance levels."""
    return detect_support_resistance(df)


def get_nearest_sr_levels(price: float, supports: list, resistances: list) -> dict:
    """Find nearest support and resistance to current price."""
    nearest_support = max([s for s in supports if s < price], default=price * 0.99)
    nearest_resistance = min([r for r in resistances if r > price], default=price * 1.01)
    
    dist_to_support = (price - nearest_support) / price * 100
    dist_to_resistance = (nearest_resistance - price) / price * 100
    
    return {
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "dist_to_support_pct": dist_to_support,
        "dist_to_resistance_pct": dist_to_resistance,
    }
