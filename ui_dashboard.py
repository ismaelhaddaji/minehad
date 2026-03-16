"""
ui_dashboard.py
Professional Quantitative Trading Dashboard for GOLD (XAUUSD).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="XAUUSD Quant Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; }
    .main { background-color: #0a0e1a; }
    
    .metric-card {
        background: linear-gradient(135deg, #0f1729 0%, #1a2340 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 16px;
        margin: 4px 0;
    }
    
    .signal-card-long {
        background: linear-gradient(135deg, #0d2b1a 0%, #1a4028 100%);
        border: 2px solid #00c853;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    
    .signal-card-short {
        background: linear-gradient(135deg, #2b0d0d 0%, #401a1a 100%);
        border: 2px solid #ff1744;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    
    .signal-card-neutral {
        background: linear-gradient(135deg, #1a1a2b 0%, #2a2a40 100%);
        border: 2px solid #3d5a80;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    
    .stMetric label { color: #7a9abf !important; font-size: 11px !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e8f4f8 !important; font-size: 22px !important; }
    .stMetric [data-testid="stMetricDelta"] { font-size: 12px !important; }
    
    h1, h2, h3 { color: #c9d8f0 !important; }
    .stSidebar { background-color: #0c1221 !important; }
    
    div[data-testid="stHorizontalBlock"] { gap: 8px; }
    
    .regime-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 13px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Imports (with error handling)
# ─────────────────────────────────────────────
try:
    from data_fetcher import fetch_all_timeframes, get_current_price, generate_synthetic_data
    from feature_engineering import compute_all_features, detect_support_resistance, detect_asian_session_range, get_session_info, get_nearest_sr_levels
    from strategies import run_all_strategies
    from regime_detection import detect_regime, get_regime_color
    from ensemble_engine import compute_ensemble_score
    from risk_manager import compute_position_size, validate_trade
    from ml_model import build_synthetic_model, predict_probability, get_feature_importance
    from backtesting import run_backtest
    from monte_carlo import run_monte_carlo, walk_forward_analysis
    from liquidity_analysis import get_liquidity_analysis
    MODULES_LOADED = True
except Exception as e:
    MODULES_LOADED = False
    st.error(f"Module load error: {e}")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "Alpha Vantage API Key",
        value="demo",
        type="password",
        help=(
            "Optional — even without a key, live data loads automatically "
            "via Yahoo Finance (GC=F gold futures). Enter your Alpha Vantage "
            "key for an additional data source."
        ),
    )

    # ── Data source status badge ─────────────────────────────────────────
    is_demo_key = api_key.lower().strip() in ("demo", "", "your_api_key_here")
    if is_demo_key:
        st.markdown(
            "<div style='background:#1a1200;border:1px solid #f0c040;"
            "border-radius:6px;padding:8px;font-size:11px;color:#f0c040;'>"
            "⚡ <b>No API key</b> — using Yahoo Finance for live data.<br>"
            "Data will still be real. Enter a key for Alpha Vantage as backup."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='background:#0d2b1a;border:1px solid #00c853;"
            "border-radius:6px;padding:8px;font-size:11px;color:#00c853;'>"
            "🔑 API key entered — trying Alpha Vantage + Yahoo Finance."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Diagnose button ──────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Diagnose Data Sources", use_container_width=True):
        with st.spinner("Testing all data sources..."):
            try:
                diag = diagnose_api(api_key)
                for src, info in diag.items():
                    if src == "summary":
                        continue
                    status = info.get("status", "?")
                    color = "#00c853" if status == "OK" else \
                            "#ff9800" if status in ("RATE_LIMITED", "NOT_INSTALLED") else "#ff1744"
                    detail = ""
                    if status == "OK":
                        p = info.get("price") or info.get("bars")
                        detail = f" — {'$' + str(p) if 'price' in info else str(p) + ' bars'}"
                    elif "msg" in info:
                        detail = f" — {info['msg'][:60]}"
                    st.markdown(
                        f"<span style='color:{color};font-size:12px;'>"
                        f"{'✓' if status=='OK' else '✗'} <b>{src}</b>: {status}{detail}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                summary = diag.get("summary", {})
                if summary.get("has_live_data"):
                    st.success(f"✅ Live data available via: {', '.join(summary['working'])}")
                else:
                    st.warning(
                        "⚠️ No live sources working.\n\n"
                        "**Fix:** Run `pip install yfinance` in your terminal "
                        "then restart the app. Yahoo Finance is free and needs no key."
                    )
            except Exception as e:
                st.error(f"Diagnostic error: {e}")

    st.markdown("---")
    st.markdown("### 📊 Analysis Settings")

    primary_tf = st.selectbox("Primary Timeframe", ["5min", "1min", "15min"], index=0)

    account_balance = st.number_input(
        "Account Balance ($)", value=10000, min_value=1000, step=1000
    )
    risk_pct = (
        st.slider("Risk Per Trade (%)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
        / 100
    )
    min_rr = st.select_slider(
        "Min Risk:Reward", options=[1.5, 2.0, 2.5, 3.0], value=2.0
    )

    st.markdown("---")
    st.markdown("### 🔄 Refresh")
    auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
    refresh_interval = 30

    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📈 Backtesting")
    run_backtest_btn = st.button("▶ Run Backtest", use_container_width=True)
    run_monte_carlo_btn = st.button("🎲 Monte Carlo", use_container_width=True)
    run_wf_btn = st.button("🔁 Walk Forward", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:10px; color:#4a6480; text-align:center;'>"
        "⚠️ For educational purposes only.<br>"
        "Not financial advice.<br>"
        "Trading involves substantial risk."
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data(api_key: str, primary_tf: str):
    """Load all market data. Returns (timeframes, features, current_price, sources)."""
    if not MODULES_LOADED:
        return None, None, None, {}

    timeframes, sources = fetch_all_timeframes(api_key)
    current_price = get_current_price(api_key)

    primary_df = timeframes.get(primary_tf, timeframes.get("5min"))
    dxy_df = timeframes.get("dxy")

    features = compute_all_features(primary_df, dxy_df)

    return timeframes, features, current_price, sources


@st.cache_data(ttl=60)
def run_full_analysis(api_key: str, primary_tf: str):
    """Run complete analysis pipeline."""
    if not MODULES_LOADED:
        return {}

    timeframes, features, current_price, sources = load_data(api_key, primary_tf)
    
    if features is None or len(features) < 20:
        return {}
    
    # Support/Resistance
    supports, resistances = detect_support_resistance(features)
    
    # Asian session range
    asian_range = detect_asian_session_range(features)
    
    # London range (proxy)
    london_range = {
        "london_high": features['high'].rolling(50).max().iloc[-1],
        "london_low": features['low'].rolling(50).min().iloc[-1],
    }
    
    # Session info
    session = get_session_info()
    
    # Regime detection
    regime_info = detect_regime(features)
    
    # Run strategies
    active_strategies, all_strategies = run_all_strategies(
        features, asian_range, london_range, supports, resistances
    )
    
    # ML model
    model_bundle = build_synthetic_model(features)
    ml_result = predict_probability(model_bundle, features) if model_bundle else {"probability": 0.5, "probability_pct": 50}
    
    # Ensemble signal
    ensemble = compute_ensemble_score(features, active_strategies, regime_info, ml_result.get("probability", 0.5))
    
    # Liquidity analysis
    liquidity = get_liquidity_analysis(features)
    
    # SR levels
    latest_price = features['close'].iloc[-1]
    sr_levels = get_nearest_sr_levels(latest_price, supports, resistances)
    
    # Position sizing
    if ensemble.get('signal', 0) != 0:
        position = compute_position_size(
            account_balance=10000,
            risk_pct=0.01,
            entry=ensemble.get('entry', latest_price),
            stop_loss=ensemble.get('stop_loss', latest_price * 0.998),
        )
        trade_valid = validate_trade(
            ensemble.get('entry', latest_price),
            ensemble.get('stop_loss', latest_price * 0.998),
            ensemble.get('take_profit', latest_price * 1.005),
        )
    else:
        position = {}
        trade_valid = {"valid": False}
    
    return {
        "features": features,
        "timeframes": timeframes,
        "current_price": current_price,
        "sources": sources,
        "supports": supports,
        "resistances": resistances,
        "asian_range": asian_range,
        "session": session,
        "regime_info": regime_info,
        "active_strategies": active_strategies,
        "all_strategies": all_strategies,
        "ml_result": ml_result,
        "ensemble": ensemble,
        "liquidity": liquidity,
        "sr_levels": sr_levels,
        "position": position,
        "trade_valid": trade_valid,
        "model_bundle": model_bundle,
    }


# ─────────────────────────────────────────────
# Main Dashboard
# ─────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(90deg, #0f1729, #1a2340); 
     padding: 16px 24px; border-radius: 10px; 
     border-bottom: 2px solid #1e3a5f; margin-bottom: 16px;'>
<h1 style='margin:0; font-size:26px; color:#f0c040 !important;'>
  🥇 XAUUSD Quantitative Trading Dashboard
</h1>
<p style='margin:0; color:#7a9abf; font-size:12px;'>
  Professional Institutional-Grade Research Platform | Multi-Strategy Ensemble Engine
</p>
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner("⚡ Analyzing markets..."):
    try:
        data = run_full_analysis(api_key, primary_tf)
    except Exception as e:
        data = {}
        st.error(f"Analysis error: {e}")

if not data:
    st.warning("⚠️ Data loading failed. Using demo mode with synthetic data.")
    # Create minimal demo data
    from data_fetcher import generate_synthetic_data
    demo_df = generate_synthetic_data("XAUUSD", "5min", 300)
    data = {"features": compute_all_features(demo_df) if MODULES_LOADED else None}

if not data or not data.get("features") is not None:
    st.stop()

features = data.get("features")
current_price_data = data.get("current_price", {"price": features['close'].iloc[-1] if features is not None else 2350})
ensemble = data.get("ensemble", {})
regime_info = data.get("regime_info", {})
session = data.get("session", {})
ml_result = data.get("ml_result", {})
supports = data.get("supports", [])
resistances = data.get("resistances", [])
asian_range = data.get("asian_range", {})
active_strategies = data.get("active_strategies", [])
all_strategies = data.get("all_strategies", [])
liquidity = data.get("liquidity", {})
position = data.get("position", {})

latest = features.iloc[-1] if features is not None and len(features) > 0 else pd.Series()
current_price = current_price_data.get("price", float(latest.get("close", 2350)))

# ── Data source banner ───────────────────────────────────────────────────────
sources = data.get("sources", {})
primary_src = sources.get(primary_tf, "Unknown")
price_src   = data.get("current_price", {}).get("source", "Unknown")
is_live     = any(
    "Live" in str(v) or "Alpha Vantage" in str(v) or "Yahoo" in str(v)
    for v in sources.values()
)
src_color  = "#00c853" if is_live else "#ff9800"
src_icon   = "🟢" if is_live else "🟡"
src_label  = "LIVE DATA" if is_live else "SYNTHETIC DATA"

st.markdown(
    f"<div style='background:#0f1729;border:1px solid {src_color};"
    f"border-radius:8px;padding:8px 14px;margin-bottom:10px;"
    f"display:flex;justify-content:space-between;align-items:center;'>"
    f"<span style='color:{src_color};font-size:12px;font-weight:bold;'>"
    f"{src_icon} {src_label}</span>"
    f"<span style='color:#7a9abf;font-size:11px;'>"
    f"Chart: <b>{primary_src}</b> &nbsp;|&nbsp; "
    f"Price: <b>{price_src}</b> &nbsp;|&nbsp; "
    f"Timeframe: <b>{primary_tf}</b>"
    f"</span>"
    f"{'<span style=\"color:#ff9800;font-size:11px;\">⚠️ Run \"Diagnose Data Sources\" in sidebar to fix</span>' if not is_live else ''}"
    f"</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# ROW 1: Market Overview
# ─────────────────────────────────────────────
st.markdown("### 📊 Market Overview")
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

prev_close = float(features['close'].iloc[-2]) if len(features) > 1 else current_price
price_change = current_price - prev_close
price_change_pct = price_change / prev_close * 100 if prev_close > 0 else 0
delta_color = "🟢" if price_change >= 0 else "🔴"

with col1:
    st.metric("XAUUSD", f"${current_price:,.2f}", f"{delta_color} {price_change:+.2f} ({price_change_pct:+.2f}%)")

with col2:
    rsi_val = float(latest.get('rsi', 50))
    rsi_color = "🔴" if rsi_val > 70 else ("🟢" if rsi_val < 30 else "🟡")
    st.metric("RSI (14)", f"{rsi_val:.1f}", f"{rsi_color} {'OB' if rsi_val > 70 else 'OS' if rsi_val < 30 else 'Neutral'}")

with col3:
    adx_val = float(latest.get('adx', 20))
    st.metric("ADX", f"{adx_val:.1f}", f"{'Strong' if adx_val > 30 else 'Moderate' if adx_val > 20 else 'Weak'}")

with col4:
    atr_val = float(latest.get('atr', 0))
    st.metric("ATR (14)", f"${atr_val:.2f}", f"{float(latest.get('atr_pct', 0)):.2f}%")

with col5:
    vwap_val = float(latest.get('vwap', current_price))
    vs_vwap = current_price - vwap_val
    st.metric("VWAP", f"${vwap_val:,.2f}", f"{'↑' if vs_vwap >= 0 else '↓'} ${vs_vwap:+.2f}")

with col6:
    regime = regime_info.get("regime", "Unknown")
    regime_short = regime.split()[0] if regime else "?"
    st.metric("Regime", regime_short, regime_info.get("description", "")[:20])

with col7:
    score = ensemble.get("score", 0)
    max_score = ensemble.get("max_score", 15)
    grade = ensemble.get("grade", "—")
    st.metric("Signal Score", f"{score}/{max_score}", f"Grade: {grade}")

# ─────────────────────────────────────────────
# ROW 2: Main Chart + Trade Setup
# ─────────────────────────────────────────────
chart_col, trade_col = st.columns([3, 1])

with chart_col:
    st.markdown("### 📈 Price Chart")
    
    # Build chart
    disp_features = features.iloc[-200:] if len(features) > 200 else features
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.02,
        subplot_titles=["XAUUSD Price", "RSI", "MACD"]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=disp_features.index,
        open=disp_features['open'],
        high=disp_features['high'],
        low=disp_features['low'],
        close=disp_features['close'],
        name="XAUUSD",
        increasing_line_color='#00e676',
        decreasing_line_color='#ff1744',
        increasing_fillcolor='#00c853',
        decreasing_fillcolor='#d50000',
    ), row=1, col=1)
    
    # EMAs
    ema_colors = {'ema_9': '#f0c040', 'ema_21': '#40c0f0', 'ema_50': '#f040c0', 'ema_200': '#c0c0c0'}
    for ema, color in ema_colors.items():
        if ema in disp_features.columns:
            fig.add_trace(go.Scatter(
                x=disp_features.index, y=disp_features[ema],
                name=ema.upper().replace('_', ' '), line=dict(color=color, width=1),
                opacity=0.8
            ), row=1, col=1)
    
    # VWAP
    if 'vwap' in disp_features.columns:
        fig.add_trace(go.Scatter(
            x=disp_features.index, y=disp_features['vwap'],
            name='VWAP', line=dict(color='#ff9800', width=1.5, dash='dot'),
            opacity=0.9
        ), row=1, col=1)
    
    # Bollinger Bands
    if 'bb_upper' in disp_features.columns:
        fig.add_trace(go.Scatter(
            x=disp_features.index, y=disp_features['bb_upper'],
            name='BB Upper', line=dict(color='rgba(100,150,255,0.4)', width=1),
            fill=None
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=disp_features.index, y=disp_features['bb_lower'],
            name='BB Lower', line=dict(color='rgba(100,150,255,0.4)', width=1),
            fill='tonexty', fillcolor='rgba(100,150,255,0.05)'
        ), row=1, col=1)
    
    # Support/Resistance
    for s in supports[-3:]:
        fig.add_hline(y=s, line_dash="dot", line_color="rgba(0,230,118,0.5)", row=1, col=1)
    for r in resistances[-3:]:
        fig.add_hline(y=r, line_dash="dot", line_color="rgba(255,23,68,0.5)", row=1, col=1)
    
    # Asian range
    if asian_range.get('asian_high') and asian_range.get('asian_low'):
        fig.add_hrect(
            y0=asian_range['asian_low'], y1=asian_range['asian_high'],
            fillcolor="rgba(255,200,0,0.07)", line_width=0,
            row=1, col=1
        )
    
    # Trade signal markers
    if ensemble.get('signal', 0) != 0:
        entry = ensemble.get('entry', current_price)
        stop = ensemble.get('stop_loss', current_price * 0.998)
        target = ensemble.get('take_profit', current_price * 1.005)
        ts = disp_features.index[-1]
        direction = ensemble.get('direction', 'LONG')
        
        # Entry marker
        fig.add_trace(go.Scatter(
            x=[ts], y=[entry],
            mode='markers+text',
            marker=dict(
                symbol='triangle-up' if direction == 'LONG' else 'triangle-down',
                size=16,
                color='#00e676' if direction == 'LONG' else '#ff1744',
                line=dict(width=2, color='white')
            ),
            text=[f" {direction}"],
            textposition='middle right',
            name=f"Signal: {direction}",
            textfont=dict(color='white', size=11)
        ), row=1, col=1)
        
        # Entry/Stop/Target lines
        fig.add_hline(y=entry, line_dash="solid", line_color="#ffffff", line_width=1.5,
                      annotation_text=f"Entry: {entry:.2f}", annotation_position="right",
                      row=1, col=1)
        fig.add_hline(y=stop, line_dash="dash", line_color="#ff5252", line_width=1.5,
                      annotation_text=f"Stop: {stop:.2f}", annotation_position="right",
                      row=1, col=1)
        fig.add_hline(y=target, line_dash="dash", line_color="#69f0ae", line_width=1.5,
                      annotation_text=f"Target: {target:.2f}", annotation_position="right",
                      row=1, col=1)
    
    # RSI subplot
    if 'rsi' in disp_features.columns:
        fig.add_trace(go.Scatter(
            x=disp_features.index, y=disp_features['rsi'],
            name='RSI', line=dict(color='#f0c040', width=1.5)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,80,80,0.5)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(80,255,80,0.5)", row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", row=2, col=1)
    
    # MACD subplot
    if 'macd' in disp_features.columns:
        fig.add_trace(go.Scatter(
            x=disp_features.index, y=disp_features['macd'],
            name='MACD', line=dict(color='#40c0f0', width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=disp_features.index, y=disp_features['macd_signal'],
            name='Signal', line=dict(color='#f040a0', width=1.5)
        ), row=3, col=1)
        colors = ['#00e676' if v >= 0 else '#ff1744' for v in disp_features['macd_hist'].fillna(0)]
        fig.add_trace(go.Bar(
            x=disp_features.index, y=disp_features['macd_hist'],
            name='Histogram', marker_color=colors, opacity=0.7
        ), row=3, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0f1729',
        font=dict(color='#c0d0e0', size=11),
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
            font=dict(size=10)
        ),
        margin=dict(l=10, r=80, t=30, b=10),
    )
    
    fig.update_xaxes(gridcolor='#1a2a3a', showgrid=True)
    fig.update_yaxes(gridcolor='#1a2a3a', showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)

with trade_col:
    st.markdown("### 🎯 Trade Setup")
    
    signal = ensemble.get("signal", 0)
    direction = ensemble.get("direction", "")
    score = ensemble.get("score", 0)
    
    if signal == 1:
        card_class = "signal-card-long"
        signal_icon = "🟢"
        bg_color = "#0d2b1a"
    elif signal == -1:
        card_class = "signal-card-short"
        signal_icon = "🔴"
        bg_color = "#2b0d0d"
    else:
        card_class = "signal-card-neutral"
        signal_icon = "⚪"
        bg_color = "#1a1a2b"
    
    st.markdown(f"""
    <div class="{card_class}">
      <div style='font-size:20px; font-weight:bold; color:{"#00e676" if signal==1 else "#ff1744" if signal==-1 else "#8888cc"};'>
        {signal_icon} {direction if direction else "NO SIGNAL"}
      </div>
      <div style='color:#8899aa; font-size:11px; margin-top:4px;'>
        Score: {score}/{ensemble.get('max_score', 15)} | Grade: {ensemble.get('grade', '—')}
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    if signal != 0:
        entry = ensemble.get('entry', current_price)
        stop = ensemble.get('stop_loss', 0)
        target = ensemble.get('take_profit', 0)
        rr = ensemble.get('risk_reward', 0)
        conf = ensemble.get('confidence_pct', 0)
        ml_prob = ml_result.get('probability_pct', 50)
        
        st.markdown(f"""
        <div style='background:#0f1729; border-radius:8px; padding:14px; margin:8px 0;'>
          <table style='width:100%; font-size:13px; border-collapse:collapse;'>
            <tr><td style='color:#8899aa; padding:4px 0;'>Entry</td>
                <td style='color:#fff; text-align:right; font-weight:bold;'>${entry:,.2f}</td></tr>
            <tr><td style='color:#ff5252; padding:4px 0;'>Stop Loss</td>
                <td style='color:#ff5252; text-align:right; font-weight:bold;'>${stop:,.2f}</td></tr>
            <tr><td style='color:#69f0ae; padding:4px 0;'>Take Profit</td>
                <td style='color:#69f0ae; text-align:right; font-weight:bold;'>${target:,.2f}</td></tr>
            <tr><td style='color:#8899aa; padding:4px 0;'>Risk:Reward</td>
                <td style='color:#f0c040; text-align:right; font-weight:bold;'>{rr:.2f}:1</td></tr>
            <tr><td style='color:#8899aa; padding:4px 0;'>Risk (pts)</td>
                <td style='color:#fff; text-align:right;'>${abs(entry-stop):.2f}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        st.markdown("**Confidence**")
        conf_color = "#00e676" if conf > 70 else ("#f0c040" if conf > 50 else "#ff5252")
        st.progress(int(min(conf, 100)))
        st.markdown(f"<span style='color:{conf_color}; font-weight:bold;'>{conf:.1f}%</span>", unsafe_allow_html=True)
        
        # ML Probability
        st.markdown("**ML Probability**")
        ml_color = "#00e676" if ml_prob > 60 else ("#f0c040" if ml_prob > 40 else "#ff5252")
        st.progress(int(min(ml_prob, 100)))
        st.markdown(f"<span style='color:{ml_color}; font-weight:bold;'>{ml_prob:.1f}%</span>", unsafe_allow_html=True)
        
        # Triggered strategies
        st.markdown("**Active Strategies:**")
        for strat_name in ensemble.get('triggered_strategies', [])[:5]:
            st.markdown(f"<span style='color:#40c0f0; font-size:12px;'>✓ {strat_name}</span>", unsafe_allow_html=True)
        
        # Score breakdown
        if ensemble.get('score_breakdown'):
            st.markdown("**Score Breakdown:**")
            for factor, pts in ensemble.get('score_breakdown', {}).items():
                st.markdown(f"<span style='color:#8899aa; font-size:11px;'>{factor}: <span style='color:#f0c040;'>{pts}</span></span>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#0f1729; border-radius:8px; padding:14px; margin:8px 0; color:#8899aa; font-size:13px;'>
          {ensemble.get('recommendation', 'Waiting for high-probability setup...')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Strategy Status:**")
        for strat in (all_strategies or []):
            is_active = strat.get('signal', 0) != 0
            icon = "🟡" if is_active else "⚫"
            st.markdown(f"<span style='font-size:11px; color:{'#f0c040' if is_active else '#445566'};'>{icon} {strat['name']}</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ROW 3: Market Analysis Details
# ─────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Market Analysis", "📐 Risk Panel", "🤖 ML Model", "📊 Backtest", "⚙️ Strategy Details"])

with tab1:
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("#### 🌍 Market Regime")
        regime = regime_info.get("regime", "Unknown")
        regime_color = get_regime_color(regime) if MODULES_LOADED else "#ffffff"
        
        st.markdown(f"""
        <div style='background:#0f1729; border-radius:10px; padding:16px; border-left:4px solid {regime_color};'>
          <div style='color:{regime_color}; font-size:18px; font-weight:bold;'>{regime}</div>
          <div style='color:#8899aa; font-size:12px; margin-top:8px;'>{regime_info.get('description', '')}</div>
          <hr style='border-color:#1e3a5f; margin:10px 0;'>
          <div style='font-size:12px;'>
            <span style='color:#8899aa;'>ADX: </span><span style='color:#fff;'>{regime_info.get('adx', 0):.1f}</span>
            &nbsp;&nbsp;
            <span style='color:#8899aa;'>Trend: </span><span style='color:#fff;'>{regime_info.get('trend_strength', 'N/A')}</span>
          </div>
          <div style='font-size:12px; margin-top:4px;'>
            <span style='color:#8899aa;'>EMA: </span><span style='color:#fff;'>{regime_info.get('ema_alignment', 'N/A').upper()}</span>
            &nbsp;&nbsp;
            <span style='color:#8899aa;'>Strategy: </span><span style='color:#f0c040;'>{regime_info.get('strategy_focus', 'N/A').replace('_', ' ').title()}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### ⏰ Session Status")
        session_data = session if session else get_session_info() if MODULES_LOADED else {}
        session_items = {
            "Asian": session_data.get('asian', False),
            "London": session_data.get('london', False),
            "New York": session_data.get('new_york', False),
            "🔥 London KZ": session_data.get('london_kill_zone', False),
            "🔥 NY KZ": session_data.get('ny_kill_zone', False),
        }
        for sess, active in session_items.items():
            color = "#00e676" if active else "#445566"
            dot = "●" if active else "○"
            st.markdown(f"<span style='color:{color}; font-size:13px;'>{dot} {sess}</span>", unsafe_allow_html=True)
    
    with col_b:
        st.markdown("#### 💧 Liquidity Analysis")
        
        if liquidity:
            price = current_price
            liq_above = liquidity.get('nearest_liquidity_above', price * 1.01)
            liq_below = liquidity.get('nearest_liquidity_below', price * 0.99)
            poc = liquidity.get('poc', price)
            vah = liquidity.get('value_area_high', price * 1.005)
            val = liquidity.get('value_area_low', price * 0.995)
            
            st.markdown(f"""
            <div style='background:#0f1729; border-radius:10px; padding:16px;'>
              <table style='width:100%; font-size:12px;'>
                <tr><td style='color:#ff5252; padding:3px 0;'>🔴 Liquidity Above</td>
                    <td style='color:#fff; text-align:right;'>${liq_above:,.2f}</td></tr>
                <tr><td style='color:#8899aa; padding:3px 0;'>📊 POC (HVN)</td>
                    <td style='color:#f0c040; text-align:right;'>${poc:,.2f}</td></tr>
                <tr><td style='color:#8899aa; padding:3px 0;'>📈 Value Area High</td>
                    <td style='color:#fff; text-align:right;'>${vah:,.2f}</td></tr>
                <tr><td style='color:#8899aa; padding:3px 0;'>📉 Value Area Low</td>
                    <td style='color:#fff; text-align:right;'>${val:,.2f}</td></tr>
                <tr><td style='color:#00e676; padding:3px 0;'>🟢 Liquidity Below</td>
                    <td style='color:#fff; text-align:right;'>${liq_below:,.2f}</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Equal highs/lows
            if liquidity.get('equal_highs'):
                st.markdown("**Equal Highs (Stop Hunt Targets):**")
                for h in liquidity.get('equal_highs', [])[-3:]:
                    st.markdown(f"<span style='color:#ff7070; font-size:12px;'>◉ ${h:,.2f}</span>", unsafe_allow_html=True)
            
            if liquidity.get('equal_lows'):
                st.markdown("**Equal Lows (Stop Hunt Targets):**")
                for l in liquidity.get('equal_lows', [])[:3]:
                    st.markdown(f"<span style='color:#70ff70; font-size:12px;'>◉ ${l:,.2f}</span>", unsafe_allow_html=True)
        
        st.markdown("#### 🌐 DXY Macro")
        dxy_trend = latest.get('dxy_trend', 0)
        dxy_corr = latest.get('dxy_correlation', -0.7)
        st.markdown(f"""
        <div style='background:#0f1729; border-radius:8px; padding:12px;'>
          <div style='font-size:13px;'>
            <span style='color:#8899aa;'>DXY Trend: </span>
            <span style='color:{"#ff5252" if dxy_trend else "#00e676"};'>
              {"📈 Bullish USD (Bearish Gold)" if dxy_trend else "📉 Bearish USD (Bullish Gold)"}
            </span>
          </div>
          <div style='font-size:13px; margin-top:6px;'>
            <span style='color:#8899aa;'>Gold/DXY Correlation: </span>
            <span style='color:#f0c040;'>{float(dxy_corr):.2f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown("#### 📏 Key Levels")
        sr_levels = data.get('sr_levels', {})
        
        price_levels = []
        for r in sorted(resistances, reverse=True)[:3]:
            price_levels.append(("RESISTANCE", r, "#ff5252"))
        price_levels.append(("── PRICE ──", current_price, "#ffffff"))
        for s in sorted(supports, reverse=True)[:3]:
            price_levels.append(("SUPPORT", s, "#00e676"))
        
        st.markdown('<div style="background:#0f1729; border-radius:10px; padding:12px;">', unsafe_allow_html=True)
        for level_type, level_price, color in price_levels:
            is_current = level_type == "── PRICE ──"
            bg = "background:#1a2a3a; border-radius:4px; padding:2px 6px;" if is_current else ""
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; padding:3px 0; {bg}'>
              <span style='color:{color}; font-size:11px;'>{level_type}</span>
              <span style='color:{color}; font-size:12px; font-weight:bold;'>${level_price:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Asian Range
        if asian_range.get('asian_high'):
            st.markdown("#### 🌏 Asian Session Range")
            ah = asian_range['asian_high']
            al = asian_range['asian_low']
            ar = ah - al
            st.markdown(f"""
            <div style='background:#0f1729; border-radius:8px; padding:12px;'>
              <div style='font-size:12px;'>
                <span style='color:#f0c040;'>High: ${ah:,.2f}</span>&nbsp;&nbsp;
                <span style='color:#f0c040;'>Low: ${al:,.2f}</span>
              </div>
              <div style='font-size:12px; color:#8899aa; margin-top:4px;'>
                Range: ${ar:.2f} | Mid: ${(ah+al)/2:,.2f}
              </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Multi-timeframe snapshot
        st.markdown("#### 🕐 Multi-Timeframe")
        timeframes = data.get('timeframes', {})
        for tf in ['1min', '5min', '15min']:
            tf_df = timeframes.get(tf)
            if tf_df is not None and len(tf_df) > 1:
                tf_close = tf_df['close'].iloc[-1]
                tf_prev = tf_df['close'].iloc[-2]
                tf_chg = (tf_close - tf_prev) / tf_prev * 100
                color = "#00e676" if tf_chg >= 0 else "#ff5252"
                arrow = "▲" if tf_chg >= 0 else "▼"
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; font-size:12px; padding:2px 0;'>
                  <span style='color:#8899aa;'>{tf}</span>
                  <span style='color:#fff;'>${tf_close:,.2f}</span>
                  <span style='color:{color};'>{arrow} {abs(tf_chg):.2f}%</span>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### 💰 Position Sizing & Risk Management")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        if position and ensemble.get('signal', 0) != 0:
            entry = ensemble.get('entry', current_price)
            stop = ensemble.get('stop_loss', current_price * 0.998)
            target = ensemble.get('take_profit', current_price * 1.005)
            
            pos_data = compute_position_size(account_balance, risk_pct, entry, stop) if MODULES_LOADED else position
            
            st.markdown(f"""
            <div style='background:#0f1729; border-radius:10px; padding:20px;'>
              <h4 style='color:#f0c040; margin-top:0;'>Position Calculator</h4>
              <table style='width:100%; font-size:13px;'>
                <tr><td style='color:#8899aa; padding:5px 0;'>Account Balance</td>
                    <td style='color:#fff; text-align:right;'>${account_balance:,.0f}</td></tr>
                <tr><td style='color:#8899aa; padding:5px 0;'>Risk Per Trade</td>
                    <td style='color:#ff9800; text-align:right;'>{risk_pct*100:.1f}% (${account_balance*risk_pct:,.0f})</td></tr>
                <tr><td style='color:#8899aa; padding:5px 0;'>Entry Price</td>
                    <td style='color:#fff; text-align:right;'>${entry:,.2f}</td></tr>
                <tr><td style='color:#ff5252; padding:5px 0;'>Stop Loss</td>
                    <td style='color:#ff5252; text-align:right;'>${stop:,.2f}</td></tr>
                <tr><td style='color:#69f0ae; padding:5px 0;'>Take Profit</td>
                    <td style='color:#69f0ae; text-align:right;'>${target:,.2f}</td></tr>
                <tr><td style='color:#8899aa; padding:5px 0;'>Price Risk</td>
                    <td style='color:#fff; text-align:right;'>${abs(entry-stop):.2f}</td></tr>
                <tr><td style='color:#8899aa; padding:5px 0;'>Recommended Lots</td>
                    <td style='color:#f0c040; text-align:right; font-weight:bold;'>{pos_data.get('lots', 0):.2f}</td></tr>
                <tr><td style='color:#8899aa; padding:5px 0;'>Max Loss</td>
                    <td style='color:#ff5252; text-align:right;'>-${pos_data.get('max_loss', 0):,.2f}</td></tr>
                <tr><td style='color:#8899aa; padding:5px 0;'>Max Gain (est.)</td>
                    <td style='color:#00e676; text-align:right;'>+${abs(account_balance*risk_pct*ensemble.get('risk_reward', 2.5)):,.2f}</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No active trade signal. Position sizing will appear when a setup is detected.")
    
    with col_r2:
        st.markdown("#### 🎲 Monte Carlo Preview")
        # Quick MC simulation
        mc_result = run_monte_carlo(data.get('all_trades', []), n_simulations=500, initial_capital=account_balance) if MODULES_LOADED else {}
        
        if mc_result:
            st.markdown(f"""
            <div style='background:#0f1729; border-radius:10px; padding:16px;'>
              <table style='width:100%; font-size:13px;'>
                <tr><td style='color:#8899aa; padding:4px 0;'>Simulations Run</td>
                    <td style='color:#fff; text-align:right;'>500</td></tr>
                <tr><td style='color:#ff5252; padding:4px 0;'>Probability of Ruin</td>
                    <td style='color:#ff5252; text-align:right; font-weight:bold;'>{mc_result.get('probability_of_ruin', 0):.1f}%</td></tr>
                <tr><td style='color:#ff9800; padding:4px 0;'>Worst Drawdown</td>
                    <td style='color:#ff9800; text-align:right;'>{mc_result.get('worst_case_drawdown', 0):.1f}%</td></tr>
                <tr><td style='color:#f0c040; padding:4px 0;'>Median Drawdown</td>
                    <td style='color:#f0c040; text-align:right;'>{mc_result.get('median_drawdown', 0):.1f}%</td></tr>
                <tr><td style='color:#8899aa; padding:4px 0;'>Expected Return</td>
                    <td style='color:#00e676; text-align:right;'>{mc_result.get('expected_return_pct', 0):.1f}%</td></tr>
                <tr><td style='color:#8899aa; padding:4px 0;'>5th Percentile</td>
                    <td style='color:#fff; text-align:right;'>${mc_result.get('p5_final_capital', 0):,.0f}</td></tr>
                <tr><td style='color:#8899aa; padding:4px 0;'>95th Percentile</td>
                    <td style='color:#fff; text-align:right;'>${mc_result.get('p95_final_capital', 0):,.0f}</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Plot sample MC curves
            curves = mc_result.get('sample_equity_curves', [])
            if curves:
                fig_mc = go.Figure()
                for i, curve in enumerate(curves[:20]):
                    fig_mc.add_trace(go.Scatter(
                        y=curve, mode='lines',
                        line=dict(width=0.5, color='rgba(64,192,240,0.3)'),
                        showlegend=False
                    ))
                # Median
                median_curve = np.median(np.array(curves), axis=0)
                fig_mc.add_trace(go.Scatter(
                    y=median_curve, mode='lines',
                    line=dict(width=2, color='#f0c040'),
                    name='Median'
                ))
                fig_mc.update_layout(
                    title="Monte Carlo Equity Curves",
                    template="plotly_dark",
                    paper_bgcolor='#0a0e1a',
                    plot_bgcolor='#0f1729',
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10),
                    font=dict(size=10)
                )
                st.plotly_chart(fig_mc, use_container_width=True)

with tab3:
    st.markdown("#### 🤖 Machine Learning Model")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        model_bundle = data.get('model_bundle', {})
        
        if model_bundle and model_bundle.get('cv_scores'):
            st.markdown("**Model Performance (Cross-Validation)**")
            cv_scores = model_bundle.get('cv_scores', {})
            
            for model_name, score_val in cv_scores.items():
                bar_pct = int(score_val * 100)
                color = "#00e676" if score_val > 0.6 else ("#f0c040" if score_val > 0.5 else "#ff5252")
                st.markdown(f"""
                <div style='margin:6px 0;'>
                  <div style='display:flex; justify-content:space-between; font-size:12px;'>
                    <span style='color:#8899aa;'>{model_name.replace('_', ' ').title()}</span>
                    <span style='color:{color};'>{score_val*100:.1f}%</span>
                  </div>
                  <div style='background:#1a2a3a; border-radius:4px; height:6px; margin-top:2px;'>
                    <div style='background:{color}; width:{bar_pct}%; height:100%; border-radius:4px;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            
            avg_acc = model_bundle.get('avg_accuracy', 0)
            st.markdown(f"""
            <div style='background:#0f1729; border-radius:8px; padding:12px; margin-top:12px;'>
              <span style='color:#8899aa;'>Ensemble Accuracy: </span>
              <span style='color:#f0c040; font-size:18px; font-weight:bold;'>{avg_acc*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Training ML model on available data...")
        
        # Current prediction
        st.markdown("**Current Prediction**")
        ml = ml_result or {}
        prob = ml.get('probability_pct', 50)
        conf = ml.get('confidence', 'medium')
        color = "#00e676" if prob > 60 else ("#f0c040" if prob > 45 else "#ff5252")
        
        st.markdown(f"""
        <div style='background:#0f1729; border-radius:10px; padding:16px;'>
          <div style='font-size:32px; font-weight:bold; color:{color}; text-align:center;'>
            {prob:.1f}%
          </div>
          <div style='text-align:center; color:#8899aa; font-size:12px;'>
            Trade Success Probability<br>Confidence: {conf.upper()}
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        if ml.get('individual_probs'):
            st.markdown("**Per-Model Predictions:**")
            for mname, mprob in ml.get('individual_probs', {}).items():
                st.markdown(f"<span style='color:#8899aa; font-size:12px;'>{mname}: <span style='color:#fff;'>{mprob*100:.1f}%</span></span>", unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("**Feature Importance**")
        if model_bundle:
            importance = get_feature_importance(model_bundle) if MODULES_LOADED else {}
            if importance:
                feat_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
                feat_df['Importance'] = feat_df['Importance'] / feat_df['Importance'].sum() * 100
                
                fig_fi = px.bar(
                    feat_df.head(10), x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Blues',
                    title='Top 10 Features'
                )
                fig_fi.update_layout(
                    template="plotly_dark", paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1729',
                    height=300, margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=False, font=dict(size=11), coloraxis_showscale=False
                )
                st.plotly_chart(fig_fi, use_container_width=True)
        
        st.markdown("**Current Feature Values**")
        key_features = ['rsi', 'macd', 'atr_pct', 'adx', 'bb_pct', 'price_vs_vwap',
                        'momentum_5', 'trend_strength', 'volume_ratio']
        feat_data = []
        for feat in key_features:
            val = latest.get(feat, 0)
            if val is not None:
                feat_data.append({"Feature": feat, "Value": f"{float(val):.3f}"})
        
        if feat_data:
            feat_display_df = pd.DataFrame(feat_data)
            st.dataframe(feat_display_df, use_container_width=True, height=200,
                         hide_index=True)

with tab4:
    st.markdown("#### 📊 Strategy Backtesting")
    
    if run_backtest_btn:
        with st.spinner("Running backtest..."):
            try:
                bt_result = run_backtest(features, initial_capital=account_balance, risk_pct=risk_pct)
                st.session_state['bt_result'] = bt_result
            except Exception as e:
                st.error(f"Backtest error: {e}")
    
    bt_result = st.session_state.get('bt_result', None)
    
    if bt_result:
        col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
        with col_bt1:
            st.metric("Total Trades", bt_result.get('total_trades', 0))
            st.metric("Win Rate", f"{bt_result.get('win_rate', 0):.1f}%")
        with col_bt2:
            st.metric("Profit Factor", f"{bt_result.get('profit_factor', 0):.2f}")
            st.metric("Sharpe Ratio", f"{bt_result.get('sharpe_ratio', 0):.2f}")
        with col_bt3:
            st.metric("Max Drawdown", f"{bt_result.get('max_drawdown', 0):.1f}%")
            st.metric("Total Return", f"{bt_result.get('total_return', 0):.1f}%")
        with col_bt4:
            st.metric("Avg Win", f"${bt_result.get('avg_win', 0):.2f}")
            st.metric("Expectancy", f"${bt_result.get('expectancy', 0):.2f}")
        
        # Equity curve
        ec = bt_result.get('equity_curve', [])
        if ec:
            fig_ec = go.Figure()
            fig_ec.add_trace(go.Scatter(
                y=ec, mode='lines', name='Equity',
                line=dict(color='#40c0f0', width=2),
                fill='tozeroy', fillcolor='rgba(64,192,240,0.1)'
            ))
            fig_ec.add_hline(y=account_balance, line_dash='dash',
                              line_color='rgba(255,255,255,0.3)')
            fig_ec.update_layout(
                title="Equity Curve",
                template="plotly_dark", paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1729',
                height=300, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_ec, use_container_width=True)
    else:
        st.info("Click '▶ Run Backtest' in the sidebar to analyze strategy performance.")
    
    # Walk Forward
    if run_wf_btn:
        with st.spinner("Running walk-forward analysis..."):
            try:
                wf_result = walk_forward_analysis(features)
                st.session_state['wf_result'] = wf_result
            except Exception as e:
                st.error(f"Walk forward error: {e}")
    
    wf_result = st.session_state.get('wf_result', None)
    if wf_result and wf_result.get('results'):
        st.markdown("#### 🔁 Walk-Forward Results")
        st.metric("Avg Win Rate", f"{wf_result.get('avg_win_rate', 0):.1f}%")
        wf_df = pd.DataFrame(wf_result['results'])
        st.dataframe(wf_df, use_container_width=True, hide_index=True)
    
    # Monte Carlo (detailed)
    if run_monte_carlo_btn:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                mc_full = run_monte_carlo(
                    bt_result.get('trades', []) if bt_result else [],
                    n_simulations=2000, initial_capital=account_balance
                )
                st.session_state['mc_full'] = mc_full
            except Exception as e:
                st.error(f"Monte Carlo error: {e}")
    
    mc_full = st.session_state.get('mc_full', None)
    if mc_full:
        st.markdown("#### 🎲 Full Monte Carlo Results")
        col_mc1, col_mc2 = st.columns(2)
        with col_mc1:
            st.metric("Probability of Ruin", f"{mc_full.get('probability_of_ruin', 0):.1f}%")
            st.metric("Worst Drawdown", f"{mc_full.get('worst_case_drawdown', 0):.1f}%")
        with col_mc2:
            st.metric("Expected Return", f"{mc_full.get('expected_return_pct', 0):.1f}%")
            st.metric("95th Percentile Capital", f"${mc_full.get('p95_final_capital', 0):,.0f}")
        
        curves = mc_full.get('sample_equity_curves', [])
        if curves:
            fig_mc2 = go.Figure()
            for curve in curves[:50]:
                fig_mc2.add_trace(go.Scatter(
                    y=curve, mode='lines',
                    line=dict(width=0.4, color='rgba(64,192,240,0.25)'),
                    showlegend=False
                ))
            median_curve = np.median(np.array(curves), axis=0)
            fig_mc2.add_trace(go.Scatter(y=median_curve, mode='lines',
                                         line=dict(width=2.5, color='#f0c040'), name='Median'))
            fig_mc2.update_layout(
                title=f"Monte Carlo: {mc_full.get('n_simulations', 2000)} Simulations",
                template="plotly_dark", paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1729',
                height=350, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_mc2, use_container_width=True)

with tab5:
    st.markdown("#### ⚙️ Strategy Details")
    
    for strat in (all_strategies or []):
        is_active = strat.get('signal', 0) != 0
        border_color = "#00c853" if strat.get('signal', 0) == 1 else ("#ff1744" if strat.get('signal', 0) == -1 else "#333355")
        bg_color = "#0d2b1a" if strat.get('signal', 0) == 1 else ("#2b0d0d" if strat.get('signal', 0) == -1 else "#0f1729")
        
        with st.expander(f"{'🟢' if strat.get('signal',0)==1 else '🔴' if strat.get('signal',0)==-1 else '⚫'} {strat['name']} — Score: {strat.get('score', 0)}", expanded=is_active):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"**Signal:** {'LONG' if strat.get('signal',0)==1 else 'SHORT' if strat.get('signal',0)==-1 else 'NEUTRAL'}")
                st.markdown(f"**Score:** {strat.get('score', 0)}")
                if strat.get('entry'):
                    st.markdown(f"**Entry:** ${strat.get('entry', 0):,.2f}")
                if strat.get('stop_loss'):
                    st.markdown(f"**Stop:** ${strat.get('stop_loss', 0):,.2f}")
                if strat.get('take_profit'):
                    st.markdown(f"**Target:** ${strat.get('take_profit', 0):,.2f}")
            with col_s2:
                st.markdown("**Reasons:**")
                for reason in strat.get('reason', ['No signal triggered']):
                    st.markdown(f"• {reason}")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown(f"<span style='color:#4a6480; font-size:11px;'>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</span>", unsafe_allow_html=True)
with col_f2:
    st.markdown(f"<span style='color:#4a6480; font-size:11px; text-align:center; display:block;'>Timeframe: {primary_tf} | Bars Loaded: {len(features) if features is not None else 0}</span>", unsafe_allow_html=True)
with col_f3:
    st.markdown("<span style='color:#4a6480; font-size:11px; text-align:right; display:block;'>⚠️ Educational purposes only — Not financial advice</span>", unsafe_allow_html=True)

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.cache_data.clear()
    st.rerun()
