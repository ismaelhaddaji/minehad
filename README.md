# 🥇 XAUUSD Quantitative Trading Dashboard

A professional institutional-grade quantitative trading research platform for GOLD (XAUUSD).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run ui_dashboard.py
```

### 3. API Key Setup
- Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- Enter it in the sidebar
- Leave as `demo` to run with high-quality synthetic data

---

## 📁 Project Structure

```
gold_trading_dashboard/
├── ui_dashboard.py         # Main Streamlit dashboard
├── data_fetcher.py         # Alpha Vantage API + synthetic data
├── feature_engineering.py  # Technical indicators (EMA, RSI, MACD, ATR, BB, VWAP, ADX...)
├── strategies.py           # 8 trading strategies
├── regime_detection.py     # Market regime classifier
├── ensemble_engine.py      # Signal scoring engine (max 15 pts, threshold 9)
├── risk_manager.py         # Position sizing & trade validation
├── ml_model.py             # Random Forest, GBM, XGBoost, Logistic Regression
├── backtesting.py          # Event-driven backtester
├── monte_carlo.py          # Monte Carlo + Walk-Forward analysis
├── liquidity_analysis.py   # Order flow & liquidity pool detection
└── requirements.txt
```

---

## 🎯 Features

### Market Analysis
- Live XAUUSD price via Alpha Vantage (falls back to synthetic data)
- Multi-timeframe: 1min, 5min, 15min
- DXY macro filter (inverse USD-Gold correlation)
- Asian/London/New York session detection

### Technical Indicators
- EMA 9/21/50/200 alignment
- RSI + divergence detection
- MACD + crossover signals
- ATR volatility measurement
- Bollinger Bands + squeeze detection
- VWAP session + price position
- ADX trend strength
- Volume spike detection

### Market Regime Detection
- Trending Up / Trending Down
- Ranging / Mean-Reversion
- High Volatility
- Low Volatility Compression (squeeze)

### Trading Strategies (8 total)
1. **London Kill Zone Breakout** — Asian range break during 07:00-10:00
2. **New York Reversal** — Liquidity sweep reversal at NY open
3. **VWAP Liquidity Sweep Reversal** — Prop trader high win-rate setup
4. **EMA Trend Following** — Full alignment pullback entry
5. **RSI Momentum** — Oversold/overbought with divergence
6. **MACD Crossover** — Signal line crossovers
7. **Bollinger Band Breakout** — Squeeze expansion plays
8. **S/R Breakout** — Structure level breaks with volume

### Ensemble Signal Engine
Combines all signals into a score (max 15):
- Trend confirmation: +2
- Momentum confirmation: +2
- Volatility breakout: +2
- Structure breakout: +2
- Volume confirmation: +2
- Kill Zone alignment: +2
- Liquidity sweep: +2
- DXY correlation: +1

**Only fires when score ≥ 9**

### Machine Learning
- Random Forest, Gradient Boosting, XGBoost, Logistic Regression
- Cross-validation accuracy reporting
- Feature importance visualization
- Trade success probability estimation

### Risk Management
- ATR-based and swing-based stop losses
- Minimum 2:1 R:R enforcement
- Position sizing calculator
- Trade validation

### Backtesting & Research
- Event-driven backtest engine
- Win rate, profit factor, Sharpe ratio, max drawdown
- Equity curve visualization
- Walk-forward optimization
- Monte Carlo simulation (probability of ruin, drawdown distribution)

### Liquidity Analysis
- Equal highs/lows detection (stop hunt targets)
- Volume profile / Point of Control
- Order block detection
- Value area high/low

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**.
It does not constitute financial advice.
Trading involves substantial risk of loss.
Past performance does not guarantee future results.
