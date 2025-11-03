# 🚀 Advanced Pairs Trading Analytics Platform

A sophisticated real-time statistical arbitrage platform for cryptocurrency pairs trading, featuring live market data ingestion, advanced analytics, and intelligent trade signal generation.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Analytics Explained](#analytics-explained)
- [Usage Guide](#usage-guide)
- [Trading Strategy](#trading-strategy)
- [Configuration](#configuration)
- [Database Schema](#database-schema)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## 🎯 Overview

This platform implements a **statistical arbitrage strategy** for cryptocurrency pairs trading. It monitors two correlated assets (e.g., BTC/USDT and ETH/USDT), calculates their spread, and generates trading signals based on mean-reversion principles.

### Key Concepts

- **Pairs Trading**: Market-neutral strategy exploiting temporary deviations in historically correlated assets
- **Mean Reversion**: Statistical phenomenon where prices tend to return to their long-term average
- **Cointegration**: Long-term equilibrium relationship between two time series
- **Z-Score**: Standardized measure of spread deviation from historical mean

---

## ✨ Features

### Real-Time Data Processing
- 🔄 **Live WebSocket Integration** - Direct connection to Binance streams
- ⚡ **Sub-second Analytics** - Fast-loop processing every 0.5s
- 📊 **Multi-timeframe Analysis** - Tick-level and resampled data

### Advanced Analytics
- 📈 **Cointegration Testing** - Augmented Dickey-Fuller (ADF) test
- 🎯 **Dynamic Hedge Ratios** - OLS regression-based beta calculation
- 📉 **Mean Reversion Metrics** - Half-life estimation
- 💹 **Risk-Adjusted Returns** - Sharpe ratio calculation
- 🔗 **Rolling Correlations** - Real-time relationship monitoring

### Trading Intelligence
- 🎲 **Automated Signal Generation** - Entry/exit signals based on z-scores
- 🎚️ **Confidence Scoring** - Weighted signal reliability metrics
- ⚠️ **Risk Management** - Stop-loss and position sizing guidance
- 🔔 **Custom Alerts** - User-defined threshold notifications

### Visualization & Export
- 📊 **Interactive Dashboards** - Real-time Plotly charts
- 💾 **Data Export** - CSV downloads for all data types
- 📱 **Responsive UI** - Streamlit-based modern interface

---

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager
- **Internet connection** (for WebSocket data)

### Installation Steps

1. **Clone or Download** the repository:
```bash
git clone https://github.com/yourusername/pairs-trading-platform.git
cd pairs-trading-platform
```

2. **Create Virtual Environment** (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies

Create a `requirements.txt` file with:

```txt
streamlit==1.28.0
pandas==2.1.0
numpy==1.24.3
plotly==5.17.0
statsmodels==0.14.0
scipy==1.11.2
websocket-client==1.6.3
```

Or install individually:
```bash
pip install streamlit pandas numpy plotly statsmodels scipy websocket-client
```

### Quick Start

```bash
streamlit run app.py
```

The platform will:
1. Initialize SQLite database (`ticks.db`)
2. Connect to Binance WebSocket streams
3. Start background analytics threads
4. Launch web interface at `http://localhost:8501`

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI Layer                     │
│  (Dashboards, Charts, Controls, Alerts)                 │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│              Analytics Engine                            │
│  ┌─────────────────┐      ┌─────────────────┐          │
│  │  Fast Analytics │      │  Slow Analytics │          │
│  │   (0.5s loop)   │      │   (30s loop)    │          │
│  │                 │      │                 │          │
│  │ • Z-Score       │      │ • ADF Test      │          │
│  │ • Correlation   │      │ • Cointegration │          │
│  │ • Volatility    │      │ • Resampling    │          │
│  │ • Signals       │      │ • Long-term     │          │
│  └────────┬────────┘      └────────┬────────┘          │
└───────────┼──────────────────────┼──────────────────────┘
            │                      │
┌───────────▼──────────────────────▼──────────────────────┐
│                SQLite Database (WAL Mode)                │
│  • ticks            • analytics                          │
│  • trade_signals    • alerts                             │
│  • alert_events                                          │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│            WebSocket Data Ingestion                      │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ BTC/USDT     │         │ ETH/USDT     │             │
│  │ Stream       │         │ Stream       │             │
│  └──────────────┘         └──────────────┘             │
│           Binance WebSocket API                          │
└─────────────────────────────────────────────────────────┘
```

### Threading Model

1. **Main Thread**: Streamlit UI rendering and user interaction
2. **WebSocket Threads**: One per trading pair for data ingestion
3. **Fast Analytics Thread**: Real-time calculations (tick-level)
4. **Slow Analytics Thread**: Statistical tests (resampled data)

### Data Flow

```
Market Data → WebSocket → SQLite → Analytics → Signals → UI
                                      ↓
                                   Alerts
```

---

## 📊 Methodology

### Pairs Trading Framework

#### 1. Pair Selection
- **Correlation Requirement**: Assets must show high correlation (>0.7)
- **Cointegration Test**: ADF test p-value < 0.05 indicates stable relationship
- **Liquidity**: High-volume pairs for minimal slippage

#### 2. Spread Construction

The **spread** represents the deviation between two assets:

```
Spread = Price_A - β × Price_B
```

Where:
- `Price_A`: Price of first asset (e.g., BTC/USDT)
- `Price_B`: Price of second asset (e.g., ETH/USDT)
- `β` (beta): Hedge ratio calculated via OLS regression

**Why use β?**
- Normalizes the relationship between assets
- Accounts for different price scales
- Creates a stationary spread suitable for mean reversion

#### 3. Hedge Ratio Estimation

Using **Ordinary Least Squares (OLS)** regression:

```python
Y = α + β×X + ε

Where:
Y = Prices of Asset A
X = Prices of Asset B
β = Hedge ratio (slope)
```

**Implementation**:
```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

X = add_constant(prices_B)  # Add intercept
model = OLS(prices_A, X).fit()
beta = model.params[1]
```

#### 4. Z-Score Calculation

Standardized measure of spread deviation:

```
Z-Score = (Current_Spread - Mean_Spread) / Std_Dev_Spread
```

**Interpretation**:
- `Z = 0`: Spread at historical mean
- `Z = +2`: Spread 2 standard deviations above mean
- `Z = -2`: Spread 2 standard deviations below mean

**Properties**:
- Dimensionless (comparable across different pairs)
- Indicates statistical significance of deviation
- Mean-reverting under cointegration

---

## 📈 Analytics Explained

### Core Metrics

#### 1. **Z-Score** (Primary Trading Signal)

**Formula**:
```
Z = (S_t - μ_S) / σ_S

Where:
S_t = Current spread
μ_S = Historical mean of spread
σ_S = Standard deviation of spread
```

**Usage**:
- **Entry Signal**: |Z| > 2.0 (configurable)
- **Exit Signal**: |Z| < 0.5 (configurable)
- **Stop Loss**: |Z| > 3.0 (configurable)

**Example**:
```
If Z = -2.5:
  → Spread is 2.5 standard deviations below mean
  → Asset A is underpriced relative to Asset B
  → Signal: LONG Asset A, SHORT Asset B
```

---

#### 2. **Hedge Ratio (β)** (Position Sizing)

**Calculation**: OLS regression coefficient

**Purpose**:
- Determines the quantity ratio between assets
- Neutralizes directional market risk
- Creates market-neutral position

**Example**:
```
If β = 0.065:
  → For every 1 BTC, trade 15.38 ETH (1/0.065)
  → Position: Long $10,000 BTC, Short $10,000 ETH
```

**Stability Check**:
```
Stability = (1 - σ_β / μ_β) × 100%

Good: > 95% (low variance in beta)
Moderate: 85-95%
Poor: < 85% (unstable relationship)
```

---

#### 3. **Rolling Correlation** (Relationship Strength)

**Formula**:
```
ρ(A,B) = Cov(A,B) / (σ_A × σ_B)

Calculated over rolling window (default: 50 periods)
```

**Interpretation**:
- **> 0.8**: Strong positive correlation (ideal for pairs trading)
- **0.6 - 0.8**: Moderate correlation (acceptable)
- **< 0.6**: Weak correlation (avoid trading)

**Why It Matters**:
- Low correlation → Higher risk of spread divergence
- High correlation → Better mean reversion properties

---

#### 4. **ADF Test** (Cointegration)

**Augmented Dickey-Fuller Test** tests for mean reversion:

**Null Hypothesis**: Spread has unit root (non-stationary)
**Alternative**: Spread is stationary (mean-reverting)

**Test Statistic**:
```
ΔS_t = α + γ×S_{t-1} + Σβ_i×ΔS_{t-i} + ε_t

Where:
γ < 0 indicates mean reversion
```

**Interpretation**:
- **p-value < 0.01**: Strong cointegration (excellent)
- **p-value < 0.05**: Cointegration (good)
- **p-value > 0.05**: No cointegration (risky)

**Example**:
```
ADF p-value = 0.02:
  → 98% confidence spread is mean-reverting
  → Suitable for pairs trading
  → Expected spread will return to mean
```

---

#### 5. **Half-Life** (Mean Reversion Speed)

**Definition**: Time for spread to revert halfway to mean

**Calculation**:
```python
# Fit AR(1) model: ΔS_t = λ×S_{t-1} + ε
model = OLS(spread_diff, spread_lag).fit()
lambda_param = model.params[1]

# Half-life formula
half_life = -log(2) / lambda_param
```

**Interpretation**:
- **< 10 periods**: Fast mean reversion (ideal for short-term trading)
- **10-50 periods**: Moderate speed (suitable)
- **> 50 periods**: Slow reversion (longer holding periods)

**Trading Implications**:
```
Fast Half-Life (5 periods):
  → Hold positions for hours
  → Frequent trading opportunities
  → Lower risk of prolonged drawdowns

Slow Half-Life (100 periods):
  → Hold positions for days
  → Fewer signals
  → Higher capital requirements
```

---

#### 6. **Sharpe Ratio** (Risk-Adjusted Returns)

**Formula**:
```
Sharpe = (μ_returns / σ_returns) × √(periods_per_year)

Where:
μ_returns = Mean strategy returns
σ_returns = Standard deviation of returns
```

**Interpretation**:
- **> 2.0**: Excellent risk-adjusted performance
- **1.0 - 2.0**: Good performance
- **< 1.0**: Poor risk-adjusted returns

**Example**:
```
Sharpe = 1.8:
  → For every unit of risk, earn 1.8 units of return
  → Better than most traditional strategies
```

---

#### 7. **Volatility** (Risk Measure)

**Formula**:
```
σ = √(Σ(r_i - μ)² / n)

Where:
r_i = Period returns
μ = Mean return
```

**Usage**:
- Position sizing adjustment
- Risk management
- Stop-loss calibration

**Rolling Window**: 50 periods (default)

---

#### 8. **Bid-Ask Spread Estimate** (Transaction Costs)

**Estimation**:
```
Bid_Ask ≈ 2 × σ_recent_prices

Using 10 most recent ticks
```

**Purpose**:
- Estimate transaction costs
- Adjust profit targets
- Filter low-profit signals

---

### Advanced Analytics

#### Signal Confidence Score

**Formula**:
```
Confidence = min(|Z-Score| / Entry_Threshold, 2.0) × Correlation

Where:
- Z-Score strength: Higher deviation = Higher confidence
- Correlation factor: Validates relationship strength
- Capped at 2.0 to prevent overconfidence
```

**Example**:
```
Z-Score = -3.0, Entry Threshold = 2.0, Correlation = 0.85

Confidence = min(3.0/2.0, 2.0) × 0.85
           = 1.5 × 0.85
           = 1.275 (127.5%)

Interpretation: High confidence signal
```

---

## 🎮 Usage Guide

### Basic Workflow

1. **Start the Platform**:
```bash
streamlit run app.py
```

2. **Configure Settings** (Sidebar):
   - Select trading pairs
   - Set analytics intervals
   - Configure trading thresholds

3. **Monitor Dashboard**:
   - Watch live metrics
   - Review trading signals
   - Check cointegration status

4. **Set Alerts**:
   - Define custom thresholds
   - Get notified of opportunities

5. **Export Data**:
   - Download ticks, analytics, or signals
   - Perform custom analysis

### Interpreting Signals

#### LONG SPREAD Signal
```
🟢 LONG SPREAD
Action: Buy Asset A, Sell Asset B
When: Z-Score < -2.0 (spread below mean)
Logic: Spread will increase back to mean
```

#### SHORT SPREAD Signal
```
🔴 SHORT SPREAD
Action: Sell Asset A, Buy Asset B
When: Z-Score > +2.0 (spread above mean)
Logic: Spread will decrease back to mean
```

### Risk Indicators

```
Risk Level Assessment:
├── 🟢 LOW    : |Z-Score| < Entry Threshold
├── 🟡 MEDIUM : Entry < |Z-Score| < Stop Loss
└── 🔴 HIGH   : |Z-Score| > Stop Loss Threshold

Correlation Status:
├── ✅ Strong    : > 0.8
├── ⚠️ Moderate : 0.6 - 0.8
└── ❌ Weak     : < 0.6
```

---

## 🎯 Trading Strategy

### Entry Conditions

**Must satisfy ALL**:
1. ✅ |Z-Score| > Entry Threshold (default: 2.0)
2. ✅ Correlation > 0.7
3. ✅ ADF p-value < 0.05 (cointegrated)
4. ✅ Stable hedge ratio (low variance)

### Position Sizing

```
Position Size = Capital × (Confidence / Max_Confidence)

Example:
Capital = $10,000
Confidence = 140%
Max Confidence = 200%

Position = $10,000 × (1.4 / 2.0) = $7,000
```

### Exit Conditions

**Any of**:
1. ✅ |Z-Score| < Exit Threshold (profit target)
2. ❌ |Z-Score| > Stop Loss Threshold (cut losses)
3. ⚠️ Correlation drops < 0.6 (relationship breakdown)
4. ⏱️ Half-life significantly increases (slower reversion)

### Example Trade Flow

```
Entry:
├── Time: 10:00:00
├── Z-Score: -2.3
├── Action: LONG BTC, SHORT ETH
├── Prices: BTC=$50,000, ETH=$3,000
├── Hedge Ratio: 0.060
├── Position: Long $5,000 BTC, Short $5,000 ETH
└── Confidence: 132%

Exit:
├── Time: 11:45:00
├── Z-Score: -0.3 (crossed exit threshold)
├── Prices: BTC=$50,200, ETH=$3,010
├── P&L: +$100 (after spread convergence)
└── Holding Period: 1h 45m
```

---

## ⚙️ Configuration

### Trading Parameters

```python
# In sidebar UI or modify defaults in app.py

DEFAULT_ENTRY_ZSCORE = 2.0      # Higher = More selective
DEFAULT_EXIT_ZSCORE = 0.5        # Lower = Earlier exits
DEFAULT_STOP_LOSS_ZSCORE = 3.0   # Risk tolerance
```

### Analytics Intervals

```python
DEFAULT_FAST_INTERVAL = 0.5      # Fast analytics (seconds)
DEFAULT_SLOW_INTERVAL = 30       # Slow analytics (seconds)
MAX_TICKS_TO_READ = 5000         # Historical data window
```

### Resample Rules

```python
# Aggregation periods for slow analytics
"None"  # No resampling (tick-level)
"1T"    # 1 minute bars
"5T"    # 5 minute bars
"15T"   # 15 minute bars
```

---

## 🗄️ Database Schema

### Tables Structure

#### 1. `ticks`
```sql
CREATE TABLE ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,           -- e.g., 'btcusdt'
    price REAL,            -- Trade price
    quantity REAL,         -- Trade volume
    trade_time INTEGER     -- Unix timestamp (ms)
);
CREATE INDEX idx_symbol_time ON ticks(symbol, trade_time);
```

#### 2. `analytics`
```sql
CREATE TABLE analytics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    resolution TEXT,       -- 'tick', '1T', '5T', etc.
    hedge_ratio REAL,      -- β from OLS
    spread_mean REAL,
    spread_std REAL,
    zscore REAL,
    rolling_corr REAL,
    volatility REAL,
    adf_p REAL,           -- ADF test p-value
    half_life REAL,       -- Mean reversion speed
    sharpe_ratio REAL,    -- Risk-adjusted returns
    bid_ask_spread REAL   -- Transaction cost estimate
);
```

#### 3. `trade_signals`
```sql
CREATE TABLE trade_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    signal_type TEXT,      -- 'LONG_SPREAD' or 'SHORT_SPREAD'
    entry_price_a REAL,
    entry_price_b REAL,
    zscore REAL,
    hedge_ratio REAL,
    confidence REAL
);
```

#### 4. `alerts`
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric TEXT,           -- e.g., 'zscore', 'rolling_corr'
    operator TEXT,         -- '>', '<', '>=', '<=', '==', '!='
    threshold REAL,
    active INTEGER,        -- 1 = active, 0 = disabled
    created_at INTEGER
);
```

#### 5. `alert_events`
```sql
CREATE TABLE alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER,      -- Foreign key to alerts
    metric TEXT,
    value REAL,
    occurred_at INTEGER
);
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Database Locked Error
```
Error: database is locked
```

**Solution**:
- Stop the app
- Delete `ticks.db` and `ticks.db-wal` files
- Restart the app
- Platform uses WAL mode to minimize locks

#### 2. No Data Appearing
```
Warning: Waiting for market data...
```

**Causes**:
- Network issues
- Binance API restrictions
- WebSocket connection failed

**Solution**:
- Check internet connection
- Verify Binance is accessible
- Check terminal for WebSocket errors

#### 3. Missing Columns Error
```
KeyError: 'half_life'
```

**Solution**:
- Delete old database: `rm ticks.db`
- Fixed in latest version with column existence checks

#### 4. High Memory Usage

**Causes**:
- Too much historical data in memory
- Long running sessions

**Solution**:
- Reduce `MAX_TICKS_TO_READ` in config
- Restart app periodically
- Clear old database periodically

### Performance Optimization

```python
# Adjust these for performance vs accuracy trade-offs

MAX_TICKS_TO_READ = 5000    # Lower = Less memory, faster queries
DEFAULT_FAST_INTERVAL = 1.0  # Higher = Less CPU usage
DEFAULT_SLOW_INTERVAL = 60   # Higher = Less frequent ADF tests
```

---

## 📚 Further Reading

### Statistical Concepts
- [Cointegration and Pairs Trading](https://www.investopedia.com/articles/trading/04/090804.asp)
- [Mean Reversion Strategies](https://www.quantstart.com/articles/Mean-Reversion-Trading-Strategy/)
- [Augmented Dickey-Fuller Test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)

### Technical Implementation
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Binance WebSocket API](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## ⚠️ Disclaimer

### Important Notices

1. **Educational Purpose Only**
   - This platform is for learning and research
   - Not financial advice
   - Not intended for production trading without extensive testing

2. **No Warranty**
   - Software provided "as is"
   - No guarantee of profitability
   - Past performance ≠ future results

3. **Risk Warning**
   - Cryptocurrency trading is highly risky
   - You can lose your entire investment
   - Only trade with capital you can afford to lose
   - Consult financial advisors before trading

4. **Technical Limitations**
   - Does not execute real trades
   - Ignores transaction costs and slippage
   - Network latency affects real-world performance
   - Market conditions change rapidly

5. **Regulatory Compliance**
   - Check local regulations before trading
   - Some jurisdictions restrict cryptocurrency trading
   - You are responsible for tax obligations

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

---

## 📧 Contact & Support

- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions
- **Email**: learningdsiiit@gmail.com


---

Made with ❤️ for quantitative trading enthusiasts
