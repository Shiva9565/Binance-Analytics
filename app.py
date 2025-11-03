# app.py - Enhanced Trading Analytics Platform
import websocket
import json
import sqlite3
import threading
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
from contextlib import contextmanager
from scipy import stats

# =====================
# CONFIG & DEFAULTS
# =====================
DB_FILE = "ticks.db"
DEFAULT_SYMBOLS = ["btcusdt", "ethusdt"]
DEFAULT_FAST_INTERVAL = 0.5
DEFAULT_SLOW_INTERVAL = 30
MAX_TICKS_TO_READ = 5000
DB_TIMEOUT = 30.0

# Trading thresholds (configurable by user)
DEFAULT_ENTRY_ZSCORE = 2.0
DEFAULT_EXIT_ZSCORE = 0.5
DEFAULT_STOP_LOSS_ZSCORE = 3.0

# =====================
# DATABASE HELPERS
# =====================
@contextmanager
def get_db_connection(timeout=DB_TIMEOUT):
    """Context manager for database connections with proper timeout and locking"""
    conn = None
    max_retries = 5
    retry_delay = 0.2
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_FILE, timeout=timeout, check_same_thread=False, isolation_level=None)
            result = conn.execute("PRAGMA journal_mode").fetchone()
            if result[0].upper() != 'WAL':
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            break
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

def execute_with_retry(func, max_retries=5, delay=0.2):
    """Execute database operation with retry logic"""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise
    raise sqlite3.OperationalError("Max retries exceeded")

# =====================
# DATABASE SETUP
# =====================
def init_db():
    """Initialize database with retry logic"""
    max_init_retries = 10
    for attempt in range(max_init_retries):
        try:
            conn = sqlite3.connect(DB_FILE, timeout=60.0, check_same_thread=False, isolation_level=None)
            cursor = conn.cursor()
            
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=60000")
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                price REAL,
                quantity REAL,
                trade_time INTEGER
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON ticks(symbol, trade_time)")
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                resolution TEXT,
                hedge_ratio REAL,
                spread_mean REAL,
                spread_std REAL,
                zscore REAL,
                rolling_corr REAL,
                volatility REAL,
                adf_p REAL,
                half_life REAL,
                sharpe_ratio REAL,
                bid_ask_spread REAL
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_time ON analytics(timestamp, resolution)")
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT,
                operator TEXT,
                threshold REAL,
                active INTEGER DEFAULT 1,
                created_at INTEGER
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER,
                metric TEXT,
                value REAL,
                occurred_at INTEGER
            )
            """)
            
            # Trade signals table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                signal_type TEXT,
                entry_price_a REAL,
                entry_price_b REAL,
                zscore REAL,
                hedge_ratio REAL,
                confidence REAL
            )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_time ON trade_signals(timestamp)")
            
            conn.commit()
            conn.close()
            return True
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_init_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    raise sqlite3.OperationalError("Could not initialize database after multiple retries")

if 'db_initialized' not in st.session_state:
    try:
        init_db()
        st.session_state['db_initialized'] = True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        st.info("Try restarting the Streamlit app or deleting the ticks.db file")
        st.stop()

# =====================
# HELPERS
# =====================
def now_ms():
    return int(time.time() * 1000)

def calculate_half_life(spread):
    """Calculate mean reversion half-life using OLS on lagged spread"""
    try:
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        if len(spread_lag) < 10:
            return None
            
        # Align the series
        spread_lag = spread_lag.iloc[1:]
        spread_diff = spread_diff.iloc[1:]
        
        X = add_constant(spread_lag)
        model = OLS(spread_diff, X).fit()
        
        lambda_param = model.params[1]
        if lambda_param < 0:
            half_life = -np.log(2) / lambda_param
            return float(half_life) if half_life > 0 and half_life < 1000 else None
        return None
    except:
        return None

def calculate_sharpe_ratio(returns, periods_per_year=252*24*60):
    """Calculate annualized Sharpe ratio"""
    try:
        if len(returns) < 2:
            return None
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return == 0:
            return None
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return float(sharpe)
    except:
        return None

OPS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

# =====================
# WEBSOCKET INGESTION
# =====================
def run_ws(symbol):
    """WebSocket ingestion with error handling"""
    def insert_tick(data):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            symbol_lower = data.get('s', '').lower()
            price = float(data.get('p', 0))
            quantity = float(data.get('q', 0))
            trade_time = int(data.get('T', 0))
            
            cursor.execute(
                "INSERT INTO ticks (symbol, price, quantity, trade_time) VALUES (?, ?, ?, ?)",
                (symbol_lower, price, quantity, trade_time)
            )
            conn.commit()

    def on_message(ws, message):
        try:
            data = json.loads(message)
            execute_with_retry(lambda: insert_tick(data))
        except Exception as e:
            print(f"ws ingestion error for {symbol}:", e)

    def on_error(ws, error):
        print(f"[{symbol}] WS Error:", error)

    def on_close(ws, close_status_code, close_msg):
        print(f"[{symbol}] WebSocket closed")

    def on_open(ws):
        print(f"[{symbol}] Connected to Binance WebSocket")

    socket = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
    ws = websocket.WebSocketApp(socket, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(ping_interval=20, ping_timeout=10)

# =====================
# ANALYTICS THREADS
# =====================
def fast_analytics_loop(symbols, fast_interval, entry_z, exit_z, stop_z):
    """Enhanced fast analytics with trading signals"""
    while True:
        try:
            with get_db_connection() as conn:
                dfs = {}
                for s in symbols:
                    df = pd.read_sql_query(
                        "SELECT * FROM ticks WHERE symbol=? ORDER BY trade_time DESC LIMIT ?",
                        conn, params=(s, MAX_TICKS_TO_READ)
                    )
                    if not df.empty:
                        df['datetime'] = pd.to_datetime(df['trade_time'], unit='ms')
                        df.sort_values('datetime', inplace=True)
                        dfs[s] = df

                if len(dfs) < 2:
                    time.sleep(fast_interval)
                    continue

                a, b = symbols[0], symbols[1]
                df_a, df_b = dfs[a], dfs[b]
                min_len = min(len(df_a), len(df_b))
                if min_len < 30:
                    time.sleep(fast_interval)
                    continue

                df_a = df_a.tail(min_len).reset_index(drop=True)
                df_b = df_b.tail(min_len).reset_index(drop=True)

                y = df_a['price'].values
                x = add_constant(df_b['price'].values)
                model = OLS(y, x).fit()
                beta = float(model.params[1])

                spread = df_a['price'] - beta * df_b['price']
                spread_mean = float(spread.mean())
                spread_std = float(spread.std()) if spread.std() != 0 else 0.0
                zscore = float((spread.iloc[-1] - spread_mean) / spread_std) if spread_std != 0 else 0.0

                window = min(50, min_len)
                rolling_corr = float(df_a['price'].rolling(window).corr(df_b['price']).iloc[-1]) if min_len >= window else 0.0
                rolling_vol = float(df_a['price'].pct_change().rolling(window).std().iloc[-1]) if min_len >= window else 0.0
                
                # Calculate half-life
                half_life = calculate_half_life(spread)
                
                # Calculate Sharpe ratio
                spread_returns = spread.pct_change().dropna()
                sharpe = calculate_sharpe_ratio(spread_returns)
                
                # Bid-ask spread estimate (using recent price volatility)
                bid_ask = float(df_a['price'].iloc[-10:].std() * 2) if len(df_a) >= 10 else 0.0

            # Generate trading signals
            signal_type = None
            confidence = 0.0
            
            if abs(zscore) >= entry_z and rolling_corr > 0.7:
                if zscore > entry_z:
                    signal_type = 'SHORT_SPREAD'  # Short A, Long B
                    confidence = min(abs(zscore) / entry_z, 2.0) * rolling_corr
                elif zscore < -entry_z:
                    signal_type = 'LONG_SPREAD'   # Long A, Short B
                    confidence = min(abs(zscore) / entry_z, 2.0) * rolling_corr
                    
                if signal_type:
                    def insert_signal():
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT INTO trade_signals 
                                (timestamp, signal_type, entry_price_a, entry_price_b, zscore, hedge_ratio, confidence)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (now_ms(), signal_type, float(df_a['price'].iloc[-1]), 
                                  float(df_b['price'].iloc[-1]), zscore, beta, confidence))
                            conn.commit()
                    execute_with_retry(insert_signal)

            # Insert analytics
            def insert_analytics():
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO analytics 
                        (timestamp, resolution, hedge_ratio, spread_mean, spread_std, zscore,
                         rolling_corr, volatility, adf_p, half_life, sharpe_ratio, bid_ask_spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (now_ms(), 'tick', beta, spread_mean, spread_std, zscore, 
                          rolling_corr, rolling_vol, None, half_life, sharpe, bid_ask))
                    conn.commit()
            
            execute_with_retry(insert_analytics)
        except Exception as e:
            print("fast analytics error:", e)
        time.sleep(fast_interval)

def slow_analytics_loop(symbols, slow_interval, resample_rule=None):
    """Slow analytics with ADF test"""
    while True:
        try:
            with get_db_connection() as conn:
                dfs = {}
                for s in symbols:
                    df = pd.read_sql_query(
                        "SELECT * FROM ticks WHERE symbol=? ORDER BY trade_time DESC LIMIT ?",
                        conn, params=(s, MAX_TICKS_TO_READ)
                    )
                    if not df.empty:
                        df['datetime'] = pd.to_datetime(df['trade_time'], unit='ms')
                        df.sort_values('datetime', inplace=True)
                        dfs[s] = df

                if len(dfs) < 2:
                    time.sleep(slow_interval)
                    continue

                a, b = symbols[0], symbols[1]
                df_a, df_b = dfs[a], dfs[b]

                if resample_rule:
                    df_a = df_a.set_index('datetime').resample(resample_rule).agg({'price':'mean'}).dropna().reset_index()
                    df_b = df_b.set_index('datetime').resample(resample_rule).agg({'price':'mean'}).dropna().reset_index()

                if len(df_a) < 30 or len(df_b) < 30:
                    time.sleep(slow_interval)
                    continue

                min_len = min(len(df_a), len(df_b))
                df_a = df_a.tail(min_len).reset_index(drop=True)
                df_b = df_b.tail(min_len).reset_index(drop=True)

                y = df_a['price'].values
                x = add_constant(df_b['price'].values)
                model = OLS(y, x).fit()
                beta = float(model.params[1])

                spread = df_a['price'] - beta * df_b['price']
                spread_mean = float(spread.mean())
                spread_std = float(spread.std()) if spread.std() != 0 else 0.0
                zscore_last = float((spread.iloc[-1] - spread_mean) / spread_std) if spread_std != 0 else 0.0

                try:
                    adf_stat, pval, *_ = adfuller(spread.dropna(), maxlag=1)
                    adf_p = float(pval)
                except Exception:
                    adf_p = None

                window = min(50, min_len)
                rolling_corr = float(df_a['price'].rolling(window).corr(df_b['price']).iloc[-1]) if min_len >= window else 0.0
                rolling_vol = float(df_a['price'].pct_change().rolling(window).std().iloc[-1]) if min_len >= window else 0.0
                
                half_life = calculate_half_life(spread)
                spread_returns = spread.pct_change().dropna()
                sharpe = calculate_sharpe_ratio(spread_returns)
                bid_ask = float(df_a['price'].iloc[-10:].std() * 2) if len(df_a) >= 10 else 0.0

            def insert_analytics():
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    resolution_label = resample_rule if resample_rule else 'raw'
                    cursor.execute("""
                        INSERT INTO analytics 
                        (timestamp, resolution, hedge_ratio, spread_mean, spread_std, zscore,
                         rolling_corr, volatility, adf_p, half_life, sharpe_ratio, bid_ask_spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (now_ms(), resolution_label, beta, spread_mean, spread_std, zscore_last, 
                          rolling_corr, rolling_vol, adf_p, half_life, sharpe, bid_ask))
                    conn.commit()
            
            execute_with_retry(insert_analytics)
        except Exception as e:
            print("slow analytics error:", e)
        time.sleep(slow_interval)

def start_background_tasks(symbols, fast_interval, slow_interval, resample_rule, entry_z, exit_z, stop_z):
    for sym in symbols:
        t = threading.Thread(target=run_ws, args=(sym,), daemon=True)
        t.start()
        time.sleep(0.1)

    t_fast = threading.Thread(target=fast_analytics_loop, args=(symbols, fast_interval, entry_z, exit_z, stop_z), daemon=True)
    t_fast.start()

    t_slow = threading.Thread(target=slow_analytics_loop, args=(symbols, slow_interval, resample_rule), daemon=True)
    t_slow.start()

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Advanced Pairs Trading Platform", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .signal-long {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .signal-short {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced Pairs Trading Analytics Platform")
st.markdown("**Real-time statistical arbitrage with intelligent trade signals**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📊 Data Settings", expanded=True):
        symbols = st.multiselect("Trading Pairs", DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS)
        fast_interval = st.number_input("Fast Analytics (s)", 0.1, 5.0, DEFAULT_FAST_INTERVAL, 0.1)
        slow_interval = st.number_input("Slow Analytics (s)", 5, 300, DEFAULT_SLOW_INTERVAL, 1)
        resample_choice = st.selectbox("Resample Rule", ["None", "1T", "5T", "15T"], index=1)
        resample_rule = None if resample_choice == "None" else resample_choice
        refresh_interval = st.number_input("UI Refresh (s)", 0.5, 10.0, 2.0, 0.5)
    
    with st.expander("🎯 Trading Parameters", expanded=True):
        entry_zscore = st.number_input("Entry Z-Score", 0.5, 5.0, DEFAULT_ENTRY_ZSCORE, 0.1,
                                       help="Z-score threshold to enter a trade")
        exit_zscore = st.number_input("Exit Z-Score", 0.1, 2.0, DEFAULT_EXIT_ZSCORE, 0.1,
                                      help="Z-score threshold to exit a trade")
        stop_loss_zscore = st.number_input("Stop Loss Z-Score", 2.0, 10.0, DEFAULT_STOP_LOSS_ZSCORE, 0.5,
                                           help="Z-score threshold to cut losses")
        
        st.info(f"📈 **Strategy**: Enter when |Z| > {entry_zscore}, Exit when |Z| < {exit_zscore}")

# Initialize background tasks
if "started" not in st.session_state:
    if len(symbols) >= 2:
        start_background_tasks(symbols, fast_interval, slow_interval, resample_rule, 
                              entry_zscore, exit_zscore, stop_loss_zscore)
        st.session_state["started"] = True
        st.success("✅ Live data ingestion and analytics started!")
    else:
        st.warning("⚠️ Select at least 2 symbols to start")

# Alerts management in sidebar
with st.sidebar:
    st.header("🔔 Alert Management")
    with st.form("alert_form", clear_on_submit=True):
        metric = st.selectbox("Metric", ["zscore", "rolling_corr", "volatility", "hedge_ratio", "adf_p", "half_life"])
        operator = st.selectbox("Operator", [">", "<", ">=", "<="])
        threshold = st.number_input("Threshold", value=2.0)
        if st.form_submit_button("Create Alert"):
            def create_alert():
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO alerts (metric, operator, threshold, active, created_at) VALUES (?, ?, ?, ?, ?)",
                              (metric, operator, float(threshold), 1, now_ms()))
                    conn.commit()
            execute_with_retry(create_alert)
            st.success(f"✅ Alert: {metric} {operator} {threshold}")

    # Show active alerts
    with get_db_connection() as conn:
        alerts_df = pd.read_sql_query("SELECT * FROM alerts WHERE active=1 ORDER BY created_at DESC", conn)

    if not alerts_df.empty:
        st.markdown("**Active Alerts**")
        for idx, row in alerts_df.iterrows():
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(f"{int(row['id'])}: {row['metric']} {row['operator']} {row['threshold']}")
            with cols[1]:
                if st.button("❌", key=f"disable_{int(row['id'])}", help="Disable"):
                    def disable_alert(alert_id):
                        conn = sqlite3.connect(DB_FILE, timeout=60.0, check_same_thread=False, isolation_level=None)
                        try:
                            cursor = conn.cursor()
                            cursor.execute("UPDATE alerts SET active=0 WHERE id=?", (alert_id,))
                            conn.commit()
                        finally:
                            conn.close()
                    execute_with_retry(lambda: disable_alert(int(row['id'])))
                    time.sleep(0.3)
                    st.rerun()
    else:
        st.info("No active alerts")

# Load data
with get_db_connection() as conn:
    df_ticks = pd.read_sql_query("SELECT * FROM ticks ORDER BY trade_time DESC LIMIT 2000", conn)
    df_analytics = pd.read_sql_query("SELECT * FROM analytics ORDER BY timestamp DESC LIMIT 300", conn)
    active_alerts = pd.read_sql_query("SELECT * FROM alerts WHERE active=1", conn)
    df_signals = pd.read_sql_query("SELECT * FROM trade_signals ORDER BY timestamp DESC LIMIT 50", conn)

if df_ticks.empty:
    st.warning("⏳ Waiting for market data...")
    time.sleep(2)
    st.rerun()

df_ticks["datetime"] = pd.to_datetime(df_ticks["trade_time"], unit="ms")
if not df_analytics.empty:
    df_analytics["datetime"] = pd.to_datetime(df_analytics["timestamp"], unit="ms")
if not df_signals.empty:
    df_signals["datetime"] = pd.to_datetime(df_signals["timestamp"], unit="ms")

# =====================
# TRADING DASHBOARD
# =====================

# Row 1: Key Metrics & Trading Signals
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("📊 Live Market Data")
    fast_rows = df_analytics[df_analytics['resolution'] == 'tick']
    if not fast_rows.empty:
        latest = fast_rows.iloc[0]
        
        # Create metrics in columns
        m1, m2, m3 = st.columns(3)
        m1.metric("Z-Score", f"{latest['zscore']:.2f}", 
                  help="Current spread deviation from mean")
        m2.metric("Correlation", f"{latest['rolling_corr']:.3f}",
                  help="50-period rolling correlation")
        m3.metric("Hedge Ratio (β)", f"{latest['hedge_ratio']:.4f}",
                  help="Optimal hedge ratio from OLS")
        
        # Additional metrics
        m4, m5, m6 = st.columns(3)
        if 'half_life' in latest.index and pd.notna(latest['half_life']):
            m4.metric("Half-Life", f"{latest['half_life']:.1f} periods",
                     help="Mean reversion speed")
        else:
            m4.metric("Half-Life", "N/A")
            
        if 'sharpe_ratio' in latest.index and pd.notna(latest['sharpe_ratio']):
            m5.metric("Sharpe Ratio", f"{latest['sharpe_ratio']:.2f}",
                     help="Risk-adjusted returns")
        else:
            m5.metric("Sharpe Ratio", "N/A")
            
        m6.metric("Volatility", f"{latest['volatility']*100:.3f}%",
                 help="Rolling volatility")

with col2:
    st.subheader("🎯 Trading Signals")
    if not df_signals.empty and len(df_signals) > 0:
        latest_signal = df_signals.iloc[0]
        time_ago = (datetime.now() - latest_signal['datetime']).total_seconds() / 60
        
        if latest_signal['signal_type'] == 'LONG_SPREAD':
            st.markdown(f"""
            <div class="signal-long">
                <h4>🟢 LONG SPREAD Signal</h4>
                <p><b>Action:</b> Long {symbols[0].upper()} @ {latest_signal['entry_price_a']:.2f} | 
                Short {symbols[1].upper()} @ {latest_signal['entry_price_b']:.2f}</p>
                <p><b>Z-Score:</b> {latest_signal['zscore']:.2f} | 
                <b>Confidence:</b> {latest_signal['confidence']:.1%} | 
                <b>Time:</b> {time_ago:.1f}m ago</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="signal-short">
                <h4>🔴 SHORT SPREAD Signal</h4>
                <p><b>Action:</b> Short {symbols[0].upper()} @ {latest_signal['entry_price_a']:.2f} | 
                Long {symbols[1].upper()} @ {latest_signal['entry_price_b']:.2f}</p>
                <p><b>Z-Score:</b> {latest_signal['zscore']:.2f} | 
                <b>Confidence:</b> {latest_signal['confidence']:.1%} | 
                <b>Time:</b> {time_ago:.1f}m ago</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show recent signals
        st.markdown("**Recent Signals (Last 10)**")
        recent_signals = df_signals.head(10)[['datetime', 'signal_type', 'zscore', 'confidence']]
        recent_signals['confidence'] = recent_signals['confidence'].apply(lambda x: f"{x:.1%}")
        recent_signals['zscore'] = recent_signals['zscore'].apply(lambda x: f"{x:.2f}")
        st.dataframe(recent_signals, use_container_width=True, height=200)
    else:
        st.info("No trading signals generated yet. Waiting for entry conditions...")

with col3:
    st.subheader("⚠️ Risk Metrics")
    if not fast_rows.empty:
        latest = fast_rows.iloc[0]
        
        # Risk assessment
        zscore = abs(latest['zscore'])
        corr = latest['rolling_corr']
        
        if zscore < entry_zscore:
            risk_level = "🟢 LOW"
            risk_color = "green"
        elif zscore < stop_loss_zscore:
            risk_level = "🟡 MEDIUM"
            risk_color = "orange"
        else:
            risk_level = "🔴 HIGH"
            risk_color = "red"
        
        st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                   unsafe_allow_html=True)
        
        # Correlation strength
        if corr > 0.8:
            st.success("✅ Strong Correlation")
        elif corr > 0.6:
            st.warning("⚠️ Moderate Correlation")
        else:
            st.error("❌ Weak Correlation")
        
        # ADF test result
        slow_rows = df_analytics[df_analytics['resolution'] == (resample_rule if resample_rule else 'raw')]
        if not slow_rows.empty and 'adf_p' in slow_rows.columns and pd.notna(slow_rows.iloc[0]['adf_p']):
            adf_p = slow_rows.iloc[0]['adf_p']
            if adf_p < 0.05:
                st.success(f"✅ Cointegrated (p={adf_p:.3f})")
            else:
                st.warning(f"⚠️ Not Cointegrated (p={adf_p:.3f})")
        
        # Spread width estimate
        if 'bid_ask_spread' in latest.index and pd.notna(latest['bid_ask_spread']):
            st.info(f"💰 Est. Spread: ${latest['bid_ask_spread']:.4f}")

# Check and display alerts
if not active_alerts.empty and not df_analytics.empty:
    latest_metrics = df_analytics.iloc[0].to_dict()
    for _, a in active_alerts.iterrows():
        m = a['metric']
        op = a['operator']
        thr = float(a['threshold'])
        if m in latest_metrics and latest_metrics[m] is not None and not pd.isna(latest_metrics[m]):
            val = float(latest_metrics[m])
            if OPS[op](val, thr):
                def log_alert():
                    with get_db_connection() as conn:
                        c = conn.cursor()
                        c.execute("INSERT INTO alert_events (alert_id, metric, value, occurred_at) VALUES (?, ?, ?, ?)",
                                  (int(a['id']), m, val, now_ms()))
                        conn.commit()
                execute_with_retry(log_alert)
                st.error(f"🚨 Alert #{int(a['id'])}: {m} {op} {thr} — Current: {val:.4f}")

st.divider()

# Row 2: Price Charts with Technical Overlays
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Price Action & Spread")
    
    # Create subplot with price and spread
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Asset Prices', 'Spread & Z-Score'),
        row_heights=[0.6, 0.4]
    )
    
    # Plot prices for both symbols
    for sym in symbols:
        sym_data = df_ticks[df_ticks['symbol'] == sym].tail(500)
        fig.add_trace(
            go.Scatter(x=sym_data['datetime'], y=sym_data['price'], 
                      name=sym.upper(), mode='lines'),
            row=1, col=1
        )
    
    # Calculate and plot spread
    if len(symbols) >= 2 and not fast_rows.empty:
        a, b = symbols[0], symbols[1]
        df_a = df_ticks[df_ticks['symbol'] == a].tail(500).reset_index(drop=True)
        df_b = df_ticks[df_ticks['symbol'] == b].tail(500).reset_index(drop=True)
        
        if len(df_a) > 0 and len(df_b) > 0:
            min_len = min(len(df_a), len(df_b))
            df_a = df_a.tail(min_len).reset_index(drop=True)
            df_b = df_b.tail(min_len).reset_index(drop=True)
            
            beta = latest['hedge_ratio']
            spread = df_a['price'] - beta * df_b['price']
            
            fig.add_trace(
                go.Scatter(x=df_a['datetime'], y=spread, 
                          name='Spread', line=dict(color='purple')),
                row=2, col=1
            )
            
            # Add mean line
            fig.add_hline(y=spread.mean(), line_dash="dash", line_color="gray", 
                         row=2, col=1, annotation_text="Mean")
            
            # Add entry/exit bands
            std = spread.std()
            fig.add_hline(y=spread.mean() + entry_zscore * std, line_dash="dot", 
                         line_color="red", row=2, col=1, annotation_text=f"+{entry_zscore}σ")
            fig.add_hline(y=spread.mean() - entry_zscore * std, line_dash="dot", 
                         line_color="green", row=2, col=1, annotation_text=f"-{entry_zscore}σ")
    
    fig.update_layout(height=600, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread Value", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True, key="price_spread_chart")

with col2:
    st.subheader("📊 Z-Score Evolution")
    
    tick_df = df_analytics[df_analytics['resolution'] == 'tick'].copy()
    if not tick_df.empty:
        tick_df = tick_df.sort_values('timestamp').tail(500)
        
        fig_z = go.Figure()
        
        # Z-score line
        fig_z.add_trace(go.Scatter(
            x=tick_df['datetime'], 
            y=tick_df['zscore'],
            name='Z-Score',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        
        # Entry thresholds
        fig_z.add_hline(y=entry_zscore, line_dash="dash", line_color="red", 
                       annotation_text=f"Entry (+{entry_zscore})")
        fig_z.add_hline(y=-entry_zscore, line_dash="dash", line_color="green",
                       annotation_text=f"Entry (-{entry_zscore})")
        fig_z.add_hline(y=0, line_color="gray", line_width=1)
        
        # Exit thresholds
        fig_z.add_hline(y=exit_zscore, line_dash="dot", line_color="orange",
                       annotation_text=f"Exit (+{exit_zscore})")
        fig_z.add_hline(y=-exit_zscore, line_dash="dot", line_color="orange",
                       annotation_text=f"Exit (-{exit_zscore})")
        
        # Stop loss thresholds
        fig_z.add_hline(y=stop_loss_zscore, line_dash="dashdot", line_color="darkred",
                       annotation_text=f"Stop Loss (+{stop_loss_zscore})")
        fig_z.add_hline(y=-stop_loss_zscore, line_dash="dashdot", line_color="darkred",
                       annotation_text=f"Stop Loss (-{stop_loss_zscore})")
        
        # Highlight extreme zones
        fig_z.add_hrect(y0=entry_zscore, y1=stop_loss_zscore, 
                       fillcolor="red", opacity=0.1, line_width=0)
        fig_z.add_hrect(y0=-stop_loss_zscore, y1=-entry_zscore, 
                       fillcolor="green", opacity=0.1, line_width=0)
        
        fig_z.update_layout(
            height=600,
            hovermode='x unified',
            yaxis_title="Z-Score",
            xaxis_title="Time"
        )
        
        st.plotly_chart(fig_z, use_container_width=True, key="zscore_chart")
    else:
        st.info("Waiting for z-score data...")

st.divider()

# Row 3: Statistical Analysis
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔄 Correlation Analysis")
    tick_df = df_analytics[df_analytics['resolution'] == 'tick'].copy()
    if not tick_df.empty:
        tick_df = tick_df.sort_values('timestamp').tail(200)
        
        fig_corr = px.line(tick_df, x='datetime', y='rolling_corr',
                          title='Rolling Correlation (50-period)')
        fig_corr.add_hline(y=0.8, line_dash="dash", line_color="green",
                          annotation_text="Strong (0.8)")
        fig_corr.add_hline(y=0.6, line_dash="dash", line_color="orange",
                          annotation_text="Moderate (0.6)")
        fig_corr.update_layout(height=300, showlegend=False)
        fig_corr.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_chart")
        
        # Correlation stats
        st.markdown(f"""
        **Statistics:**
        - Mean: {tick_df['rolling_corr'].mean():.3f}
        - Std Dev: {tick_df['rolling_corr'].std():.3f}
        - Current: {tick_df['rolling_corr'].iloc[-1]:.3f}
        """)

with col2:
    st.subheader("📉 Volatility Tracking")
    if not tick_df.empty:
        fig_vol = px.line(tick_df, x='datetime', y='volatility',
                         title='Rolling Volatility')
        fig_vol.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True, key="vol_chart")
        
        # Volatility stats
        vol_pct = tick_df['volatility'] * 100
        st.markdown(f"""
        **Statistics:**
        - Mean: {vol_pct.mean():.3f}%
        - Std Dev: {vol_pct.std():.3f}%
        - Current: {vol_pct.iloc[-1]:.3f}%
        """)

with col3:
    st.subheader("⏱️ Mean Reversion Speed")
    if not tick_df.empty and 'half_life' in tick_df.columns and tick_df['half_life'].notna().any():
        half_life_data = tick_df[tick_df['half_life'].notna()].tail(200)
        
        if not half_life_data.empty:
            fig_hl = px.line(half_life_data, x='datetime', y='half_life',
                            title='Half-Life (periods)')
            fig_hl.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_hl, use_container_width=True, key="halflife_chart")
            
            # Half-life interpretation
            avg_hl = half_life_data['half_life'].mean()
            if avg_hl < 10:
                st.success(f"✅ Fast mean reversion (~{avg_hl:.1f} periods)")
            elif avg_hl < 50:
                st.info(f"ℹ️ Moderate mean reversion (~{avg_hl:.1f} periods)")
            else:
                st.warning(f"⚠️ Slow mean reversion (~{avg_hl:.1f} periods)")
        else:
            st.info("Accumulating half-life data...")
    else:
        st.info("Calculating mean reversion speed...")

st.divider()

# Row 4: Performance & Signal History
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📜 Signal History & Performance")
    
    if not df_signals.empty:
        # Signal type distribution
        signal_counts = df_signals['signal_type'].value_counts()
        
        fig_dist = px.pie(values=signal_counts.values, names=signal_counts.index,
                         title='Signal Distribution',
                         color_discrete_map={'LONG_SPREAD': 'green', 'SHORT_SPREAD': 'red'})
        st.plotly_chart(fig_dist, use_container_width=True, key="signal_dist")
        
        # Signal confidence over time
        fig_conf = px.scatter(df_signals.tail(50), x='datetime', y='confidence',
                            color='signal_type', size='confidence',
                            title='Signal Confidence Over Time',
                            color_discrete_map={'LONG_SPREAD': 'green', 'SHORT_SPREAD': 'red'})
        st.plotly_chart(fig_conf, use_container_width=True, key="signal_confidence")
    else:
        st.info("No signals generated yet")

with col2:
    st.subheader("🎲 Hedge Ratio Stability")
    
    slow_df = df_analytics[df_analytics['resolution'] == (resample_rule if resample_rule else 'raw')].copy()
    if not slow_df.empty:
        slow_df = slow_df.sort_values('timestamp').tail(200)
        
        fig_beta = px.line(slow_df, x='datetime', y='hedge_ratio',
                          title=f'Hedge Ratio Evolution ({resample_rule or "raw"})')
        
        # Add mean line
        mean_beta = slow_df['hedge_ratio'].mean()
        fig_beta.add_hline(y=mean_beta, line_dash="dash", line_color="gray",
                          annotation_text=f"Mean: {mean_beta:.4f}")
        
        st.plotly_chart(fig_beta, use_container_width=True, key="beta_chart")
        
        # Beta statistics
        st.markdown(f"""
        **Hedge Ratio Statistics:**
        - Mean: {mean_beta:.4f}
        - Std Dev: {slow_df['hedge_ratio'].std():.4f}
        - Current: {slow_df['hedge_ratio'].iloc[-1]:.4f}
        - Stability: {(1 - slow_df['hedge_ratio'].std() / mean_beta) * 100:.1f}%
        """)
        
        if slow_df['hedge_ratio'].std() / mean_beta < 0.05:
            st.success("✅ Highly stable hedge ratio")
        elif slow_df['hedge_ratio'].std() / mean_beta < 0.15:
            st.info("ℹ️ Moderately stable hedge ratio")
        else:
            st.warning("⚠️ Unstable hedge ratio - review strategy")

st.divider()

# Row 5: Advanced Analytics & Export
tab1, tab2, tab3, tab4 = st.tabs(["📊 Detailed Stats", "📥 Data Export", "🔬 Cointegration", "💡 Trading Guide"])

with tab1:
    st.subheader("Detailed Statistical Metrics")
    
    if not df_analytics.empty:
        # Create comprehensive stats table
        latest_tick = df_analytics[df_analytics['resolution'] == 'tick'].iloc[0] if not df_analytics[df_analytics['resolution'] == 'tick'].empty else None
        latest_slow = df_analytics[df_analytics['resolution'] != 'tick'].iloc[0] if not df_analytics[df_analytics['resolution'] != 'tick'].empty else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fast Analytics (Tick-based)**")
            if latest_tick is not None:
                stats_df = pd.DataFrame({
                    'Metric': ['Z-Score', 'Hedge Ratio', 'Spread Mean', 'Spread Std', 
                              'Correlation', 'Volatility', 'Half-Life', 'Sharpe Ratio'],
                    'Value': [
                        f"{latest_tick['zscore']:.4f}",
                        f"{latest_tick['hedge_ratio']:.4f}",
                        f"{latest_tick['spread_mean']:.4f}",
                        f"{latest_tick['spread_std']:.4f}",
                        f"{latest_tick['rolling_corr']:.4f}",
                        f"{latest_tick['volatility']*100:.4f}%",
                        f"{latest_tick['half_life']:.2f}" if 'half_life' in latest_tick.index and pd.notna(latest_tick['half_life']) else "N/A",
                        f"{latest_tick['sharpe_ratio']:.2f}" if 'sharpe_ratio' in latest_tick.index and pd.notna(latest_tick['sharpe_ratio']) else "N/A"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown(f"**Slow Analytics ({resample_rule or 'raw'})**")
            if latest_slow is not None:
                stats_df2 = pd.DataFrame({
                    'Metric': ['Z-Score', 'Hedge Ratio', 'ADF p-value', 'Correlation', 
                              'Volatility', 'Half-Life', 'Sharpe Ratio'],
                    'Value': [
                        f"{latest_slow['zscore']:.4f}",
                        f"{latest_slow['hedge_ratio']:.4f}",
                        f"{latest_slow['adf_p']:.4f}" if 'adf_p' in latest_slow.index and pd.notna(latest_slow['adf_p']) else "N/A",
                        f"{latest_slow['rolling_corr']:.4f}",
                        f"{latest_slow['volatility']*100:.4f}%",
                        f"{latest_slow['half_life']:.2f}" if 'half_life' in latest_slow.index and pd.notna(latest_slow['half_life']) else "N/A",
                        f"{latest_slow['sharpe_ratio']:.2f}" if 'sharpe_ratio' in latest_slow.index and pd.notna(latest_slow['sharpe_ratio']) else "N/A"
                    ]
                })
                st.dataframe(stats_df2, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not df_ticks.empty:
            csv_ticks = df_ticks.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📄 Download Raw Ticks",
                csv_ticks,
                "ticks.csv",
                "text/csv",
                key="download_ticks"
            )
            st.caption(f"{len(df_ticks)} tick records")
    
    with col2:
        if not df_analytics.empty:
            csv_analytics = df_analytics.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Analytics",
                csv_analytics,
                "analytics.csv",
                "text/csv",
                key="download_analytics"
            )
            st.caption(f"{len(df_analytics)} analytics records")
    
    with col3:
        if not df_signals.empty:
            csv_signals = df_signals.to_csv(index=False).encode('utf-8')
            st.download_button(
                "🎯 Download Signals",
                csv_signals,
                "signals.csv",
                "text/csv",
                key="download_signals"
            )
            st.caption(f"{len(df_signals)} signal records")

with tab3:
    st.subheader("Cointegration Analysis")
    
    st.markdown("""
    **Understanding Cointegration:**
    - Cointegration tests if two assets have a long-term equilibrium relationship
    - We use the Augmented Dickey-Fuller (ADF) test on the spread
    - **p-value < 0.05** suggests cointegration (good for pairs trading)
    - Lower p-values indicate stronger mean-reversion properties
    """)
    
    if not slow_df.empty and 'adf_p' in slow_df.columns and slow_df['adf_p'].notna().any():
        adf_data = slow_df[slow_df['adf_p'].notna()].tail(100)
        
        fig_adf = px.line(adf_data, x='datetime', y='adf_p',
                         title='ADF p-value Over Time (Lower is Better)')
        fig_adf.add_hline(y=0.05, line_dash="dash", line_color="red",
                         annotation_text="Significance Level (0.05)")
        fig_adf.add_hrect(y0=0, y1=0.05, fillcolor="green", opacity=0.1,
                         annotation_text="Cointegrated Zone", annotation_position="top right")
        st.plotly_chart(fig_adf, use_container_width=True, key="adf_chart")
        
        # Current status
        current_adf = adf_data['adf_p'].iloc[-1]
        if current_adf < 0.01:
            st.success(f"✅ **Strongly Cointegrated** (p={current_adf:.4f})")
            st.info("Excellent pairs trading opportunity!")
        elif current_adf < 0.05:
            st.success(f"✅ **Cointegrated** (p={current_adf:.4f})")
            st.info("Good pairs trading opportunity")
        else:
            st.warning(f"⚠️ **Not Cointegrated** (p={current_adf:.4f})")
            st.info("Pairs may not have stable long-term relationship")
    else:
        st.info("Calculating cointegration... This requires more historical data.")

with tab4:
    st.subheader("📚 Pairs Trading Strategy Guide")
    
    st.markdown(f"""
    ### How This Platform Works
    
    #### 1️⃣ **Spread Calculation**
    - Spread = Price_A - β × Price_B
    - β (hedge ratio) is calculated using Ordinary Least Squares regression
    - Z-score = (Current Spread - Mean Spread) / Std Dev
    
    #### 2️⃣ **Entry Signals** (|Z-Score| > {entry_zscore})
    - **LONG SPREAD**: When Z-score < -{entry_zscore}
      - Action: Buy {symbols[0].upper()}, Sell {symbols[1].upper()}
      - Expectation: Spread will increase back to mean
    
    - **SHORT SPREAD**: When Z-score > {entry_zscore}
      - Action: Sell {symbols[0].upper()}, Buy {symbols[1].upper()}
      - Expectation: Spread will decrease back to mean
    
    #### 3️⃣ **Exit Signals** (|Z-Score| < {exit_zscore})
    - Close positions when spread returns near the mean
    - Take profit as mean reversion occurs
    
    #### 4️⃣ **Risk Management**
    - **Stop Loss**: Exit if |Z-Score| > {stop_loss_zscore}
    - **Correlation Check**: Only trade when correlation > 0.7
    - **Half-Life**: Shorter half-life = faster mean reversion = better
    
    #### 5️⃣ **Key Metrics to Watch**
    - **Z-Score**: Primary trading signal
    - **Correlation**: Measures pair relationship strength
    - **ADF p-value**: Tests for cointegration (< 0.05 is good)
    - **Half-Life**: Speed of mean reversion
    - **Sharpe Ratio**: Risk-adjusted returns
    - **Hedge Ratio Stability**: Consistent β is preferred
    
    ### ⚠️ Important Disclaimers
    - This is for **educational purposes only**
    - Past performance doesn't guarantee future results
    - Always consider transaction costs and slippage
    - Test strategies thoroughly before live trading
    - Never risk more than you can afford to lose
    
    ### 💡 Pro Tips
    1. Monitor multiple timeframes (tick vs resampled)
    2. Wait for high confidence signals (confidence > 70%)
    3. Check cointegration regularly
    4. Adjust thresholds based on market conditions
    5. Use stop losses religiously
    """)

st.divider()

# Footer with system status
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticks Collected", f"{len(df_ticks):,}")
col2.metric("Analytics Points", f"{len(df_analytics):,}")
col3.metric("Signals Generated", f"{len(df_signals):,}")
col4.metric("Active Alerts", f"{len(active_alerts)}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: {refresh_interval}s")

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()