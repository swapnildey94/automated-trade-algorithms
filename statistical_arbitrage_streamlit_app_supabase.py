import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px # Added for correlation plot and OLS visualization
import time
import importlib
import sys
import schedule
import threading
from supabase import create_client, Client
from datetime import datetime, timedelta
import queue

# --- Configuration (from notebook, can be adjusted in UI) ---
DEFAULT_CONFIG = {
    'timeframe': '1d', # Changed default to daily
    'lookback_period': 30, # days for hedge ratio/z-score calculation
    'entry_threshold': 1.1,
    'exit_threshold': 0.3,
    'final_exit_threshold': 0.1, # Not currently used in generate_trade_signals
    'stop_loss_threshold': 3.1,
    'trading_fee': 0.001,
    'slippage': 0.001,
    'start_date': (datetime.now() - timedelta(days=365)), # Extended default range for daily
    'end_date': datetime.now(),
    'api_base_url': 'https://api.india.delta.exchange'
}

# --- Supabase Configuration ---
SUPABASE_URL = "https://ehllfhxlxxhqjseowmkm.supabase.co"  # Replace with your Supabase URL
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVobGxmaHhseHhocWpzZW93bWttIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDYwODc1MjksImV4cCI6MjA2MTY2MzUyOX0.7zxBe7zL5wLnqT7U0hg6denx9Lq2pJJNzBP1v78Dz2Q"  # Replace with your Supabase API key
SUPABASE_TABLE = "crypto_session_trades_scan"

def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        return None

# --- API Functions (from original script) ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_products(api_base_url):
    """Fetch available products (tickers) from DeltaExchange API."""
    try:
        url = f"{api_base_url}/v2/products"
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        products = data.get('result', [])
        df_products = pd.DataFrame(products)
        # Filter for perpetual futures, adjust if other contract types are needed
        df_filtered = df_products[df_products['contract_type'].isin(['perpetual_futures'])]
        if not df_filtered.empty:
            columns = ['id', 'symbol', 'description', 'contract_type']
            # Ensure all selected columns exist before trying to select them
            existing_columns = [col for col in columns if col in df_filtered.columns]
            df_filtered = df_filtered[existing_columns]
        return df_filtered
    except Exception as e:
        st.error(f"Error fetching products: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600) # Cache candle data for 10 minutes
def get_historical_candles(api_base_url, product_id, product_symbol, timeframe, start_time_ts, end_time_ts):
    """Fetch historical OHLC candle data for a specific product."""
    try:
        url = f"{api_base_url}/v2/history/candles"
        headers = {'Accept': 'application/json'}
        params = {
            'resolution': timeframe,
            'symbol': product_symbol,
            'start': str(start_time_ts),
            'end': str(end_time_ts)
        }
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        candles = data.get('result', [])
        df_candles = pd.DataFrame(candles)
        if not df_candles.empty:
            df_candles = df_candles.rename(columns={
                'time': 'timestamp', 'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume'
            })
            df_candles['timestamp'] = pd.to_datetime(df_candles['timestamp'], unit='s')
            df_candles = df_candles.set_index('timestamp')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_candles.columns: # Check if column exists
                    df_candles[col] = pd.to_numeric(df_candles[col])
            df_candles = df_candles.sort_index()
        return df_candles
    except Exception as e:
        st.error(f"Error fetching historical candles for {product_symbol}: {e}")
        return pd.DataFrame()

# --- Statistical Arbitrage Algorithm Functions (from original script) ---
def calculate_hedge_ratio(data_to_process, lookback_period_days, timeframe_selection):
    """
    Calculates the rolling hedge ratio (beta) using OLS regression on log prices.
    lookback_period_days: Number of days to consider for each rolling calculation.
    timeframe_selection: The candle timeframe (e.g., '1h', '1d') to determine points per day.
    """
    points_per_day = {
        '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2,
        '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1, '1w': 1/7, 'D': 1, 'W': 1/7
    }
    # Ensure timeframe_selection is valid, default to 1 if not found (e.g. for '1d')
    lookback_points = int(lookback_period_days * points_per_day.get(timeframe_selection, 1)) # Default to 1 point per day if timeframe not in dict (e.g. '1D')
    if lookback_points < 2: lookback_points = 2 # Minimum points for regression

    data = data_to_process.copy()
    data['hedge_ratio'] = np.nan
    data['spread'] = np.nan
    # Store intercept for potential visualization, though not strictly used by current strategy logic
    data['ols_intercept'] = np.nan 

    for i in range(lookback_points, len(data)):
        # Slice data for the current lookback window
        y_series = data['log_price_a'].iloc[i-lookback_points:i]
        X_series = data['log_price_b'].iloc[i-lookback_points:i]
        
        # Ensure there's enough non-null data
        if y_series.isnull().any() or X_series.isnull().any() or len(y_series) < 2 or len(X_series) < 2:
            data.loc[data.index[i], 'hedge_ratio'] = np.nan 
            data.loc[data.index[i], 'ols_intercept'] = np.nan
            continue
            
        # Prepare data for OLS: y = log_price_a, X = const + log_price_b
        X_with_const = sm.add_constant(X_series, prepend=True)
        
        try:
            model = sm.OLS(y_series, X_with_const).fit()
            # Ensure model fitting was successful and parameters are available
            if len(model.params) > 1: # params are [const, beta]
                 intercept = model.params.iloc[0]
                 beta = model.params.iloc[1] # Beta is the hedge ratio
                 data.loc[data.index[i], 'ols_intercept'] = intercept
                 data.loc[data.index[i], 'hedge_ratio'] = beta
                 # Calculate spread: log_price_a - beta * log_price_b
                 data.loc[data.index[i], 'spread'] = data['log_price_a'].iloc[i] - beta * data['log_price_b'].iloc[i]
            else:
                data.loc[data.index[i], 'hedge_ratio'] = np.nan 
                data.loc[data.index[i], 'ols_intercept'] = np.nan
        except Exception: 
            data.loc[data.index[i], 'hedge_ratio'] = np.nan
            data.loc[data.index[i], 'ols_intercept'] = np.nan
    return data

def calculate_zscore(data_with_spread, lookback_period_days, timeframe_selection):
    """
    Calculates the Z-score of the spread.
    lookback_period_days: Number of days for rolling mean and std dev of the spread.
    timeframe_selection: The candle timeframe to determine points per day.
    """
    points_per_day = {
        '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2,
        '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1, '1w': 1/7, 'D': 1, 'W': 1/7
    }
    lookback_points = int(lookback_period_days * points_per_day.get(timeframe_selection, 1))
    if lookback_points < 2: lookback_points = 2 # Min periods for rolling calculation

    result_df = data_with_spread.copy()
    # Calculate rolling mean and standard deviation of the spread
    result_df['spread_ma'] = result_df['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).mean()
    result_df['spread_std'] = result_df['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).std()
    
    # Calculate Z-score: (spread - spread_ma) / spread_std
    result_df['zscore'] = (result_df['spread'] - result_df['spread_ma']) / result_df['spread_std']
    return result_df

def calculate_secondary_quantity(primary_quantity, hedge_ratio, primary_price, secondary_price, primary_is_asset_a):
    """Calculates the quantity of the secondary asset to achieve a hedge."""
    if primary_price == 0 or secondary_price == 0: return 0 
    
    primary_notional = primary_quantity * primary_price 

    if primary_is_asset_a:
        secondary_notional_target = primary_notional * hedge_ratio 
        secondary_asset_quantity = secondary_notional_target / secondary_price if secondary_price != 0 else 0
    else: 
        if hedge_ratio == 0: return 0 
        secondary_notional_target = primary_notional / hedge_ratio
        secondary_asset_quantity = secondary_notional_target / secondary_price if secondary_price != 0 else 0
    return abs(secondary_asset_quantity) 

def generate_trade_signals(data_df, config_params):
    """Generates trading signals based on Z-score thresholds."""
    signals = pd.DataFrame(index=data_df.index)
    signals['zscore'] = data_df['zscore']
    signals['signal'] = 0.0  
    signals['position'] = 0  

    entry_thresh = config_params['entry_threshold']
    exit_thresh = config_params['exit_threshold'] 
    stop_loss_thresh = config_params['stop_loss_threshold'] 
    current_pos = 0

    for i in range(1, len(signals)): 
        z = signals['zscore'].iloc[i]
        prev_z = signals['zscore'].iloc[i-1]

        if pd.isna(z) or pd.isna(prev_z): 
            signals.loc[signals.index[i], 'position'] = current_pos
            continue

        if current_pos == 0: 
            if prev_z >= -entry_thresh and z < -entry_thresh: 
                signals.loc[signals.index[i], 'signal'] = 1.0
                current_pos = 1
            elif prev_z <= entry_thresh and z > entry_thresh: 
                signals.loc[signals.index[i], 'signal'] = -1.0
                current_pos = -1
        elif current_pos == 1: 
            if (prev_z <= -exit_thresh and z > -exit_thresh) or \
               (prev_z >= -stop_loss_thresh and z < -stop_loss_thresh and stop_loss_thresh > entry_thresh): 
                signals.loc[signals.index[i], 'signal'] = 2.0 
                current_pos = 0
        elif current_pos == -1: 
            if (prev_z >= exit_thresh and z < exit_thresh) or \
               (prev_z <= stop_loss_thresh and z > stop_loss_thresh and stop_loss_thresh > entry_thresh): 
                signals.loc[signals.index[i], 'signal'] = 2.0 
                current_pos = 0
        
        signals.loc[signals.index[i], 'position'] = current_pos
        
    signals['buy_signal_z'] = np.where(signals['signal'] == 1.0, signals['zscore'], np.nan)
    signals['sell_signal_z'] = np.where(signals['signal'] == -1.0, signals['zscore'], np.nan)
    signals['exit_signal_z'] = np.where(signals['signal'] == 2.0, signals['zscore'], np.nan)
    return signals

def generate_trade_log(signals_df, data_with_prices_hedge_ratio, global_config, user_primary_quantity, primary_is_A_selected):
    """Generates a log of trades with P&L calculations."""
    trade_log_list = []
    active_trade = None
    slippage = global_config['slippage']
    trading_fee_rate = global_config['trading_fee']

    required_cols = ['close_a', 'close_b', 'hedge_ratio', 'symbol_a', 'symbol_b']
    if not all(col in data_with_prices_hedge_ratio.columns for col in required_cols):
        st.error(f"Missing required columns in data for trade log: {required_cols}")
        return pd.DataFrame()


    for timestamp, row in signals_df.iterrows():
        signal = row['signal']
        zscore_at_signal = row['zscore']

        if pd.isna(zscore_at_signal) or timestamp not in data_with_prices_hedge_ratio.index:
            continue

        current_price_a = data_with_prices_hedge_ratio.loc[timestamp, 'close_a']
        current_price_b = data_with_prices_hedge_ratio.loc[timestamp, 'close_b']
        current_hedge_ratio = data_with_prices_hedge_ratio.loc[timestamp, 'hedge_ratio']
        symbol_a = data_with_prices_hedge_ratio.loc[timestamp, 'symbol_a'] 
        symbol_b = data_with_prices_hedge_ratio.loc[timestamp, 'symbol_b'] 


        if active_trade is None: 
            if signal == 1.0 or signal == -1.0: 
                if pd.isna(current_hedge_ratio) or current_price_a == 0 or current_price_b == 0:
                    continue 
                
                qty_a, qty_b = 0, 0
                if primary_is_A_selected:
                    qty_a = user_primary_quantity
                    qty_b = calculate_secondary_quantity(qty_a, current_hedge_ratio, current_price_a, current_price_b, True)
                else: 
                    qty_b = user_primary_quantity
                    qty_a = calculate_secondary_quantity(qty_b, current_hedge_ratio, current_price_b, current_price_a, False) 
                
                if qty_a == 0 or qty_b == 0: continue 

                trade_type_str = f'Buy Spread (L {symbol_a}, S {symbol_b})' if signal == 1.0 else f'Sell Spread (S {symbol_a}, L {symbol_b})'
                active_trade = {
                    'Entry Timestamp': timestamp, 'Trade Type': trade_type_str,
                    'Entry Z-score': zscore_at_signal,
                    'Entry Price A (Orig)': current_price_a, 'Entry Price B (Orig)': current_price_b,
                    'Entry Hedge Ratio': current_hedge_ratio,
                    'Qty A': qty_a, 'Qty B': qty_b, 'Primary is A': primary_is_A_selected,
                    'Symbol A': symbol_a, 'Symbol B': symbol_b, 
                    'Exit Timestamp': None, 'Exit Price A (Orig)': None, 'Exit Price B (Orig)': None, 'Exit Z-score': None,
                    'PnL Asset A (USD)': None, 'PnL Asset B (USD)': None, 'Total Fees (USD)': None, 
                    'Net PnL (USD)': None, 'PnL % (Primary Notional)': None
                }
        elif active_trade is not None: 
            is_eod_closure = (timestamp == signals_df.index[-1] and active_trade['Exit Timestamp'] is None)
            if signal == 2.0 or is_eod_closure: 
                active_trade['Exit Timestamp'] = timestamp
                active_trade['Exit Price A (Orig)'] = current_price_a
                active_trade['Exit Price B (Orig)'] = current_price_b
                active_trade['Exit Z-score'] = zscore_at_signal if signal == 2.0 else data_with_prices_hedge_ratio.loc[timestamp, 'zscore'] 

                if is_eod_closure and signal != 2.0: 
                     active_trade['Trade Type'] += ' (Closed at EOD)'
                
                entry_p_a = active_trade['Entry Price A (Orig)']
                entry_p_b = active_trade['Entry Price B (Orig)']
                exit_p_a = active_trade['Exit Price A (Orig)']
                exit_p_b = active_trade['Exit Price B (Orig)']
                q_a = active_trade['Qty A']
                q_b = active_trade['Qty B']
                
                pnl_a_gross, pnl_b_gross, fee_a, fee_b = 0,0,0,0

                if 'Buy Spread' in active_trade['Trade Type']: 
                    eff_entry_a = entry_p_a * (1 + slippage) 
                    eff_exit_a = exit_p_a * (1 - slippage)   
                    pnl_a_gross = (eff_exit_a - eff_entry_a) * q_a
                    fee_a = (q_a * abs(eff_entry_a) + q_a * abs(eff_exit_a)) * trading_fee_rate
                    
                    eff_entry_b_sell = entry_p_b * (1 - slippage) 
                    eff_exit_b_buy = exit_p_b * (1 + slippage)   
                    pnl_b_gross = (eff_entry_b_sell - eff_exit_b_buy) * q_b
                    fee_b = (q_b * abs(eff_entry_b_sell) + q_b * abs(eff_exit_b_buy)) * trading_fee_rate
                elif 'Sell Spread' in active_trade['Trade Type']: 
                    eff_entry_a_sell = entry_p_a * (1 - slippage)
                    eff_exit_a_buy = exit_p_a * (1 + slippage)
                    pnl_a_gross = (eff_entry_a_sell - eff_exit_a_buy) * q_a
                    fee_a = (q_a * abs(eff_entry_a_sell) + q_a * abs(eff_exit_a_buy)) * trading_fee_rate

                    eff_entry_b = entry_p_b * (1 + slippage)
                    eff_exit_b = exit_p_b * (1 - slippage)
                    pnl_b_gross = (eff_exit_b - eff_entry_b) * q_b
                    fee_b = (q_b * abs(eff_entry_b) + q_b * abs(eff_exit_b)) * trading_fee_rate
                
                active_trade['PnL Asset A (USD)'] = pnl_a_gross - fee_a
                active_trade['PnL Asset B (USD)'] = pnl_b_gross - fee_b
                active_trade['Total Fees (USD)'] = fee_a + fee_b
                active_trade['Net PnL (USD)'] = (pnl_a_gross - fee_a) + (pnl_b_gross - fee_b)
                
                initial_primary_notional = 0
                if active_trade['Primary is A']:
                    initial_primary_notional = abs(q_a * entry_p_a) 
                else:
                    initial_primary_notional = abs(q_b * entry_p_b)
                
                active_trade['PnL % (Primary Notional)'] = (active_trade['Net PnL (USD)'] / initial_primary_notional) * 100 if initial_primary_notional != 0 else 0

                trade_log_list.append(active_trade.copy())
                active_trade = None 
    return pd.DataFrame(trade_log_list)


# --- Optimization Functions ---
def run_backtest_for_optimization(params_dict, data_with_z_precalculated, base_config, quantity, primary_is_A):
    """
    Runs a single backtest iteration with a given set of parameters.
    `data_with_z_precalculated` must contain 'zscore', 'close_a', 'close_b', 'hedge_ratio', 'symbol_a', 'symbol_b'.
    """
    iter_config = base_config.copy()
    iter_config['entry_threshold'] = params_dict['entry_threshold']
    iter_config['exit_threshold'] = params_dict['exit_threshold']
    iter_config['stop_loss_threshold'] = params_dict['stop_loss_threshold']

    trade_signals = generate_trade_signals(data_with_z_precalculated, iter_config)
    
    detailed_trade_log = generate_trade_log(
        signals_df=trade_signals, 
        data_with_prices_hedge_ratio=data_with_z_precalculated, 
        global_config=iter_config, 
        user_primary_quantity=quantity, 
        primary_is_A_selected=primary_is_A
    )

    if not detailed_trade_log.empty:
        total_pnl = detailed_trade_log['Net PnL (USD)'].sum()
        return total_pnl
    else:
        return -np.inf 

def optimize_strategy_parameters(data_with_z_precalculated, base_config, quantity, primary_is_A):
    """
    Optimizes entry, exit, and stop-loss thresholds using grid search.
    """
    st.write("Starting parameter optimization...")

    entry_thresholds = np.round(np.arange(0.8, 2.1, 0.2), 2)  
    exit_thresholds = np.round(np.arange(0.1, 0.8, 0.1), 2)   
    stop_loss_thresholds = np.round(np.arange(1.5, 3.6, 0.2), 2) 

    best_pnl = -np.inf
    best_params = {}
    iteration_count = 0
    
    valid_combinations = 0
    for entry_t in entry_thresholds:
        for exit_t in exit_thresholds:
            if exit_t >= entry_t: continue
            for stop_loss_t in stop_loss_thresholds:
                if stop_loss_t <= entry_t: continue 
                if stop_loss_t <= exit_t: continue 
                valid_combinations +=1
    
    if valid_combinations == 0:
        st.warning("No valid parameter combinations to test with the defined ranges and constraints. Adjust ranges.")
        return None, -np.inf

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Optimizing... Total valid combinations to test: {valid_combinations}")


    for entry_t in entry_thresholds:
        for exit_t in exit_thresholds:
            if exit_t >= entry_t:
                continue

            for stop_loss_t in stop_loss_thresholds:
                if stop_loss_t <= entry_t:
                    continue
                if stop_loss_t <= exit_t: 
                    continue

                iteration_count += 1
                current_params = {
                    'entry_threshold': entry_t,
                    'exit_threshold': exit_t,
                    'stop_loss_threshold': stop_loss_t
                }
                
                if iteration_count % 10 == 0 or iteration_count == valid_combinations :
                     current_best_pnl = "N/A" if best_pnl == -np.inf else f"${best_pnl:.2f}"
                     status_text.text(f"Optimizing: Combination {iteration_count}/{valid_combinations} | Current Best PnL: {current_best_pnl}")
                
                pnl = run_backtest_for_optimization(
                    current_params,
                    data_with_z_precalculated, 
                    base_config,
                    quantity,
                    primary_is_A
                )

                if pnl > best_pnl:
                    best_pnl = pnl
                    best_params = current_params
                
                progress_bar.progress(iteration_count / valid_combinations)

    status_text.text(f"Optimization complete! Processed {iteration_count} valid combinations.")
    if iteration_count > 0 : progress_bar.progress(1.0) 
    else: progress_bar.empty()


    if best_pnl == -np.inf: 
        st.warning("Optimization did not find any profitable trades with the tested parameter combinations.")
        return None, -np.inf
        
    return best_params, best_pnl


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📈 Statistical Arbitrage Trading Algorithm & Optimizer") 
st.markdown("""
This application implements a statistical arbitrage trading strategy for cryptocurrency pairs. 
Select assets, configure parameters, run the analysis, or optimize parameters to find the best Z-score thresholds.
Charts are interactive (zoom/pan). Default timeframe is daily.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration Parameters")

api_base_url = DEFAULT_CONFIG['api_base_url'] 
products_df = get_products(api_base_url)

if products_df.empty:
    st.sidebar.error("Could not fetch product list from API. Please check API URL or network.")
    st.stop()

product_options = {f"{row['symbol']} ({row.get('description', 'N/A')})": row['id'] 
                   for _, row in products_df.iterrows() if 'symbol' in row and 'id' in row} 
product_display_names = list(product_options.keys())

if not product_display_names:
    st.sidebar.error("No suitable products found from API to populate dropdowns.")
    st.stop()
    
default_a_symbol_desc = next((s for s in product_display_names if "BTCUSD" in s.upper()), product_display_names[0])
default_b_symbol_desc = next((s for s in product_display_names if "ETHUSD" in s.upper() and s != default_a_symbol_desc), 
                             product_display_names[1] if len(product_display_names) > 1 else product_display_names[0])


asset_a_display = st.sidebar.selectbox("Asset A:", product_display_names, 
                                       index=product_display_names.index(default_a_symbol_desc) if default_a_symbol_desc in product_display_names else 0)
asset_b_display = st.sidebar.selectbox("Asset B:", product_display_names, 
                                       index=product_display_names.index(default_b_symbol_desc) if default_b_symbol_desc in product_display_names else (1 if len(product_display_names) > 1 else 0))

asset_a_id = product_options[asset_a_display]
asset_b_id = product_options[asset_b_display]

asset_a_symbol = products_df.loc[products_df['id'] == asset_a_id, 'symbol'].iloc[0] if asset_a_id in products_df['id'].values else "UNKNOWN_A"
asset_b_symbol = products_df.loc[products_df['id'] == asset_b_id, 'symbol'].iloc[0] if asset_b_id in products_df['id'].values else "UNKNOWN_B"


primary_asset_choice = st.sidebar.radio("Primary Asset:", (asset_a_symbol, asset_b_symbol), key=f"primary_asset_{asset_a_symbol}_{asset_b_symbol}")
quantity = st.sidebar.number_input("Quantity of Primary Asset:", min_value=0.000001, value=1.0, step=0.001, format="%.6f")

timeframe_options = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
timeframe = st.sidebar.selectbox("Timeframe:", timeframe_options, index=timeframe_options.index(DEFAULT_CONFIG['timeframe']))

start_date = st.sidebar.date_input("Start Date:", value=DEFAULT_CONFIG['start_date'])
end_date = st.sidebar.date_input("End Date:", value=DEFAULT_CONFIG['end_date'])

st.sidebar.subheader("Strategy Parameters (for single run):")
lookback_period = st.sidebar.number_input("Lookback Period (days for MA/StdDev):", min_value=2, value=DEFAULT_CONFIG['lookback_period'])
entry_threshold = st.sidebar.number_input("Entry Z-score Threshold:", value=DEFAULT_CONFIG['entry_threshold'], step=0.1, format="%.1f")
exit_threshold = st.sidebar.number_input("Exit Z-score Threshold (Mean Reversion):", value=DEFAULT_CONFIG['exit_threshold'], step=0.1, format="%.1f")
stop_loss_threshold = st.sidebar.number_input("Stop Loss Z-score Threshold:", value=DEFAULT_CONFIG['stop_loss_threshold'], step=0.1, format="%.1f")

st.sidebar.subheader("Trading Costs:")
trading_fee = st.sidebar.number_input("Trading Fee (e.g., 0.001 for 0.1%):", value=DEFAULT_CONFIG['trading_fee'], step=0.0001, format="%.4f")
slippage = st.sidebar.number_input("Slippage (e.g., 0.001 for 0.1%):", value=DEFAULT_CONFIG['slippage'], step=0.0001, format="%.4f")

col1, col2 = st.sidebar.columns(2)
run_button = col1.button("🚀 Run Analysis", use_container_width=True)
optimize_button = col2.button("⚙️ Optimize Params", use_container_width=True)

# Add scheduled analysis button with timeframe-based name
schedule_button_text = f"⏱️ Start {timeframe} Analysis"
hourly_analysis_button = st.sidebar.button(schedule_button_text, use_container_width=True)

# Add placeholders for scheduled analysis status, next run time, and last run time
hourly_analysis_status = st.sidebar.empty()
next_run_time_placeholder = st.sidebar.empty()
last_run_time_placeholder = st.sidebar.empty()


# --- Main Area for Outputs ---
if run_button or optimize_button or hourly_analysis_button: 
    if asset_a_id == asset_b_id:
        st.error("Asset A and Asset B cannot be the same. Please select different assets.")
        st.stop()
    if start_date >= end_date and not hourly_analysis_button:
        st.error("Start Date must be before End Date.")
        st.stop()

    current_run_config = DEFAULT_CONFIG.copy()
    current_run_config.update({
        'timeframe': timeframe,
        'lookback_period': lookback_period, 
        'entry_threshold': entry_threshold, 
        'exit_threshold': exit_threshold,   
        'stop_loss_threshold': stop_loss_threshold, 
        'trading_fee': trading_fee,
        'slippage': slippage
    })

    ticker_config_selected = {
        'asset_a': {'id': asset_a_id, 'symbol': asset_a_symbol},
        'asset_b': {'id': asset_b_id, 'symbol': asset_b_symbol},
        'primary_asset': {'id': asset_a_id, 'symbol': asset_a_symbol} if primary_asset_choice == asset_a_symbol else {'id': asset_b_id, 'symbol': asset_b_symbol},
        'quantity': quantity, 
        'timeframe': timeframe,
        'start_timestamp': int(datetime.combine(start_date, datetime.min.time()).timestamp()),
        'end_timestamp': int(datetime.combine(end_date, datetime.max.time()).timestamp())
    }
    primary_is_A = ticker_config_selected['primary_asset']['id'] == ticker_config_selected['asset_a']['id']

    data_fetch_spinner = st.spinner("Fetching and preparing data...")
    with data_fetch_spinner:
        asset_a_data = get_historical_candles(
            api_base_url, ticker_config_selected['asset_a']['id'], ticker_config_selected['asset_a']['symbol'],
            ticker_config_selected['timeframe'], ticker_config_selected['start_timestamp'], ticker_config_selected['end_timestamp']
        )
        asset_b_data = get_historical_candles(
            api_base_url, ticker_config_selected['asset_b']['id'], ticker_config_selected['asset_b']['symbol'],
            ticker_config_selected['timeframe'], ticker_config_selected['start_timestamp'], ticker_config_selected['end_timestamp']
        )

    if asset_a_data.empty or asset_b_data.empty:
        st.error("Failed to fetch data for one or both assets. Cannot proceed.")
        st.stop()

    asset_a_data['log_price'] = np.log(asset_a_data['close'])
    asset_b_data['log_price'] = np.log(asset_b_data['close'])

    merged_df = pd.merge(
        asset_a_data[['close', 'log_price']].add_prefix('a_'),
        asset_b_data[['close', 'log_price']].add_prefix('b_'),
        left_index=True, right_index=True, how='inner'
    )
    merged_df.rename(columns={'a_close': 'close_a', 'a_log_price': 'log_price_a',
                              'b_close': 'close_b', 'b_log_price': 'log_price_b'}, inplace=True)
    
    merged_df['symbol_a'] = asset_a_symbol
    merged_df['symbol_b'] = asset_b_symbol


    if merged_df.empty:
        st.warning("No overlapping data found for the selected assets and timeframe.")
        st.stop()

    calc_spinner = st.spinner("Calculating hedge ratio, spread, and Z-score...")
    with calc_spinner:
        # Pass merged_df which has log_price_a and log_price_b
        data_with_hedge = calculate_hedge_ratio(merged_df, current_run_config['lookback_period'], current_run_config['timeframe'])
        data_with_z = calculate_zscore(data_with_hedge, current_run_config['lookback_period'], current_run_config['timeframe'])
        # data_with_z now contains: close_a, log_price_a, close_b, log_price_b, symbol_a, symbol_b, 
        # ols_intercept, hedge_ratio, spread, spread_ma, spread_std, zscore

    if data_with_z.empty or data_with_z['zscore'].isnull().all():
         st.warning("Could not calculate Z-scores. Check data or lookback period.")
         st.stop()

    # --- Action specific logic ---
    if hourly_analysis_button:
        st.header(f"⏱️ {timeframe} Analysis Setup")
        
        # Check if Supabase configuration is set
        if SUPABASE_URL == "YOUR_SUPABASE_URL" or SUPABASE_KEY == "YOUR_SUPABASE_KEY":
            st.error("Supabase configuration is not set. Please update the SUPABASE_URL and SUPABASE_KEY variables in the code.")
            st.code("""
# In statistical_arbitrage_streamlit_app.py, update:
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-api-key"
            """)
            st.stop()
        
        # Display configuration for scheduled analysis
        st.subheader(f"Configuration for {timeframe} Analysis")
        
        config_data = []
        config_data.append({"Parameter": "Primary Asset", "Value": primary_asset_choice})
        config_data.append({"Parameter": "Secondary Asset", "Value": asset_b_symbol if primary_asset_choice == asset_a_symbol else asset_a_symbol})
        config_data.append({"Parameter": "Primary Quantity", "Value": f"{quantity}"})
        config_data.append({"Parameter": "Timeframe", "Value": timeframe})
        config_data.append({"Parameter": "Lookback Period (days)", "Value": f"{lookback_period}"})
        config_data.append({"Parameter": "Entry Z-score Threshold", "Value": f"{entry_threshold}"})
        config_data.append({"Parameter": "Exit Z-score Threshold", "Value": f"{exit_threshold}"})
        config_data.append({"Parameter": "Stop Loss Z-score Threshold", "Value": f"{stop_loss_threshold}"})
        config_data.append({"Parameter": "Analysis Frequency", "Value": f"Every {timeframe}"})
        config_data.append({"Parameter": "Storage", "Value": f"Supabase table: {SUPABASE_TABLE}"})
        
        st.table(pd.DataFrame(config_data).set_index("Parameter"))
        
        # Start the scheduler
        confirm_button_text = f"✅ Confirm and Start {timeframe} Analysis"
        if st.button(confirm_button_text, key="confirm_scheduled"):
            try:
                status_container = st.empty()
                next_run_container = st.empty()
                last_run_container = st.empty()
                
                status_container.info("🔄 Initializing scheduled analysis...")
                
                # Create queues for communicating run times
                next_run_queue = queue.Queue()
                last_run_queue = queue.Queue()
                
                # Start the scheduler with the queues
                scheduler_thread = start_scheduled_analysis(
                    asset_a_id, asset_b_id, 
                    asset_a_symbol, asset_b_symbol, 
                    primary_is_A, quantity, 
                    current_run_config,
                    next_run_queue,
                    last_run_queue
                )
                
                # Wait briefly for the initial run to complete
                time.sleep(2)
                
                # Check for initial run completion
                if not last_run_queue.empty():
                    last_run_time = last_run_queue.get()
                    last_run_container.success(f"✅ Last run: {last_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if not next_run_queue.empty():
                        next_run_time = next_run_queue.get()
                        next_run_container.info(f"⏰ Next scheduled run: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        status_container.success("✅ Scheduled analysis is running!")
                    else:
                        status_container.warning("⚠️ Schedule started but next run time not available")
                else:
                    status_container.error("❌ Initial analysis did not complete. Check the logs for errors.")
                
                st.success(f"{timeframe} analysis has been started successfully!")
                st.info(f"""
                The analysis will run immediately and then every {timeframe}.
                Results will be stored in the Supabase database.
                You can close this page and the analysis will continue to run in the background.
                """)
                
                # Start a separate thread to update the run time displays
                def update_run_time_displays():
                    next_run = None
                    last_run = None
                    
                    while True:
                        try:
                            # Try to get the next run time from the queue (non-blocking)
                            try:
                                new_next_run = next_run_queue.get(block=False)
                                if new_next_run:
                                    next_run = new_next_run
                            except queue.Empty:
                                pass
                            
                            # Try to get the last run time from the queue (non-blocking)
                            try:
                                new_last_run = last_run_queue.get(block=False)
                                if new_last_run:
                                    last_run = new_last_run
                            except queue.Empty:
                                pass
                            
                            # Update the countdown timer if we have a next run time
                            if next_run:
                                next_run_formatted = next_run.strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Calculate time remaining
                                now = datetime.now()
                                time_remaining = next_run - now
                                
                                # Format time remaining
                                if time_remaining.total_seconds() > 0:
                                    hours, remainder = divmod(time_remaining.total_seconds(), 3600)
                                    minutes, seconds = divmod(remainder, 60)
                                    
                                    if hours > 0:
                                        countdown = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                                    elif minutes > 0:
                                        countdown = f"{int(minutes)}m {int(seconds)}s"
                                    else:
                                        countdown = f"{int(seconds)}s"
                                        
                                    next_run_time_placeholder.info(f"⏰ Next run: {next_run_formatted} (in {countdown})")
                                else:
                                    # If the next run time has passed, show "Running now..."
                                    next_run_time_placeholder.info("⏰ Analysis is running now...")
                            
                            # Update the last run time display if we have a last run time
                            if last_run:
                                last_run_formatted = last_run.strftime("%Y-%m-%d %H:%M:%S")
                                last_run_time_placeholder.success(f"✅ Last run: {last_run_formatted}")
                            
                            time.sleep(1)  # Update every second
                        except Exception as e:
                            print(f"Error updating run time displays: {e}")
                            time.sleep(5)  # Wait a bit longer if there's an error
                
                # Start the display update thread
                display_thread = threading.Thread(target=update_run_time_displays, daemon=True)
                display_thread.start()
                
                # Display a sample of what will be stored
                st.subheader("Sample Data to be Stored")
                
                # Run the analysis once to get a sample
                run_scheduled_analysis(
                    asset_a_id, asset_b_id, 
                    asset_a_symbol, asset_b_symbol, 
                    primary_is_A, quantity, 
                    current_run_config,
                    last_run_queue
                )
                
                st.info("The first analysis run has been completed. Check your Supabase database for the results.")
                
            except Exception as e:
                st.error(f"Error starting hourly analysis: {e}")
                hourly_analysis_status.error("❌ Failed to start hourly analysis!")
    
    elif run_button:
        st.header("📊 Analysis Results (Single Run)")
        
        # --- Price History Plot (Existing) ---
        st.subheader(f"📈 Price History: {asset_a_symbol} vs {asset_b_symbol}")
        fig_prices_plotly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_prices_plotly.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df['close_a'], name=asset_a_symbol, line=dict(color='blue')),
            secondary_y=False,
        )
        fig_prices_plotly.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df['close_b'], name=asset_b_symbol, line=dict(color='red')),
            secondary_y=True,
        )
        fig_prices_plotly.update_layout(title_text=f'Price History: {asset_a_symbol} and {asset_b_symbol}', xaxis_title='Date', legend_title_text='Assets')
        fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_a_symbol} Price</b>", secondary_y=False, color='blue') 
        fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_b_symbol} Price</b>", secondary_y=True, color='red') 
        st.plotly_chart(fig_prices_plotly, use_container_width=True)

        # --- NEW: Price Percentage Change Correlation Plot ---
        st.subheader(f"🔗 Price Percentage Change Correlation: {asset_a_symbol} vs {asset_b_symbol}")
        correlation_df = merged_df.copy()
        # Calculate percentage change. Using fillna(0) for the first row if needed, or dropna()
        correlation_df['a_pct_change'] = correlation_df['close_a'].pct_change().fillna(0) * 100
        correlation_df['b_pct_change'] = correlation_df['close_b'].pct_change().fillna(0) * 100
        
        plot_corr_df = correlation_df[['a_pct_change', 'b_pct_change']].dropna()

        if not plot_corr_df.empty and len(plot_corr_df) > 1:
            correlation_value = plot_corr_df['a_pct_change'].corr(plot_corr_df['b_pct_change'])
            fig_correlation = px.scatter(
                plot_corr_df,
                x='a_pct_change',
                y='b_pct_change',
                title=f'{asset_a_symbol} vs {asset_b_symbol} {timeframe} % Change Correlation (ρ = {correlation_value:.2f})', # Added timeframe to title
                labels={'a_pct_change': f'{asset_a_symbol} % Change', 'b_pct_change': f'{asset_b_symbol} % Change'},
                trendline='ols',  # Ordinary Least Squares trendline
                template='plotly_white'
            )
            fig_correlation.update_layout(xaxis_title=f'{asset_a_symbol} % Change', yaxis_title=f'{asset_b_symbol} % Change')
            st.plotly_chart(fig_correlation, use_container_width=True)
        else:
            st.warning("Not enough data points to calculate and plot price percentage change correlation.")


        # --- Z-Score Plot (Existing) ---
        st.subheader("📉 Z-Score of the Spread with Trade Signals (Interactive)")
        trade_signals = generate_trade_signals(data_with_z, current_run_config) 
        
        fig_zscore_plotly = go.Figure()
        fig_zscore_plotly.add_trace(go.Scatter(x=data_with_z.index, y=data_with_z['zscore'], mode='lines', name='Z-score', line=dict(color='green', width=2)))
        shapes, annotations = [], []
        thresholds_to_plot = {
            'Entry': (current_run_config['entry_threshold'], 'red'),
            'Exit': (current_run_config['exit_threshold'], 'blue'),
            'Stop Loss': (current_run_config['stop_loss_threshold'], 'purple')
        }
        min_x_val, max_x_val = data_with_z.index.min(), data_with_z.index.max()
        for name, (val, color) in thresholds_to_plot.items():
            shapes.append(dict(type='line', x0=min_x_val, y0=val, x1=max_x_val, y1=val, line=dict(color=color, width=1, dash='dash')))
            annotations.append(dict(x=min_x_val, y=val, text=f"{name} (+{val:.1f})", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=color)))
            shapes.append(dict(type='line', x0=min_x_val, y0=-val, x1=max_x_val, y1=-val, line=dict(color=color, width=1, dash='dash')))
            annotations.append(dict(x=min_x_val, y=-val, text=f"{name} (-{val:.1f})", showarrow=False, xanchor="left", yanchor="top", font=dict(color=color)))
        fig_zscore_plotly.add_trace(go.Scatter(x=data_with_z.index, y=[0]*len(data_with_z), mode='lines', name='Mean (0)', line=dict(color='black', width=1)))
        fig_zscore_plotly.update_layout(shapes=shapes, annotations=annotations)
        buy_signals_df = trade_signals[trade_signals['signal'] == 1.0]
        sell_signals_df = trade_signals[trade_signals['signal'] == -1.0]
        exit_signals_df = trade_signals[trade_signals['signal'] == 2.0]
        fig_zscore_plotly.add_trace(go.Scatter(x=buy_signals_df.index, y=buy_signals_df['zscore'], mode='markers', name=f'Buy Spread', marker=dict(color='green', size=10, symbol='triangle-up')))
        fig_zscore_plotly.add_trace(go.Scatter(x=sell_signals_df.index, y=sell_signals_df['zscore'], mode='markers', name=f'Sell Spread', marker=dict(color='red', size=10, symbol='triangle-down')))
        fig_zscore_plotly.add_trace(go.Scatter(x=exit_signals_df.index, y=exit_signals_df['zscore'], mode='markers', name='Exit Trade', marker=dict(color='blue', size=10, symbol='circle')))
        fig_zscore_plotly.update_layout(title_text='Z-score with Signals', xaxis_title='Date', yaxis_title='Z-score', legend_title_text='Signals & Levels')
        st.plotly_chart(fig_zscore_plotly, use_container_width=True)

        # --- NEW: OLS Regression Visualization for Hedge Ratio (Latest Window) ---
        st.subheader(f"🔍 OLS Regression for Hedge Ratio (Latest Lookback Window)")
        last_valid_hedge_data = data_with_hedge[data_with_hedge[['hedge_ratio', 'ols_intercept']].notna().all(axis=1)].iloc[-1:] # Ensure both are notna
        
        if not last_valid_hedge_data.empty:
            last_hr_timestamp = last_valid_hedge_data.index[0]
            latest_beta = last_valid_hedge_data['hedge_ratio'].iloc[0]
            latest_intercept = last_valid_hedge_data['ols_intercept'].iloc[0] 

            points_per_day_map = {
                '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2,
                '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1, '1w': 1/7, 'D': 1, 'W': 1/7
            }
            lookback_pts = int(current_run_config['lookback_period'] * points_per_day_map.get(current_run_config['timeframe'], 1))
            if lookback_pts < 2: lookback_pts = 2

            try:
                idx_loc = data_with_hedge.index.get_loc(last_hr_timestamp)
                
                if idx_loc >= lookback_pts:
                    # Slice from merged_df to get the original log prices for the window
                    window_data_for_plot = merged_df.iloc[idx_loc - lookback_pts + 1 : idx_loc + 1] # Correct slicing for OLS window
                    
                    if not window_data_for_plot.empty and 'log_price_a' in window_data_for_plot and 'log_price_b' in window_data_for_plot:
                        y_series_plot = window_data_for_plot['log_price_a']
                        X_series_plot = window_data_for_plot['log_price_b']

                        if not X_series_plot.empty and not y_series_plot.empty: # Ensure series are not empty
                            x_line = np.array([X_series_plot.min(), X_series_plot.max()])
                            y_line = latest_intercept + latest_beta * x_line

                            fig_ols = go.Figure()
                            fig_ols.add_trace(go.Scatter(x=X_series_plot, y=y_series_plot, mode='markers', name='Log Prices in Window',
                                                         marker=dict(color='rgba(100,100,100,0.5)')))
                            fig_ols.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name=f'OLS Fit (β={latest_beta:.4f})',
                                                         line=dict(color='orange', width=2)))
                            
                            fig_ols.update_layout(
                                title=f'OLS Regression: log({asset_a_symbol}) vs log({asset_b_symbol})<br>Window ending {last_hr_timestamp.strftime("%Y-%m-%d %H:%M")}',
                                xaxis_title=f'Log Price ({asset_b_symbol})',
                                yaxis_title=f'Log Price ({asset_a_symbol})',
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_ols, use_container_width=True)
                        else:
                            st.warning("Not enough data points in the OLS regression visualization window after slicing.")
                    else:
                        st.warning("Could not retrieve sufficient data for the OLS regression visualization window.")
                else:
                    st.warning("Not enough historical data points to form the lookback window for OLS visualization.")
            except KeyError:
                st.warning(f"Timestamp {last_hr_timestamp} not found in data index for OLS visualization.")
        else:
            st.warning("No valid hedge ratio calculated, cannot display OLS regression visualization.")


        # --- Trade Log (Existing) ---
        st.subheader("📜 Detailed Trade Log with P&L")
        detailed_trade_log = generate_trade_log(trade_signals, data_with_z, current_run_config, quantity, primary_is_A)
        if not detailed_trade_log.empty:
            st.dataframe(detailed_trade_log.style.format({
                "Entry Z-score": "{:.2f}", "Exit Z-score": "{:.2f}",
                "Entry Price A (Orig)": "{:.2f}", "Entry Price B (Orig)": "{:.2f}",
                "Exit Price A (Orig)": "{:.2f}", "Exit Price B (Orig)": "{:.2f}",
                "Entry Hedge Ratio": "{:.4f}", "Qty A": "{:.6f}", "Qty B": "{:.6f}",
                "PnL Asset A (USD)": "{:.2f}", "PnL Asset B (USD)": "{:.2f}",
                "Total Fees (USD)": "{:.2f}", "Net PnL (USD)": "{:.2f}",
                "PnL % (Primary Notional)": "{:.2f}%"
            }))
            st.markdown(f"**Overall Net P&L: ${detailed_trade_log['Net PnL (USD)'].sum():.2f}**")
            st.markdown(f"**Total Trades: {len(detailed_trade_log)}**")
        else:
            st.info("No trades were generated based on the current signals and data.")
        
        # --- MODIFIED: Latest Trade Quantity Context ---
        st.subheader("💡 Latest Market & Strategy Context")
        latest_full_data = data_with_z.join(trade_signals[['signal','position']], how='left').dropna(subset=['zscore', 'close_a', 'close_b', 'hedge_ratio'])
        if not latest_full_data.empty:
            latest_full_data_point = latest_full_data.iloc[-1]
            price_a_latest = latest_full_data_point['close_a']
            price_b_latest = latest_full_data_point['close_b']
            hedge_ratio_latest_val = latest_full_data_point['hedge_ratio']
            latest_zscore_val = latest_full_data_point['zscore']
            latest_signal_action_val = latest_full_data_point['signal']
            latest_position_val = latest_full_data_point['position']

            primary_asset_sym = asset_a_symbol if primary_is_A else asset_b_symbol
            secondary_asset_sym = asset_b_symbol if primary_is_A else asset_a_symbol
            primary_price_latest_val = price_a_latest if primary_is_A else price_b_latest
            secondary_price_latest_val = price_b_latest if primary_is_A else price_a_latest

            context_data = []
            context_data.append({"Parameter": "Latest Timestamp", "Value": latest_full_data_point.name.strftime('%Y-%m-%d %H:%M:%S')})
            context_data.append({"Parameter": f"Primary Asset ({primary_asset_sym}) Price", "Value": f"${primary_price_latest_val:.2f}"})
            context_data.append({"Parameter": f"Secondary Asset ({secondary_asset_sym}) Price", "Value": f"${secondary_price_latest_val:.2f}"})
            context_data.append({"Parameter": "Latest Hedge Ratio (β)", "Value": f"{hedge_ratio_latest_val:.4f}" if pd.notna(hedge_ratio_latest_val) else "N/A"})
            context_data.append({"Parameter": "Latest Z-score", "Value": f"{latest_zscore_val:.4f}" if pd.notna(latest_zscore_val) else "N/A"})
            
            context_data.append({"Parameter": "--- Strategy Parameters ---", "Value": "--- ---"}) # Separator
            context_data.append({"Parameter": "Entry Z-score Threshold", "Value": f"{current_run_config['entry_threshold']:.2f}"})
            context_data.append({"Parameter": "Exit Z-score Threshold", "Value": f"{current_run_config['exit_threshold']:.2f}"})
            context_data.append({"Parameter": "Stop Loss Z-score Threshold", "Value": f"{current_run_config['stop_loss_threshold']:.2f}"})
            
            context_data.append({"Parameter": "--- Trade Setup ---", "Value": "--- ---"}) # Separator
            context_data.append({"Parameter": "User Specified Primary Quantity", "Value": f"{quantity} {primary_asset_sym}"})

            if pd.notna(hedge_ratio_latest_val) and primary_price_latest_val > 0 and secondary_price_latest_val > 0:
                secondary_qty_calc = calculate_secondary_quantity(
                    quantity, hedge_ratio_latest_val, primary_price_latest_val, secondary_price_latest_val, primary_is_A
                )
                context_data.append({"Parameter": "Calculated Secondary Quantity", "Value": f"{secondary_qty_calc:.6f} {secondary_asset_sym}"})
            else:
                context_data.append({"Parameter": "Calculated Secondary Quantity", "Value": "N/A (Invalid inputs for calculation)"})

            context_df = pd.DataFrame(context_data)
            st.table(context_df.set_index("Parameter")) # Using st.table for a cleaner look

            trade_action_exp = "Hold / No new signal."
            asset1_sym_for_exp = asset_a_symbol 
            asset2_sym_for_exp = asset_b_symbol
            
            # Determine quantities for explanation based on which asset is primary
            # And use the calculated secondary quantity if available
            qty1_for_exp_val, qty2_for_exp_val = "N/A", "N/A"
            if 'secondary_qty_calc' in locals() and pd.notna(secondary_qty_calc): # Check if secondary_qty_calc was successfully computed
                qty1_for_exp_val = quantity if primary_is_A else secondary_qty_calc
                qty2_for_exp_val = secondary_qty_calc if primary_is_A else quantity
                qty1_for_exp_str = f"{qty1_for_exp_val:.6f}"
                qty2_for_exp_str = f"{qty2_for_exp_val:.6f}"
            else: # Fallback if secondary quantity couldn't be calculated
                qty1_for_exp_str = f"{quantity if primary_is_A else 'N/A'}"
                qty2_for_exp_str = f"{'N/A' if primary_is_A else quantity}"


            if latest_signal_action_val == 1.0: 
                trade_action_exp = f"Signal: BUY SPREAD. Action: LONG {qty1_for_exp_str} {asset1_sym_for_exp}, SHORT {qty2_for_exp_str} {asset2_sym_for_exp}."
            elif latest_signal_action_val == -1.0: 
                trade_action_exp = f"Signal: SELL SPREAD. Action: SHORT {qty1_for_exp_str} {asset1_sym_for_exp}, LONG {qty2_for_exp_str} {asset2_sym_for_exp}."
            elif latest_signal_action_val == 2.0:
                 trade_action_exp = "Signal: EXIT TRADE. Action: Close current spread position."
            
            st.markdown(f"**Latest Signal Interpretation:** {trade_action_exp}")
            st.markdown(f"**Position after this signal (if any):** {'Long Spread' if latest_position_val == 1 else ('Short Spread' if latest_position_val == -1 else 'No Position')}")
            
        else:
            st.warning("Could not display latest market & strategy context (no valid recent data).")


    elif optimize_button:
        st.header("🛠️ Parameter Optimization Results")
        best_params_found, max_pnl_found = optimize_strategy_parameters(
            data_with_z, 
            current_run_config, 
            quantity, 
            primary_is_A 
        )

        if best_params_found:
            st.success(f"Optimization Complete!")
            st.subheader("🏆 Best Parameters Found:")
            st.markdown(f"- **Entry Z-score Threshold:** `{best_params_found['entry_threshold']:.2f}`")
            st.markdown(f"- **Exit Z-score Threshold (Mean Reversion):** `{best_params_found['exit_threshold']:.2f}`")
            st.markdown(f"- **Stop Loss Z-score Threshold:** `{best_params_found['stop_loss_threshold']:.2f}`")
            
            st.subheader(f"💰 Maximum P&L Achieved with these parameters: ${max_pnl_found:.2f}")
            
            st.markdown("---")
            st.markdown("You can now manually input these parameters in the sidebar and click '**Run Analysis**' to see the detailed charts and trade log for this optimal configuration.")
            
        else:
            st.error("Optimization finished, but no profitable parameter set was found within the tested ranges. Try adjusting optimization ranges or data period.")


# --- Functions for Hourly Analysis and Supabase Storage ---
def store_analysis_in_supabase(
    primary_asset_symbol, primary_asset_price, 
    secondary_asset_symbol, secondary_asset_price,
    hedge_ratio, z_score, 
    entry_z_threshold, exit_z_threshold, stoploss_z_threshold,
    primary_quantity, secondary_quantity
):
    """
    Store the results of the statistical arbitrage analysis in Supabase.
    
    Args:
        primary_asset_symbol (str): Symbol of the primary asset
        primary_asset_price (float): Current price of the primary asset
        secondary_asset_symbol (str): Symbol of the secondary asset
        secondary_asset_price (float): Current price of the secondary asset
        hedge_ratio (float): Calculated hedge ratio
        z_score (float): Current Z-score
        entry_z_threshold (float): Entry Z-score threshold
        exit_z_threshold (float): Exit Z-score threshold
        stoploss_z_threshold (float): Stop-loss Z-score threshold
        primary_quantity (float): Quantity of the primary asset
        secondary_quantity (float): Calculated quantity of the secondary asset
    
    Returns:
        bool: True if the data was successfully stored, False otherwise
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        # Prepare data for insertion
        data = {
            "run_timestamp": datetime.now().isoformat(),
            "primary_asset_symbol": primary_asset_symbol,
            "primary_asset_price": float(primary_asset_price),
            "secondary_asset_symbol": secondary_asset_symbol,
            "secondary_asset_price": float(secondary_asset_price),
            "hedge_ratio": float(hedge_ratio),
            "z_score": float(z_score),
            "entry_z_threshold": float(entry_z_threshold),
            "exit_z_threshold": float(exit_z_threshold),
            "stoploss_z_threshold": float(stoploss_z_threshold),
            "primary_quantity": float(primary_quantity),
            "secondary_quantity": float(secondary_quantity)
        }
        
        # Insert data into Supabase
        response = supabase.table(SUPABASE_TABLE).insert(data).execute()
        
        # Check if the insertion was successful
        if hasattr(response, 'data') and response.data:
            return True
        return False
    except Exception as e:
        print(f"Error storing data in Supabase: {e}")
        return False

def run_scheduled_analysis(asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue=None):
    """
    Run the statistical arbitrage analysis on a scheduled basis and store the results in Supabase.
    
    Args:
        asset_a_id (str): ID of asset A
        asset_b_id (str): ID of asset B
        asset_a_symbol (str): Symbol of asset A
        asset_b_symbol (str): Symbol of asset B
        primary_is_A (bool): Whether asset A is the primary asset
        quantity (float): Quantity of the primary asset
        config (dict): Configuration parameters for the analysis
        last_run_queue (queue.Queue, optional): Queue to communicate the last run time back to the UI
    
    Returns:
        None
    """
    try:
        # Record the run time
        run_time = datetime.now()
        print(f"[{run_time}] Running scheduled analysis for {asset_a_symbol} and {asset_b_symbol}...")
        
        # Always update the last run time if a queue is provided
        if last_run_queue:
            try:
                # Clear any old values and put the new run time
                while not last_run_queue.empty():
                    last_run_queue.get()
                last_run_queue.put(run_time)
                print(f"[{run_time}] Updated last run time in queue")
            except Exception as e:
                print(f"[{datetime.now()}] Error updating last run time queue: {e}")
        
        # Calculate timestamps for data fetching (last 30 days)
        end_timestamp = int(datetime.now().timestamp())
        start_timestamp = int((datetime.now() - timedelta(days=config['lookback_period'])).timestamp())
        
        # Fetch historical data
        asset_a_data = get_historical_candles(
            config['api_base_url'], asset_a_id, asset_a_symbol,
            config['timeframe'], start_timestamp, end_timestamp
        )
        asset_b_data = get_historical_candles(
            config['api_base_url'], asset_b_id, asset_b_symbol,
            config['timeframe'], start_timestamp, end_timestamp
        )
        
        if asset_a_data.empty or asset_b_data.empty:
            print(f"[{datetime.now()}] Failed to fetch data for one or both assets. Cannot proceed with scheduled analysis.")
            return
        
        # Prepare data for analysis
        asset_a_data['log_price'] = np.log(asset_a_data['close'])
        asset_b_data['log_price'] = np.log(asset_b_data['close'])
        
        merged_df = pd.merge(
            asset_a_data[['close', 'log_price']].add_prefix('a_'),
            asset_b_data[['close', 'log_price']].add_prefix('b_'),
            left_index=True, right_index=True, how='inner'
        )
        merged_df.rename(columns={'a_close': 'close_a', 'a_log_price': 'log_price_a',
                                 'b_close': 'close_b', 'b_log_price': 'log_price_b'}, inplace=True)
        
        merged_df['symbol_a'] = asset_a_symbol
        merged_df['symbol_b'] = asset_b_symbol
        
        if merged_df.empty:
            print(f"[{datetime.now()}] No overlapping data found for the selected assets and timeframe.")
            return
        
        # Calculate hedge ratio and Z-score
        data_with_hedge = calculate_hedge_ratio(merged_df, config['lookback_period'], config['timeframe'])
        data_with_z = calculate_zscore(data_with_hedge, config['lookback_period'], config['timeframe'])
        
        if data_with_z.empty or data_with_z['zscore'].isnull().all():
            print(f"[{datetime.now()}] Could not calculate Z-scores. Check data or lookback period.")
            return
        
        # Get the latest data point
        latest_data = data_with_z.iloc[-1]
        
        # Get the latest prices and calculated values
        price_a_latest = latest_data['close_a']
        price_b_latest = latest_data['close_b']
        hedge_ratio_latest = latest_data['hedge_ratio']
        zscore_latest = latest_data['zscore']
        
        # Determine primary and secondary asset details
        primary_asset_sym = asset_a_symbol if primary_is_A else asset_b_symbol
        secondary_asset_sym = asset_b_symbol if primary_is_A else asset_a_symbol
        primary_price_latest = price_a_latest if primary_is_A else price_b_latest
        secondary_price_latest = price_b_latest if primary_is_A else price_a_latest
        
        # Calculate secondary quantity
        secondary_quantity = calculate_secondary_quantity(
            quantity, hedge_ratio_latest, primary_price_latest, secondary_price_latest, primary_is_A
        )
        
        # Store the analysis results in Supabase
        success = store_analysis_in_supabase(
            primary_asset_sym, primary_price_latest,
            secondary_asset_sym, secondary_price_latest,
            hedge_ratio_latest, zscore_latest,
            config['entry_threshold'], config['exit_threshold'], config['stop_loss_threshold'],
            quantity, secondary_quantity
        )
        
        if success:
            print(f"[{datetime.now()}] Successfully stored analysis results in Supabase.")
        else:
            print(f"[{datetime.now()}] Failed to store analysis results in Supabase.")
    
    except Exception as e:
        print(f"[{datetime.now()}] Error in scheduled analysis: {e}")

def start_scheduled_analysis(asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, next_run_queue=None, last_run_queue=None):
    """
    Start a scheduler for running the statistical arbitrage analysis at intervals based on the selected timeframe.
    
    Args:
        asset_a_id (str): ID of asset A
        asset_b_id (str): ID of asset B
        asset_a_symbol (str): Symbol of asset A
        asset_b_symbol (str): Symbol of asset B
        primary_is_A (bool): Whether asset A is the primary asset
        quantity (float): Quantity of the primary asset
        config (dict): Configuration parameters for the analysis
        next_run_queue (queue.Queue, optional): Queue to communicate the next run time back to the UI
        last_run_queue (queue.Queue, optional): Queue to communicate the last run time back to the UI
    
    Returns:
        threading.Thread: The scheduler thread
    """
    def calculate_next_run_time(timeframe):
        """Calculate the next run time based on the timeframe."""
        now = datetime.now()
        
        if timeframe == '1m':
            return now + timedelta(minutes=1)
        elif timeframe == '5m':
            minutes_to_add = 5 - (now.minute % 5)
            if minutes_to_add == 0:
                minutes_to_add = 5
            return now + timedelta(minutes=minutes_to_add)
        elif timeframe == '15m':
            minutes_to_add = 15 - (now.minute % 15)
            if minutes_to_add == 0:
                minutes_to_add = 15
            return now + timedelta(minutes=minutes_to_add)
        elif timeframe == '30m':
            minutes_to_add = 30 - (now.minute % 30)
            if minutes_to_add == 0:
                minutes_to_add = 30
            return now + timedelta(minutes=minutes_to_add)
        elif timeframe == '1h':
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif timeframe == '2h':
            hours_to_add = 2 - (now.hour % 2)
            if hours_to_add == 0:
                hours_to_add = 2
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
        elif timeframe == '4h':
            hours_to_add = 4 - (now.hour % 4)
            if hours_to_add == 0:
                hours_to_add = 4
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
        elif timeframe == '6h':
            hours_to_add = 6 - (now.hour % 6)
            if hours_to_add == 0:
                hours_to_add = 6
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
        elif timeframe == '12h':
            hours_to_add = 12 - (now.hour % 12)
            if hours_to_add == 0:
                hours_to_add = 12
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
        elif timeframe in ['1d', 'D']:
            return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        elif timeframe in ['1w', 'W']:
            days_until_monday = 7 - now.weekday()
            if days_until_monday == 7:
                days_until_monday = 0
            return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday))
        else:
            # Default to hourly
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    def run_scheduler():
        try:
            print(f"[{datetime.now()}] Starting immediate analysis run...")
            
            # Run the analysis immediately
            run_scheduled_analysis(asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue)
            
            # Set the schedule interval based on the selected timeframe
            timeframe = config['timeframe']
            
            # Calculate and communicate the next run time
            next_run_time = calculate_next_run_time(timeframe)
            if next_run_queue and next_run_time:
                # Clear any old values and put the new next run time
                while not next_run_queue.empty():
                    next_run_queue.get()
                next_run_queue.put(next_run_time)
            
            print(f"[{datetime.now()}] Initial analysis complete. Setting up schedule...")
        
            # Map timeframe to schedule interval
            job = None
            if timeframe == '1m':
                job = schedule.every(1).minutes.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '5m':
                job = schedule.every(5).minutes.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '15m':
                job = schedule.every(15).minutes.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '30m':
                job = schedule.every(30).minutes.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '1h':
                job = schedule.every(1).hour.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '2h':
                job = schedule.every(2).hours.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '4h':
                job = schedule.every(4).hours.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '6h':
                job = schedule.every(6).hours.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe == '12h':
                job = schedule.every(12).hours.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe in ['1d', 'D']:
                job = schedule.every().day.at("00:00").do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            elif timeframe in ['1w', 'W']:
                job = schedule.every().monday.at("00:00").do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            else:
                # Default to hourly if timeframe is not recognized
                job = schedule.every(1).hour.do(
                    run_scheduled_analysis, 
                    asset_a_id, asset_b_id, asset_a_symbol, asset_b_symbol, primary_is_A, quantity, config, last_run_queue
                )
            
            print(f"[{datetime.now()}] Scheduled analysis to run at {timeframe} intervals")
            
            # Keep the scheduler running
            while True:
                schedule.run_pending()
                
                # Update next run time if needed
                if next_run_queue and job and job.next_run:
                    # Put the next run time in the queue for the UI to display
                    try:
                        # Non-blocking put to avoid queue filling up
                        if next_run_queue.empty():
                            next_run_queue.put(job.next_run)
                    except:
                        pass
                
                # Check more frequently for shorter timeframes
                if timeframe in ['1m', '5m']:
                    time.sleep(10)  # Check every 10 seconds for minute-based timeframes
                else:
                    time.sleep(60)  # Check every minute for longer timeframes
        except Exception as e:
            print(f"[{datetime.now()}] Error in scheduler thread: {e}")
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    return scheduler_thread

# --- Instructions for Running ---
st.sidebar.markdown("---")
st.sidebar.info("""
**How to Use:**
1. Select Asset A and Asset B.
2. Choose the Primary Asset and set its Quantity.
3. Adjust Timeframe, Date Range, and Strategy Parameters (for single run).
4. Click '**Run Analysis**' for a single backtest or '**Optimize Params**' to find optimal Z-score thresholds.
5. Click '**Start [Timeframe] Analysis**' to run the analysis at the selected timeframe interval and store results in Supabase.
The app will fetch data, perform calculations, and display results.
""")