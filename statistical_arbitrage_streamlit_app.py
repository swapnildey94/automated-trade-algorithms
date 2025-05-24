import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    # Ensure timeframe_selection is valid, default to 24 if not found (e.g. for '1d')
    lookback_points = int(lookback_period_days * points_per_day.get(timeframe_selection, 1)) # Default to 1 point per day if timeframe not in dict (e.g. '1D')
    if lookback_points < 2: lookback_points = 2 # Minimum points for regression

    data = data_to_process.copy()
    data['hedge_ratio'] = np.nan
    data['spread'] = np.nan

    for i in range(lookback_points, len(data)):
        # Slice data for the current lookback window
        y_series = data['log_price_a'].iloc[i-lookback_points:i]
        X_series = data['log_price_b'].iloc[i-lookback_points:i]
        
        # Ensure there's enough non-null data
        if y_series.isnull().any() or X_series.isnull().any() or len(y_series) < 2 or len(X_series) < 2:
            data.loc[data.index[i], 'hedge_ratio'] = np.nan # Mark as NaN if insufficient data
            continue
            
        # Prepare data for OLS: y = log_price_a, X = const + log_price_b
        X_with_const = sm.add_constant(X_series, prepend=True)
        
        try:
            model = sm.OLS(y_series, X_with_const).fit()
            # Ensure model fitting was successful and parameters are available
            if len(model.params) > 1: # params are [const, beta]
                 beta = model.params.iloc[1] # Beta is the hedge ratio
                 data.loc[data.index[i], 'hedge_ratio'] = beta
                 # Calculate spread: log_price_a - beta * log_price_b
                 data.loc[data.index[i], 'spread'] = data['log_price_a'].iloc[i] - beta * data['log_price_b'].iloc[i]
            else:
                data.loc[data.index[i], 'hedge_ratio'] = np.nan # Mark as NaN if model params are not as expected
        except Exception: 
            # Catch any errors during OLS fitting (e.g., perfect multicollinearity, though unlikely with add_constant)
            data.loc[data.index[i], 'hedge_ratio'] = np.nan
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
    # min_periods is set to ensure some stability in early calculations
    result_df['spread_ma'] = result_df['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).mean()
    result_df['spread_std'] = result_df['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).std()
    
    # Calculate Z-score: (spread - spread_ma) / spread_std
    result_df['zscore'] = (result_df['spread'] - result_df['spread_ma']) / result_df['spread_std']
    return result_df

def calculate_secondary_quantity(primary_quantity, hedge_ratio, primary_price, secondary_price, primary_is_asset_a):
    """Calculates the quantity of the secondary asset to achieve a hedge."""
    if primary_price == 0 or secondary_price == 0: return 0 # Avoid division by zero
    
    primary_notional = primary_quantity * primary_price # Notional value of the primary asset

    if primary_is_asset_a:
        # If primary is Asset A, we are hedging Asset A with Asset B
        # Spread = log(A) - beta * log(B). For delta neutrality, Value(A) approx beta * Value(B) * (Price_A / Price_B) (approximation for log prices)
        # A more direct notional hedge: Notional_A = beta * Notional_B_equivalent_in_A_terms
        # Or, if beta is from log(Pa) ~ c + beta*log(Pb), then d(Pa)/Pa = beta * d(Pb)/Pb
        # Value_A = Q_A * P_A. Value_B = Q_B * P_B.
        # We want Q_A * P_A = hedge_ratio_value_based * Q_B * P_B
        # The hedge_ratio from OLS of log prices is elasticity: (dPa/Pa) / (dPb/Pb)
        # For value hedge: Q_A*P_A = HedgeRatio_dollar * Q_B*P_B
        # Here, hedge_ratio is beta from log(Pa) = c + beta*log(Pb)
        # Notional A = Q_A * P_A
        # Notional B target = (Q_A * P_A) / hedge_ratio (if hedge_ratio is P_A / P_B type)
        # If hedge_ratio is beta from log prices, it's more complex.
        # The provided code implies: Notional_A_target_change = beta * Notional_B_target_change
        # Let's assume the existing logic's interpretation of hedge_ratio:
        # If primary is A (long A, short B or vice-versa for spread)
        # We have Q_A. Notional A = Q_A * P_A.
        # We need Q_B such that Q_B * P_B is appropriately related by beta to Q_A * P_A.
        # Original notebook might have implied: Value of asset A = beta * Value of asset B
        # So, Q_A * P_A = beta * Q_B * P_B  => Q_B = (Q_A * P_A) / (beta * P_B)
        # The code seems to use `secondary_notional_target = primary_notional * hedge_ratio`
        # This means Q_B * P_B = (Q_A * P_A) * hedge_ratio
        # Q_B = (Q_A * P_A * hedge_ratio) / P_B
        secondary_notional_target = primary_notional * hedge_ratio 
        secondary_asset_quantity = secondary_notional_target / secondary_price if secondary_price != 0 else 0
    else: # Primary is Asset B
        # We have Q_B. Notional B = Q_B * P_B.
        # We need Q_A such that Q_A * P_A is related.
        # Using the same logic: Q_A * P_A = beta * Q_B * P_B
        # Here, primary is B. So, primary_notional = Q_B * P_B.
        # secondary_notional_target (for A) = primary_notional (of B) / hedge_ratio (if beta relates A to B, then B to A is 1/beta)
        # The code uses: `secondary_notional_target = primary_notional / hedge_ratio`
        # This means Q_A * P_A = (Q_B * P_B) / hedge_ratio
        # Q_A = (Q_B * P_B) / (hedge_ratio * P_A)
        if hedge_ratio == 0: return 0 # Avoid division by zero
        secondary_notional_target = primary_notional / hedge_ratio
        secondary_asset_quantity = secondary_notional_target / secondary_price if secondary_price != 0 else 0
    return abs(secondary_asset_quantity) # Quantity should be positive

def generate_trade_signals(data_df, config_params):
    """Generates trading signals based on Z-score thresholds."""
    signals = pd.DataFrame(index=data_df.index)
    signals['zscore'] = data_df['zscore']
    signals['signal'] = 0.0  # 0: Hold, 1: Enter Long Spread, -1: Enter Short Spread, 2: Exit
    signals['position'] = 0  # 0: No position, 1: Long Spread, -1: Short Spread

    entry_thresh = config_params['entry_threshold']
    exit_thresh = config_params['exit_threshold'] # For mean reversion
    stop_loss_thresh = config_params['stop_loss_threshold'] # For stop loss
    current_pos = 0

    for i in range(1, len(signals)): # Start from the second row to use previous Z-score
        z = signals['zscore'].iloc[i]
        prev_z = signals['zscore'].iloc[i-1]

        if pd.isna(z) or pd.isna(prev_z): # Skip if Z-score is NaN
            signals.loc[signals.index[i], 'position'] = current_pos
            continue

        if current_pos == 0: # No active position, look for entry
            if prev_z >= -entry_thresh and z < -entry_thresh: # Z-score crosses below -entry_thresh (buy spread)
                signals.loc[signals.index[i], 'signal'] = 1.0
                current_pos = 1
            elif prev_z <= entry_thresh and z > entry_thresh: # Z-score crosses above +entry_thresh (sell spread)
                signals.loc[signals.index[i], 'signal'] = -1.0
                current_pos = -1
        elif current_pos == 1: # Currently Long Spread (expect Z-score to revert to mean, i.e., increase)
            # Exit if Z-score crosses above -exit_thresh (mean reversion)
            # OR if Z-score crosses below -stop_loss_thresh (stop-loss)
            if (prev_z <= -exit_thresh and z > -exit_thresh) or \
               (prev_z >= -stop_loss_thresh and z < -stop_loss_thresh and stop_loss_thresh > entry_thresh): # Ensure SL is wider
                signals.loc[signals.index[i], 'signal'] = 2.0 # Exit signal
                current_pos = 0
        elif current_pos == -1: # Currently Short Spread (expect Z-score to revert to mean, i.e., decrease)
            # Exit if Z-score crosses below +exit_thresh (mean reversion)
            # OR if Z-score crosses above +stop_loss_thresh (stop-loss)
            if (prev_z >= exit_thresh and z < exit_thresh) or \
               (prev_z <= stop_loss_thresh and z > stop_loss_thresh and stop_loss_thresh > entry_thresh): # Ensure SL is wider
                signals.loc[signals.index[i], 'signal'] = 2.0 # Exit signal
                current_pos = 0
        
        signals.loc[signals.index[i], 'position'] = current_pos
        
    # For plotting convenience
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

    # Ensure required columns are present
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
        symbol_a = data_with_prices_hedge_ratio.loc[timestamp, 'symbol_a'] # Relies on these columns existing
        symbol_b = data_with_prices_hedge_ratio.loc[timestamp, 'symbol_b'] # Relies on these columns existing


        if active_trade is None: # No active trade, look for an entry signal
            if signal == 1.0 or signal == -1.0: # Entry signal
                if pd.isna(current_hedge_ratio) or current_price_a == 0 or current_price_b == 0:
                    continue # Skip if data is invalid for trade execution
                
                qty_a, qty_b = 0, 0
                if primary_is_A_selected:
                    qty_a = user_primary_quantity
                    qty_b = calculate_secondary_quantity(qty_a, current_hedge_ratio, current_price_a, current_price_b, True)
                else: # Primary is B
                    qty_b = user_primary_quantity
                    # Note: calculate_secondary_quantity expects (primary_qty, hedge_ratio, primary_price, secondary_price, primary_is_A)
                    # If primary is B, then A is secondary.
                    qty_a = calculate_secondary_quantity(qty_b, current_hedge_ratio, current_price_b, current_price_a, False) 
                
                if qty_a == 0 or qty_b == 0: continue # Skip if quantities are zero

                trade_type_str = f'Buy Spread (L {symbol_a}, S {symbol_b})' if signal == 1.0 else f'Sell Spread (S {symbol_a}, L {symbol_b})'
                active_trade = {
                    'Entry Timestamp': timestamp, 'Trade Type': trade_type_str,
                    'Entry Z-score': zscore_at_signal,
                    'Entry Price A (Orig)': current_price_a, 'Entry Price B (Orig)': current_price_b,
                    'Entry Hedge Ratio': current_hedge_ratio,
                    'Qty A': qty_a, 'Qty B': qty_b, 'Primary is A': primary_is_A_selected,
                    'Symbol A': symbol_a, 'Symbol B': symbol_b, # Store symbols
                    'Exit Timestamp': None, 'Exit Price A (Orig)': None, 'Exit Price B (Orig)': None, 'Exit Z-score': None,
                    'PnL Asset A (USD)': None, 'PnL Asset B (USD)': None, 'Total Fees (USD)': None, 
                    'Net PnL (USD)': None, 'PnL % (Primary Notional)': None
                }
        elif active_trade is not None: # Active trade exists, look for an exit signal or EOD
            is_eod_closure = (timestamp == signals_df.index[-1] and active_trade['Exit Timestamp'] is None)
            if signal == 2.0 or is_eod_closure: # Exit signal or forced closure at end of data
                active_trade['Exit Timestamp'] = timestamp
                active_trade['Exit Price A (Orig)'] = current_price_a
                active_trade['Exit Price B (Orig)'] = current_price_b
                active_trade['Exit Z-score'] = zscore_at_signal if signal == 2.0 else data_with_prices_hedge_ratio.loc[timestamp, 'zscore'] # Use current z-score if EOD

                if is_eod_closure and signal != 2.0: # Mark if closed at EOD without explicit exit signal
                     active_trade['Trade Type'] += ' (Closed at EOD)'
                
                # Retrieve details from active_trade dict
                entry_p_a = active_trade['Entry Price A (Orig)']
                entry_p_b = active_trade['Entry Price B (Orig)']
                exit_p_a = active_trade['Exit Price A (Orig)']
                exit_p_b = active_trade['Exit Price B (Orig)']
                q_a = active_trade['Qty A']
                q_b = active_trade['Qty B']
                
                pnl_a_gross, pnl_b_gross, fee_a, fee_b = 0,0,0,0

                # Calculate P&L considering slippage and fees
                # For "Buy Spread": Long A, Short B
                if 'Buy Spread' in active_trade['Trade Type']: 
                    # Asset A: Bought at entry, Sold at exit
                    eff_entry_a = entry_p_a * (1 + slippage) # Higher effective buy price
                    eff_exit_a = exit_p_a * (1 - slippage)   # Lower effective sell price
                    pnl_a_gross = (eff_exit_a - eff_entry_a) * q_a
                    fee_a = (q_a * abs(eff_entry_a) + q_a * abs(eff_exit_a)) * trading_fee_rate
                    
                    # Asset B: Sold at entry, Bought at exit
                    eff_entry_b_sell = entry_p_b * (1 - slippage) # Lower effective sell price
                    eff_exit_b_buy = exit_p_b * (1 + slippage)   # Higher effective buy price
                    pnl_b_gross = (eff_entry_b_sell - eff_exit_b_buy) * q_b
                    fee_b = (q_b * abs(eff_entry_b_sell) + q_b * abs(eff_exit_b_buy)) * trading_fee_rate
                # For "Sell Spread": Short A, Long B
                elif 'Sell Spread' in active_trade['Trade Type']: 
                    # Asset A: Sold at entry, Bought at exit
                    eff_entry_a_sell = entry_p_a * (1 - slippage)
                    eff_exit_a_buy = exit_p_a * (1 + slippage)
                    pnl_a_gross = (eff_entry_a_sell - eff_exit_a_buy) * q_a
                    fee_a = (q_a * abs(eff_entry_a_sell) + q_a * abs(eff_exit_a_buy)) * trading_fee_rate

                    # Asset B: Bought at entry, Sold at exit
                    eff_entry_b = entry_p_b * (1 + slippage)
                    eff_exit_b = exit_p_b * (1 - slippage)
                    pnl_b_gross = (eff_exit_b - eff_entry_b) * q_b
                    fee_b = (q_b * abs(eff_entry_b) + q_b * abs(eff_exit_b)) * trading_fee_rate
                
                active_trade['PnL Asset A (USD)'] = pnl_a_gross - fee_a
                active_trade['PnL Asset B (USD)'] = pnl_b_gross - fee_b
                active_trade['Total Fees (USD)'] = fee_a + fee_b
                active_trade['Net PnL (USD)'] = (pnl_a_gross - fee_a) + (pnl_b_gross - fee_b)
                
                # Calculate PnL % based on the initial notional value of the primary asset leg
                initial_primary_notional = 0
                if active_trade['Primary is A']:
                    initial_primary_notional = abs(q_a * entry_p_a) # Use original entry price for notional
                else:
                    initial_primary_notional = abs(q_b * entry_p_b)
                
                active_trade['PnL % (Primary Notional)'] = (active_trade['Net PnL (USD)'] / initial_primary_notional) * 100 if initial_primary_notional != 0 else 0

                trade_log_list.append(active_trade.copy())
                active_trade = None # Reset for the next trade
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
        data_with_prices_hedge_ratio=data_with_z_precalculated, # This df needs all price, hedge, and symbol info
        global_config=iter_config, # Contains slippage, trading_fee, and the optimized thresholds
        user_primary_quantity=quantity, 
        primary_is_A_selected=primary_is_A
    )

    if not detailed_trade_log.empty:
        total_pnl = detailed_trade_log['Net PnL (USD)'].sum()
        return total_pnl
    else:
        return -np.inf # Return negative infinity if no trades were made

def optimize_strategy_parameters(data_with_z_precalculated, base_config, quantity, primary_is_A):
    """
    Optimizes entry, exit, and stop-loss thresholds using grid search.
    `data_with_z_precalculated` is the DataFrame output from calculate_zscore, also containing prices, hedge_ratio, and symbols.
    """
    st.write("Starting parameter optimization...")

    # Define parameter ranges for optimization (can be made configurable later via UI)
    # Using round to avoid floating point precision issues in arange
    entry_thresholds = np.round(np.arange(0.8, 2.1, 0.2), 2)  # e.g., [0.8, 1.0, ..., 2.0]
    exit_thresholds = np.round(np.arange(0.1, 0.8, 0.1), 2)   # e.g., [0.1, 0.2, ..., 0.7]
    stop_loss_thresholds = np.round(np.arange(1.5, 3.6, 0.2), 2) # e.g., [1.5, 1.7, ..., 3.5]

    best_pnl = -np.inf
    best_params = {}
    iteration_count = 0
    
    # Calculate total iterations for progress bar, considering constraints
    valid_combinations = 0
    for entry_t in entry_thresholds:
        for exit_t in exit_thresholds:
            if exit_t >= entry_t: continue
            for stop_loss_t in stop_loss_thresholds:
                if stop_loss_t <= entry_t: continue # Stop loss should be wider than entry
                if stop_loss_t <= exit_t: continue # Stop loss should also be wider than profit taking exit
                valid_combinations +=1
    
    if valid_combinations == 0:
        st.warning("No valid parameter combinations to test with the defined ranges and constraints. Adjust ranges.")
        return None, -np.inf

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Optimizing... Total valid combinations to test: {valid_combinations}")


    for entry_t in entry_thresholds:
        for exit_t in exit_thresholds:
            # Constraint: Exit threshold for mean reversion should be smaller (closer to zero) than entry threshold
            if exit_t >= entry_t:
                continue

            for stop_loss_t in stop_loss_thresholds:
                # Constraint: Stop-loss threshold should be wider than entry threshold
                if stop_loss_t <= entry_t:
                    continue
                # Constraint: Stop-loss threshold should ideally be wider than profit-taking exit threshold too
                if stop_loss_t <= exit_t: # This means stop loss is tighter than profit taking, which is unusual
                    continue


                iteration_count += 1
                current_params = {
                    'entry_threshold': entry_t,
                    'exit_threshold': exit_t,
                    'stop_loss_threshold': stop_loss_t
                }
                
                # Update status less frequently to avoid slowing down Streamlit
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
    if iteration_count > 0 : progress_bar.progress(1.0) # Ensure progress bar completes
    else: progress_bar.empty()


    if best_pnl == -np.inf: # Check if any profitable trade was found
        st.warning("Optimization did not find any profitable trades with the tested parameter combinations.")
        return None, -np.inf
        
    return best_params, best_pnl


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📈 Statistical Arbitrage Trading Algorithm & Optimizer") # Updated title
st.markdown("""
This application implements a statistical arbitrage trading strategy for cryptocurrency pairs. 
Select assets, configure parameters, run the analysis, or optimize parameters to find the best Z-score thresholds.
Charts are interactive (zoom/pan). Default timeframe is daily.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration Parameters")

api_base_url = DEFAULT_CONFIG['api_base_url'] # Using default, can be made configurable
products_df = get_products(api_base_url)

if products_df.empty:
    st.sidebar.error("Could not fetch product list from API. Please check API URL or network.")
    st.stop()

# Create product options for selectbox
product_options = {f"{row['symbol']} ({row.get('description', 'N/A')})": row['id'] 
                   for _, row in products_df.iterrows() if 'symbol' in row and 'id' in row} # Ensure symbol and id exist
product_display_names = list(product_options.keys())

if not product_display_names:
    st.sidebar.error("No suitable products found from API to populate dropdowns.")
    st.stop()
    
# Default selections for assets
default_a_symbol_desc = next((s for s in product_display_names if "BTC" in s.upper()), product_display_names[0])
default_b_symbol_desc = next((s for s in product_display_names if "ETH" in s.upper() and s != default_a_symbol_desc), 
                             product_display_names[1] if len(product_display_names) > 1 else product_display_names[0])


asset_a_display = st.sidebar.selectbox("Asset A:", product_display_names, 
                                       index=product_display_names.index(default_a_symbol_desc) if default_a_symbol_desc in product_display_names else 0)
asset_b_display = st.sidebar.selectbox("Asset B:", product_display_names, 
                                       index=product_display_names.index(default_b_symbol_desc) if default_b_symbol_desc in product_display_names else (1 if len(product_display_names) > 1 else 0))

asset_a_id = product_options[asset_a_display]
asset_b_id = product_options[asset_b_display]

# Get symbols from products_df based on ID
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

# --- Action Buttons ---
col1, col2 = st.sidebar.columns(2)
run_button = col1.button("🚀 Run Analysis", use_container_width=True)
optimize_button = col2.button("⚙️ Optimize Params", use_container_width=True)


# --- Main Area for Outputs ---
if run_button or optimize_button: # Common data prep for both actions
    if asset_a_id == asset_b_id:
        st.error("Asset A and Asset B cannot be the same. Please select different assets.")
        st.stop()
    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        st.stop()

    # Prepare base configuration
    current_run_config = DEFAULT_CONFIG.copy()
    current_run_config.update({
        'timeframe': timeframe,
        'lookback_period': lookback_period, # In days
        'entry_threshold': entry_threshold, # From UI, used for single run or as part of base_config for opt.
        'exit_threshold': exit_threshold,   # From UI
        'stop_loss_threshold': stop_loss_threshold, # From UI
        'trading_fee': trading_fee,
        'slippage': slippage
    })

    ticker_config_selected = {
        'asset_a': {'id': asset_a_id, 'symbol': asset_a_symbol},
        'asset_b': {'id': asset_b_id, 'symbol': asset_b_symbol},
        'primary_asset': {'id': asset_a_id, 'symbol': asset_a_symbol} if primary_asset_choice == asset_a_symbol else {'id': asset_b_id, 'symbol': asset_b_symbol},
        'quantity': quantity, # User input quantity for primary asset
        'timeframe': timeframe,
        'start_timestamp': int(datetime.combine(start_date, datetime.min.time()).timestamp()),
        'end_timestamp': int(datetime.combine(end_date, datetime.max.time()).timestamp())
    }
    primary_is_A = ticker_config_selected['primary_asset']['id'] == ticker_config_selected['asset_a']['id']

    # --- Data Fetching and Preparation ---
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

    # Calculate log prices
    asset_a_data['log_price'] = np.log(asset_a_data['close'])
    asset_b_data['log_price'] = np.log(asset_b_data['close'])

    # Merge dataframes
    merged_df = pd.merge(
        asset_a_data[['close', 'log_price']].add_prefix('a_'),
        asset_b_data[['close', 'log_price']].add_prefix('b_'),
        left_index=True, right_index=True, how='inner'
    )
    merged_df.rename(columns={'a_close': 'close_a', 'a_log_price': 'log_price_a',
                              'b_close': 'close_b', 'b_log_price': 'log_price_b'}, inplace=True)
    
    # IMPORTANT: Add symbol columns here for generate_trade_log and optimization
    merged_df['symbol_a'] = asset_a_symbol
    merged_df['symbol_b'] = asset_b_symbol


    if merged_df.empty:
        st.warning("No overlapping data found for the selected assets and timeframe.")
        st.stop()

    # Calculate hedge ratio and Z-score (common for both analysis and optimization base)
    calc_spinner = st.spinner("Calculating hedge ratio, spread, and Z-score...")
    with calc_spinner:
        data_with_hedge = calculate_hedge_ratio(merged_df, current_run_config['lookback_period'], current_run_config['timeframe'])
        data_with_z = calculate_zscore(data_with_hedge, current_run_config['lookback_period'], current_run_config['timeframe'])
        # data_with_z now contains: close_a, log_price_a, close_b, log_price_b, symbol_a, symbol_b, hedge_ratio, spread, spread_ma, spread_std, zscore

    if data_with_z.empty or data_with_z['zscore'].isnull().all():
         st.warning("Could not calculate Z-scores. Check data or lookback period.")
         st.stop()

    # --- Action specific logic ---
    if run_button:
        st.header("📊 Analysis Results (Single Run)")
        st.subheader("📈 Price History (Interactive)")
        fig_prices_plotly = make_subplots(specs=[[{"secondary_y": True}]])
        fig_prices_plotly.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df['close_a'], name=asset_a_symbol, line=dict(color='blue')),
            secondary_y=False,
        )
        fig_prices_plotly.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df['close_b'], name=asset_b_symbol, line=dict(color='red')),
            secondary_y=True,
        )
        fig_prices_plotly.update_layout(title_text='Price History', xaxis_title='Date', legend_title_text='Assets')
        fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_a_symbol} Price</b>", secondary_y=False, color='blue') # type: ignore
        fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_b_symbol} Price</b>", secondary_y=True, color='red') # type: ignore
        st.plotly_chart(fig_prices_plotly, use_container_width=True)

        st.subheader("📉 Z-Score of the Spread with Trade Signals (Interactive)")
        trade_signals = generate_trade_signals(data_with_z, current_run_config) # Use UI params for single run
        
        fig_zscore_plotly = go.Figure()
        fig_zscore_plotly.add_trace(go.Scatter(x=data_with_z.index, y=data_with_z['zscore'], mode='lines', name='Z-score', line=dict(color='green', width=2)))
        # Threshold lines
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
        # Signals
        buy_signals_df = trade_signals[trade_signals['signal'] == 1.0]
        sell_signals_df = trade_signals[trade_signals['signal'] == -1.0]
        exit_signals_df = trade_signals[trade_signals['signal'] == 2.0]
        fig_zscore_plotly.add_trace(go.Scatter(x=buy_signals_df.index, y=buy_signals_df['zscore'], mode='markers', name=f'Buy Spread', marker=dict(color='green', size=10, symbol='triangle-up')))
        fig_zscore_plotly.add_trace(go.Scatter(x=sell_signals_df.index, y=sell_signals_df['zscore'], mode='markers', name=f'Sell Spread', marker=dict(color='red', size=10, symbol='triangle-down')))
        fig_zscore_plotly.add_trace(go.Scatter(x=exit_signals_df.index, y=exit_signals_df['zscore'], mode='markers', name='Exit Trade', marker=dict(color='blue', size=10, symbol='circle')))
        fig_zscore_plotly.update_layout(title_text='Z-score with Signals', xaxis_title='Date', yaxis_title='Z-score', legend_title_text='Signals & Levels')
        st.plotly_chart(fig_zscore_plotly, use_container_width=True)

        st.subheader("📜 Detailed Trade Log with P&L")
        # data_with_z already contains symbol_a and symbol_b from merged_df
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
        
        # Latest Trade Quantity Context (from original script)
        st.subheader("💡 Latest Trade Quantity Context")
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

            if pd.notna(hedge_ratio_latest_val) and primary_price_latest_val > 0 and secondary_price_latest_val > 0:
                secondary_qty_calc = calculate_secondary_quantity(
                    quantity, hedge_ratio_latest_val, primary_price_latest_val, secondary_price_latest_val, primary_is_A
                )
                st.write(f"**Latest Timestamp:** {latest_full_data_point.name.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Primary Asset:** {primary_asset_sym} at price ${primary_price_latest_val:.2f}")
                st.write(f"**Secondary Asset:** {secondary_asset_sym} at price ${secondary_price_latest_val:.2f}")
                st.write(f"**Hedge Ratio (β):** {hedge_ratio_latest_val:.4f}")
                st.write(f"**User Specified Primary Quantity:** {quantity}")
                st.write(f"**Calculated Secondary Quantity for hedge:** {secondary_qty_calc:.6f} of {secondary_asset_sym}")
                st.write(f"**Latest Z-score:** {latest_zscore_val:.4f}")

                trade_action_exp = "Hold / No new signal."
                # Determine action based on primary_is_A for correct asset order in explanation
                asset1_sym_for_exp = asset_a_symbol 
                asset2_sym_for_exp = asset_b_symbol
                qty1_for_exp = quantity if primary_is_A else secondary_qty_calc
                qty2_for_exp = secondary_qty_calc if primary_is_A else quantity

                if latest_signal_action_val == 1.0: # Buy Spread (Long A, Short B by definition of spread calculation)
                    trade_action_exp = f"Signal: BUY SPREAD. Action: LONG {qty1_for_exp:.6f} {asset1_sym_for_exp}, SHORT {qty2_for_exp:.6f} {asset2_sym_for_exp}."
                elif latest_signal_action_val == -1.0: # Sell Spread (Short A, Long B)
                    trade_action_exp = f"Signal: SELL SPREAD. Action: SHORT {qty1_for_exp:.6f} {asset1_sym_for_exp}, LONG {qty2_for_exp:.6f} {asset2_sym_for_exp}."
                elif latest_signal_action_val == 2.0:
                     trade_action_exp = "Signal: EXIT TRADE. Action: Close current spread position."
                st.write(f"**Latest Signal Interpretation:** {trade_action_exp}")
                st.write(f"**Position after this signal (if any):** {'Long Spread' if latest_position_val == 1 else ('Short Spread' if latest_position_val == -1 else 'No Position')}")
            else:
                st.warning("Could not calculate latest secondary quantity due to missing hedge ratio or zero prices.")
        else:
            st.warning("Could not display latest trade quantity context (no valid recent data).")


    elif optimize_button:
        st.header("🛠️ Parameter Optimization Results")
        # data_with_z is already prepared and contains all necessary columns including symbols
        best_params_found, max_pnl_found = optimize_strategy_parameters(
            data_with_z, # This is the precalculated DataFrame with prices, hedge, z-score, symbols
            current_run_config, # Base config with fees, slippage, lookback_period, timeframe etc.
            quantity, # User input quantity for primary asset
            primary_is_A # User input primary asset choice
        )

        if best_params_found:
            st.success(f"Optimization Complete!")
            st.subheader("🏆 Best Parameters Found:")
            # Display parameters in a more readable way
            st.markdown(f"- **Entry Z-score Threshold:** `{best_params_found['entry_threshold']:.2f}`")
            st.markdown(f"- **Exit Z-score Threshold (Mean Reversion):** `{best_params_found['exit_threshold']:.2f}`")
            st.markdown(f"- **Stop Loss Z-score Threshold:** `{best_params_found['stop_loss_threshold']:.2f}`")
            
            st.subheader(f"💰 Maximum P&L Achieved with these parameters: ${max_pnl_found:.2f}")
            
            st.markdown("---")
            st.markdown("You can now manually input these parameters in the sidebar and click '**Run Analysis**' to see the detailed charts and trade log for this optimal configuration.")
            
            # To automatically run analysis with best params, you'd need to update session state or re-run.
            # For simplicity, manual input is suggested.
        else:
            st.error("Optimization finished, but no profitable parameter set was found within the tested ranges. Try adjusting optimization ranges or data period.")


# --- Instructions for Running ---
st.sidebar.markdown("---")
st.sidebar.info("""
**How to Use:**
1. Select Asset A and Asset B.
2. Choose the Primary Asset and set its Quantity.
3. Adjust Timeframe, Date Range, and Strategy Parameters (for single run).
4. Click '**Run Analysis**' for a single backtest or '**Optimize Params**' to find optimal Z-score thresholds.
The app will fetch data, perform calculations, and display results.
""")