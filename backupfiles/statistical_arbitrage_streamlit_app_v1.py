import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # Matplotlib no longer primary for main charts
# import seaborn as sns # Seaborn style not directly applicable to Plotly in the same way
import requests
# import json # Not explicitly used in the final Streamlit script
from datetime import datetime, timedelta
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration (from notebook, can be adjusted in UI) ---
DEFAULT_CONFIG = {
    'timeframe': '1d', # Changed default to daily
    'lookback_period': 30, # days
    'entry_threshold': 1.1,
    'exit_threshold': 0.3,
    'final_exit_threshold': 0.1,
    'stop_loss_threshold': 3.1,
    'trading_fee': 0.001,
    'slippage': 0.001,
    'start_date': (datetime.now() - timedelta(days=365)), # Extended default range for daily
    'end_date': datetime.now(),
    'api_base_url': 'https://api.india.delta.exchange'
}

# --- API Functions (from notebook) ---
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
        df_filtered = df_products[df_products['contract_type'].isin(['perpetual_futures'])]
        if not df_filtered.empty:
            columns = ['id', 'symbol', 'description', 'contract_type']
            df_filtered = df_filtered[columns]
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
                df_candles[col] = pd.to_numeric(df_candles[col])
            df_candles = df_candles.sort_index()
        return df_candles
    except Exception as e:
        st.error(f"Error fetching historical candles for {product_symbol}: {e}")
        return pd.DataFrame()

# --- Statistical Arbitrage Algorithm Functions (largely unchanged) ---
def calculate_hedge_ratio(data_to_process, lookback_period_days, timeframe_selection):
    points_per_day = {
        '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2,
        '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1, '1w': 1/7
    }
    lookback_points = int(lookback_period_days * points_per_day.get(timeframe_selection, 24))
    if lookback_points < 2: lookback_points = 2

    data = data_to_process.copy()
    data['hedge_ratio'] = np.nan
    data['spread'] = np.nan

    for i in range(lookback_points, len(data)):
        y = data['log_price_a'].iloc[i-lookback_points:i]
        X_series = data['log_price_b'].iloc[i-lookback_points:i]
        
        if y.isnull().any() or X_series.isnull().any() or len(y) < 2 or len(X_series) < 2:
            continue
            
        X = sm.add_constant(X_series, prepend=True)
        
        try:
            model = sm.OLS(y, X).fit()
            if len(model.params) > 1:
                 beta = model.params.iloc[1]
                 data.loc[data.index[i], 'hedge_ratio'] = beta
                 data.loc[data.index[i], 'spread'] = data['log_price_a'].iloc[i] - beta * data['log_price_b'].iloc[i]
            else:
                data.loc[data.index[i], 'hedge_ratio'] = np.nan
        except Exception: 
            data.loc[data.index[i], 'hedge_ratio'] = np.nan
    return data

def calculate_zscore(data_with_spread, lookback_period_days, timeframe_selection):
    points_per_day = {
        '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2,
        '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1, '1w': 1/7
    }
    lookback_points = int(lookback_period_days * points_per_day.get(timeframe_selection, 24))
    if lookback_points < 2: lookback_points = 2

    result = data_with_spread.copy()
    result['spread_ma'] = result['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).mean()
    result['spread_std'] = result['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).std()
    result['zscore'] = (result['spread'] - result['spread_ma']) / result['spread_std']
    return result

def calculate_secondary_quantity(primary_quantity, hedge_ratio, primary_price, secondary_price, primary_is_asset_a):
    if primary_price == 0 or secondary_price == 0: return 0
    primary_notional = primary_quantity * primary_price
    if primary_is_asset_a:
        if secondary_price == 0: return 0 # Prevent division by zero
        secondary_notional_target = primary_notional * hedge_ratio
        secondary_asset_quantity = secondary_notional_target / secondary_price
    else:
        if hedge_ratio == 0 or secondary_price == 0: return 0 # Prevent division by zero
        secondary_notional_target = primary_notional / hedge_ratio
        secondary_asset_quantity = secondary_notional_target / secondary_price
    return secondary_asset_quantity

def generate_trade_signals(data_df, config_params):
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
        elif current_pos == 1: # Long Spread
            if (prev_z <= -exit_thresh and z > -exit_thresh) or \
               (prev_z >= -stop_loss_thresh and z < -stop_loss_thresh): # Stop Loss condition added
                signals.loc[signals.index[i], 'signal'] = 2.0
                current_pos = 0
        elif current_pos == -1: # Short Spread
            if (prev_z >= exit_thresh and z < exit_thresh) or \
               (prev_z <= stop_loss_thresh and z > stop_loss_thresh): # Stop Loss condition added
                signals.loc[signals.index[i], 'signal'] = 2.0
                current_pos = 0
        signals.loc[signals.index[i], 'position'] = current_pos
        
    signals['buy_signal_z'] = np.where(signals['signal'] == 1.0, signals['zscore'], np.nan)
    signals['sell_signal_z'] = np.where(signals['signal'] == -1.0, signals['zscore'], np.nan)
    signals['exit_signal_z'] = np.where(signals['signal'] == 2.0, signals['zscore'], np.nan)
    return signals

def generate_trade_log(signals_df, data_with_prices_hedge_ratio, global_config, user_primary_quantity, primary_is_A_selected):
    trade_log_list = []
    active_trade = None
    slippage = global_config['slippage']
    trading_fee_rate = global_config['trading_fee']

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
                    'Exit Timestamp': None, 'Exit Price A (Orig)': None, 'Exit Price B (Orig)': None, 'Exit Z-score': None,
                    'PnL Asset A (USD)': None, 'PnL Asset B (USD)': None, 'Total Fees (USD)': None, 
                    'Net PnL (USD)': None, 'PnL % (Primary Notional)': None
                }
        elif active_trade is not None:
            if signal == 2.0 or (timestamp == signals_df.index[-1] and active_trade['Exit Timestamp'] is None): # Ensure EOD closure happens only once
                active_trade['Exit Timestamp'] = timestamp
                active_trade['Exit Price A (Orig)'] = current_price_a
                active_trade['Exit Price B (Orig)'] = current_price_b
                active_trade['Exit Z-score'] = zscore_at_signal
                if timestamp == signals_df.index[-1] and signal != 2.0:
                     active_trade['Trade Type'] += ' (Closed at EOD)'
                
                entry_p_a = active_trade['Entry Price A (Orig)']
                entry_p_b = active_trade['Entry Price B (Orig)']
                exit_p_a = active_trade['Exit Price A (Orig)']
                exit_p_b = active_trade['Exit Price B (Orig)']
                q_a = active_trade['Qty A']
                q_b = active_trade['Qty B']
                pnl_a_gross, pnl_b_gross, fee_a, fee_b = 0,0,0,0

                if 'Buy Spread' in active_trade['Trade Type']: # Long A, Short B
                    eff_entry_a = entry_p_a * (1 + slippage)
                    eff_exit_a = exit_p_a * (1 - slippage)
                    pnl_a_gross = (eff_exit_a - eff_entry_a) * q_a
                    fee_a = (q_a * abs(eff_entry_a) + q_a * abs(eff_exit_a)) * trading_fee_rate # Use abs for notional value
                    
                    eff_entry_b_sell = entry_p_b * (1 - slippage)
                    eff_exit_b_buy = exit_p_b * (1 + slippage)
                    pnl_b_gross = (eff_entry_b_sell - eff_exit_b_buy) * q_b
                    fee_b = (q_b * abs(eff_entry_b_sell) + q_b * abs(eff_exit_b_buy)) * trading_fee_rate
                elif 'Sell Spread' in active_trade['Trade Type']: # Short A, Long B
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
                
                initial_primary_notional = (abs(q_a * entry_p_a)) if active_trade['Primary is A'] else (abs(q_b * entry_p_b))
                active_trade['PnL % (Primary Notional)'] = (active_trade['Net PnL (USD)'] / initial_primary_notional) * 100 if initial_primary_notional != 0 else 0

                trade_log_list.append(active_trade.copy())
                active_trade = None
    return pd.DataFrame(trade_log_list)


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📈 Statistical Arbitrage Trading Algorithm")
st.markdown("""
This application implements a statistical arbitrage trading strategy for cryptocurrency pairs. 
Select assets, configure parameters, and run the analysis to see potential trade signals and P&L. 
Charts are interactive (zoom/pan). Default timeframe is daily.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Configuration Parameters")

api_base_url = DEFAULT_CONFIG['api_base_url']
products_df = get_products(api_base_url)

if products_df.empty:
    st.sidebar.error("Could not fetch product list from API. Please check API URL or network.")
    st.stop()

product_options = {f"{row['symbol']} ({row['description']})": row['id'] for _, row in products_df.iterrows()}
product_display_names = list(product_options.keys())

default_a_symbol_desc = next((s for s in product_display_names if "BTCUSD" in s), product_display_names[0] if product_display_names else None)
default_b_symbol_desc = next((s for s in product_display_names if "ETHUSD" in s), product_display_names[1] if len(product_display_names) > 1 else (product_display_names[0] if product_display_names else None))

asset_a_display = st.sidebar.selectbox("Asset A:", product_display_names, index=product_display_names.index(default_a_symbol_desc) if default_a_symbol_desc and default_a_symbol_desc in product_display_names else 0)
asset_b_display = st.sidebar.selectbox("Asset B:", product_display_names, index=product_display_names.index(default_b_symbol_desc) if default_b_symbol_desc and default_b_symbol_desc in product_display_names else (1 if len(product_display_names) > 1 else 0))

asset_a_id = product_options[asset_a_display]
asset_b_id = product_options[asset_b_display]
asset_a_symbol = products_df[products_df['id'] == asset_a_id]['symbol'].iloc[0]
asset_b_symbol = products_df[products_df['id'] == asset_b_id]['symbol'].iloc[0]


primary_asset_choice = st.sidebar.radio("Primary Asset:", (asset_a_symbol, asset_b_symbol), key=f"primary_asset_{asset_a_symbol}_{asset_b_symbol}") # Dynamic key for radio
quantity = st.sidebar.number_input("Quantity of Primary Asset:", min_value=0.000001, value=1.0, step=0.001, format="%.6f")

timeframe_options = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
timeframe = st.sidebar.selectbox("Timeframe:", timeframe_options, index=timeframe_options.index(DEFAULT_CONFIG['timeframe'])) # Default '1d'

start_date = st.sidebar.date_input("Start Date:", value=DEFAULT_CONFIG['start_date'])
end_date = st.sidebar.date_input("End Date:", value=DEFAULT_CONFIG['end_date'])

st.sidebar.subheader("Strategy Parameters:")
lookback_period = st.sidebar.number_input("Lookback Period (candles for MA/StdDev):", min_value=2, value=DEFAULT_CONFIG['lookback_period']) # Changed label for clarity
entry_threshold = st.sidebar.number_input("Entry Z-score Threshold:", value=DEFAULT_CONFIG['entry_threshold'], step=0.1, format="%.1f")
exit_threshold = st.sidebar.number_input("Exit Z-score Threshold (Mean Reversion):", value=DEFAULT_CONFIG['exit_threshold'], step=0.1, format="%.1f")
stop_loss_threshold = st.sidebar.number_input("Stop Loss Z-score Threshold:", value=DEFAULT_CONFIG['stop_loss_threshold'], step=0.1, format="%.1f")

st.sidebar.subheader("Trading Costs:")
trading_fee = st.sidebar.number_input("Trading Fee (e.g., 0.001 for 0.1%):", value=DEFAULT_CONFIG['trading_fee'], step=0.0001, format="%.4f")
slippage = st.sidebar.number_input("Slippage (e.g., 0.001 for 0.1%):", value=DEFAULT_CONFIG['slippage'], step=0.0001, format="%.4f")

run_button = st.sidebar.button("🚀 Run Analysis")

# --- Main Area for Outputs ---
if run_button:
    if asset_a_id == asset_b_id:
        st.error("Asset A and Asset B cannot be the same. Please select different assets.")
    elif start_date >= end_date:
        st.error("Start Date must be before End Date.")
    else:
        st.header("📊 Analysis Results")
        
        current_run_config = DEFAULT_CONFIG.copy()
        current_run_config.update({
            'timeframe': timeframe,
            'lookback_period': lookback_period, # This is in days in the config, but used as #candles in functions. Adjusted label.
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

        with st.spinner("Fetching and preparing data..."):
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
        else:
            st.subheader("📈 Price History (Interactive)")
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
            else:
                fig_prices_plotly = make_subplots(specs=[[{"secondary_y": True}]])
                fig_prices_plotly.add_trace(
                    go.Scatter(x=merged_df.index, y=merged_df['close_a'], name=asset_a_symbol, line=dict(color='blue')),
                    secondary_y=False,
                )
                fig_prices_plotly.add_trace(
                    go.Scatter(x=merged_df.index, y=merged_df['close_b'], name=asset_b_symbol, line=dict(color='red')),
                    secondary_y=True,
                )
                fig_prices_plotly.update_layout(
                    title_text='Price History of Selected Assets',
                    xaxis_title='Date',
                    legend_title_text='Assets'
                )
                fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_a_symbol} Price</b>", secondary_y=False, color='blue') # type: ignore
                fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_b_symbol} Price</b>", secondary_y=True, color='red') # type: ignore
                st.plotly_chart(fig_prices_plotly, use_container_width=True)

                with st.spinner("Calculating hedge ratio, spread, and Z-score..."):
                    # lookback_period in config is in days. Functions expect # of candles.
                    # This translation is now done inside calculate_hedge_ratio and calculate_zscore
                    data_with_hedge = calculate_hedge_ratio(merged_df, current_run_config['lookback_period'], current_run_config['timeframe'])
                    data_with_z = calculate_zscore(data_with_hedge, current_run_config['lookback_period'], current_run_config['timeframe'])
                
                st.subheader("📉 Z-Score of the Spread with Trade Signals (Interactive)")
                if data_with_z.empty or data_with_z['zscore'].isnull().all():
                     st.warning("Could not calculate Z-scores. Check data or lookback period.")
                else:
                    trade_signals = generate_trade_signals(data_with_z, current_run_config)
                    
                    fig_zscore_plotly = go.Figure()
                    fig_zscore_plotly.add_trace(go.Scatter(x=data_with_z.index, y=data_with_z['zscore'], mode='lines', name='Z-score', line=dict(color='green', width=2)))

                    # Threshold lines
                    shapes = []
                    annotations = []
                    thresholds_to_plot = {
                        'Entry': (current_run_config['entry_threshold'], 'red'),
                        'Exit': (current_run_config['exit_threshold'], 'blue'),
                        'Stop Loss': (current_run_config['stop_loss_threshold'], 'purple')
                    }
                    for name, (val, color) in thresholds_to_plot.items():
                        # Positive threshold
                        shapes.append(dict(type='line', x0=data_with_z.index.min(), y0=val, x1=data_with_z.index.max(), y1=val, line=dict(color=color, width=1, dash='dash')))
                        annotations.append(dict(x=data_with_z.index.min(), y=val, xref="x", yref="y", text=f"{name} (+{val})", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=color)))
                        # Negative threshold
                        shapes.append(dict(type='line', x0=data_with_z.index.min(), y0=-val, x1=data_with_z.index.max(), y1=-val, line=dict(color=color, width=1, dash='dash')))
                        annotations.append(dict(x=data_with_z.index.min(), y=-val, xref="x", yref="y", text=f"{name} (-{val})", showarrow=False, xanchor="left", yanchor="top", font=dict(color=color)))
                    
                    fig_zscore_plotly.add_trace(go.Scatter(x=data_with_z.index, y=[0]*len(data_with_z), mode='lines', name='Mean (0)', line=dict(color='black', width=1)))

                    fig_zscore_plotly.update_layout(shapes=shapes, annotations=annotations)


                    # Signals
                    buy_signals_df = trade_signals[trade_signals['signal'] == 1.0]
                    sell_signals_df = trade_signals[trade_signals['signal'] == -1.0]
                    exit_signals_df = trade_signals[trade_signals['signal'] == 2.0]

                    fig_zscore_plotly.add_trace(go.Scatter(x=buy_signals_df.index, y=buy_signals_df['zscore'], mode='markers', name=f'Buy Spread (L {asset_a_symbol}, S {asset_b_symbol})', marker=dict(color='green', size=10, symbol='triangle-up')))
                    fig_zscore_plotly.add_trace(go.Scatter(x=sell_signals_df.index, y=sell_signals_df['zscore'], mode='markers', name=f'Sell Spread (S {asset_a_symbol}, L {asset_b_symbol})', marker=dict(color='red', size=10, symbol='triangle-down')))
                    fig_zscore_plotly.add_trace(go.Scatter(x=exit_signals_df.index, y=exit_signals_df['zscore'], mode='markers', name='Exit Trade', marker=dict(color='blue', size=10, symbol='circle')))

                    fig_zscore_plotly.update_layout(
                        title_text='Z-score of the Spread with Trade Signals',
                        xaxis_title='Date',
                        yaxis_title='Z-score',
                        legend_title_text='Signals & Levels'
                    )
                    st.plotly_chart(fig_zscore_plotly, use_container_width=True)

                    st.subheader("📜 Detailed Trade Log with P&L")
                    data_for_log = data_with_z.copy()
                    if 'symbol_a' not in data_for_log.columns: data_for_log['symbol_a'] = asset_a_symbol
                    if 'symbol_b' not in data_for_log.columns: data_for_log['symbol_b'] = asset_b_symbol
                    
                    detailed_trade_log = generate_trade_log(trade_signals, data_for_log, current_run_config, quantity, primary_is_A) # Pass current_run_config
                    if not detailed_trade_log.empty:
                        st.dataframe(detailed_trade_log.style.format({
                            "Entry Z-score": "{:.2f}", "Exit Z-score": "{:.2f}",
                            "Entry Price A (Orig)": "{:.2f}", "Entry Price B (Orig)": "{:.2f}",
                            "Exit Price A (Orig)": "{:.2f}", "Exit Price B (Orig)": "{:.2f}",
                            "Entry Hedge Ratio": "{:.4f}",
                            "Qty A": "{:.6f}", "Qty B": "{:.6f}",
                            "PnL Asset A (USD)": "{:.2f}", "PnL Asset B (USD)": "{:.2f}",
                            "Total Fees (USD)": "{:.2f}", "Net PnL (USD)": "{:.2f}",
                            "PnL % (Primary Notional)": "{:.2f}%"
                        }))
                        st.markdown(f"**Overall Net P&L: ${detailed_trade_log['Net PnL (USD)'].sum():.2f}**")
                        st.markdown(f"**Total Trades: {len(detailed_trade_log)}**")
                    else:
                        st.info("No trades were generated based on the current signals and data.")
                    
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
                            st.write(f"**Hedge Ratio (β for log_price_a vs log_price_b):** {hedge_ratio_latest_val:.4f}")
                            st.write(f"**User Specified Primary Quantity:** {quantity}")
                            st.write(f"**Calculated Secondary Quantity for hedge:** {secondary_qty_calc:.6f} of {secondary_asset_sym}")
                            st.write(f"**Latest Z-score:** {latest_zscore_val:.4f}")

                            trade_action_exp = "Hold / No new signal."
                            if latest_signal_action_val == 1.0:
                                trade_action_exp = f"Signal: BUY SPREAD. Action: LONG {quantity:.6f} {primary_asset_sym}, SHORT {secondary_qty_calc:.6f} {secondary_asset_sym}." if primary_is_A else f"Signal: BUY SPREAD. Action: SHORT {quantity:.6f} {primary_asset_sym}, LONG {secondary_qty_calc:.6f} {secondary_asset_sym}."
                            elif latest_signal_action_val == -1.0:
                                trade_action_exp = f"Signal: SELL SPREAD. Action: SHORT {quantity:.6f} {primary_asset_sym}, LONG {secondary_qty_calc:.6f} {secondary_asset_sym}." if primary_is_A else f"Signal: SELL SPREAD. Action: LONG {quantity:.6f} {primary_asset_sym}, SHORT {secondary_qty_calc:.6f} {secondary_asset_sym}."
                            elif latest_signal_action_val == 2.0:
                                 trade_action_exp = "Signal: EXIT TRADE. Action: Close current spread position."
                            st.write(f"**Latest Signal Interpretation:** {trade_action_exp}")
                            st.write(f"**Position after this signal (if any):** {'Long Spread' if latest_position_val == 1 else ('Short Spread' if latest_position_val == -1 else 'No Position')}")
                        else:
                            st.warning("Could not calculate latest secondary quantity due to missing hedge ratio or zero prices.")
                    else:
                        st.warning("Could not display latest trade quantity context (no valid recent data).")

# --- Instructions for Running ---
st.sidebar.markdown("---")
st.sidebar.info("""
**How to Use:**
1. Select Asset A and Asset B.
2. Choose the Primary Asset and set its Quantity.
3. Adjust Timeframe (default daily), Date Range, and Strategy Parameters.
4. Click 'Run Analysis'.
The app will fetch data, perform calculations, and display interactive charts and a trade log.
""")