import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px # Added for correlation plot and OLS visualization
import argparse
# Modular imports
from regression import calculate_hedge_ratio, calculate_zscore
from backtesting import generate_trade_signals, generate_trade_log, calculate_secondary_quantity
from visualization import plot_price_history, plot_correlation
from scenario_analysis import run_scenario_analysis
from optimization import optimize_strategy_parameters
from cli import run_analysis_cli

# --- Configuration (from notebook, can be adjusted in UI) ---
DEFAULT_CONFIG = {
    'timeframe': '1h', # Changed default to hourly
    'lookback_period': 30, # days for hedge ratio/z-score calculation
    'entry_threshold': 2.2,
    'exit_threshold': 0.3,
    'final_exit_threshold': 0.1, # Not currently used in generate_trade_signals
    'stop_loss_threshold': 3.5,
    'trading_fee': 0.001,
    'slippage': 0.001,
    'start_date': (datetime.now() - timedelta(days=120)), # Extended default range for daily
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


# --- Main Area for Outputs ---
if run_button or optimize_button: 
    if asset_a_id == asset_b_id:
        st.error("Asset A and Asset B cannot be the same. Please select different assets.")
        st.stop()
    if start_date >= end_date:
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

    # Use scenario analysis module for all core calculations
    scenario_result = run_scenario_analysis(
        asset_a_data, asset_b_data, asset_a_symbol, asset_b_symbol, current_run_config, quantity, primary_is_A
    )
    if 'error' in scenario_result:
        st.warning(scenario_result['error'])
        st.stop()
    merged_df = scenario_result['merged_df']
    data_with_hedge = scenario_result['data_with_hedge']
    data_with_z = scenario_result['data_with_z']
    trade_signals = scenario_result['trade_signals']
    detailed_trade_log = scenario_result['detailed_trade_log']

    if data_with_z.empty or data_with_z['zscore'].isnull().all():
         st.warning("Could not calculate Z-scores. Check data or lookback period.")
         st.stop()

    # --- Action specific logic ---
    if run_button:
        st.header("📊 Analysis Results (Single Run)")
        
        # --- Price History Plot (Existing) ---
        st.subheader(f"📈 Price History: {asset_a_symbol} vs {asset_b_symbol}")
        fig_prices_plotly = plot_price_history(merged_df, asset_a_symbol, asset_b_symbol)
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
            context_data.append({"Parameter": "Latest Hedge Ratio (\u03B2)", "Value": f"{hedge_ratio_latest_val:.4f}" if pd.notna(hedge_ratio_latest_val) else "N/A"})
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

            # --- LOGGING LATEST CONTEXT ---
            try:
                from utils import log_latest_context
                log_latest_context(context_data)
            except Exception as e:
                st.warning(f"Could not log latest trade context: {e}")

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

# --- CLI Mode Functionality ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical Arbitrage CLI Runner")
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode (no Streamlit UI)')
    parser.add_argument('--asset_a', type=str, help='Symbol for Asset A (e.g., BTCUSD)')
    parser.add_argument('--asset_b', type=str, help='Symbol for Asset B (e.g., ETHUSD)')
    parser.add_argument('--quantity', type=float, default=1.0, help='Quantity of primary asset')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_CONFIG['timeframe'], help='Timeframe (e.g., 1d)')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--lookback', type=int, default=DEFAULT_CONFIG['lookback_period'], help='Lookback period (days)')
    parser.add_argument('--entry', type=float, default=DEFAULT_CONFIG['entry_threshold'], help='Entry Z-score threshold')
    parser.add_argument('--exit', type=float, default=DEFAULT_CONFIG['exit_threshold'], help='Exit Z-score threshold')
    parser.add_argument('--stop', type=float, default=DEFAULT_CONFIG['stop_loss_threshold'], help='Stop loss Z-score threshold')
    parser.add_argument('--fee', type=float, default=DEFAULT_CONFIG['trading_fee'], help='Trading fee')
    parser.add_argument('--slippage', type=float, default=DEFAULT_CONFIG['slippage'], help='Slippage')
    parser.add_argument('--api', type=str, default=DEFAULT_CONFIG['api_base_url'], help='API base URL')
    parser.add_argument('--log', type=str, default='trade_log_output.csv', help='Output log file path')
    args = parser.parse_args()
    if args.cli:
        # Parse dates
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else DEFAULT_CONFIG['start_date']
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else DEFAULT_CONFIG['end_date']
        except Exception as e:
            print(f"[ERROR] Invalid date format: {e}")
            exit(1)
        if not args.asset_a or not args.asset_b:
            print("[ERROR] --asset_a and --asset_b are required in CLI mode.")
            exit(1)
        exit(run_analysis_cli(
            args.asset_a, args.asset_b, args.quantity, args.timeframe, start_date, end_date,
            args.lookback, args.entry, args.exit, args.stop, args.fee, args.slippage, args.api, args.log,
            get_products, get_historical_candles, DEFAULT_CONFIG
        ))