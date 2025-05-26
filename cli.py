# cli.py
"""
Command-line interface for statistical arbitrage app.
"""
import argparse
import numpy as np
import pandas as pd
from regression import calculate_hedge_ratio, calculate_zscore
from backtesting import generate_trade_signals, generate_trade_log
from datetime import datetime

def run_analysis_cli(
    asset_a_symbol, asset_b_symbol, quantity, timeframe, start_date, end_date,
    lookback_period, entry_threshold, exit_threshold, stop_loss_threshold, trading_fee, slippage, api_base_url, log_file_path,
    get_products, get_historical_candles, DEFAULT_CONFIG
):
    products_df = get_products(api_base_url)
    if products_df.empty:
        print("[ERROR] Could not fetch product list from API.")
        return 1
    asset_a_id = products_df.loc[products_df['symbol'] == asset_a_symbol, 'id'].iloc[0] if asset_a_symbol in products_df['symbol'].values else None
    asset_b_id = products_df.loc[products_df['symbol'] == asset_b_symbol, 'id'].iloc[0] if asset_b_symbol in products_df['symbol'].values else None
    if not asset_a_id or not asset_b_id or asset_a_id == asset_b_id:
        print("[ERROR] Invalid asset symbols or same asset selected.")
        return 1
    start_timestamp = int(pd.Timestamp(start_date).replace(hour=0, minute=0, second=0).timestamp())
    end_timestamp = int(pd.Timestamp(end_date).replace(hour=23, minute=59, second=59).timestamp())
    asset_a_data = get_historical_candles(api_base_url, asset_a_id, asset_a_symbol, timeframe, start_timestamp, end_timestamp)
    asset_b_data = get_historical_candles(api_base_url, asset_b_id, asset_b_symbol, timeframe, start_timestamp, end_timestamp)
    if asset_a_data.empty or asset_b_data.empty:
        print("[ERROR] Failed to fetch data for one or both assets.")
        return 1
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
        print("[ERROR] No overlapping data found for the selected assets and timeframe.")
        return 1
    data_with_hedge = calculate_hedge_ratio(merged_df, lookback_period, timeframe)
    data_with_z = calculate_zscore(data_with_hedge, lookback_period, timeframe)
    if data_with_z.empty or data_with_z['zscore'].isnull().all():
        print("[ERROR] Could not calculate Z-scores. Check data or lookback period.")
        return 1
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
    primary_is_A = True  # For CLI, default to A as primary
    trade_signals = generate_trade_signals(data_with_z, current_run_config)
    detailed_trade_log = generate_trade_log(trade_signals, data_with_z, current_run_config, quantity, primary_is_A)
    if detailed_trade_log.empty:
        print("[INFO] No trades generated.")
    else:
        detailed_trade_log.to_csv(log_file_path, index=False)
        print(f"[SUCCESS] Trade log written to {log_file_path}")
        print(f"[SUMMARY] Net P&L: ${detailed_trade_log['Net PnL (USD)'].sum():.2f} | Total Trades: {len(detailed_trade_log)}")

    # --- LOGGING LATEST CONTEXT ---
    try:
        from utils import log_latest_context
        # Build context_data similar to Streamlit app
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
            context_data.append({"Parameter": "--- Strategy Parameters ---", "Value": "--- ---"})
            context_data.append({"Parameter": "Entry Z-score Threshold", "Value": f"{current_run_config['entry_threshold']:.2f}"})
            context_data.append({"Parameter": "Exit Z-score Threshold", "Value": f"{current_run_config['exit_threshold']:.2f}"})
            context_data.append({"Parameter": "Stop Loss Z-score Threshold", "Value": f"{current_run_config['stop_loss_threshold']:.2f}"})
            context_data.append({"Parameter": "--- Trade Setup ---", "Value": "--- ---"})
            context_data.append({"Parameter": "User Specified Primary Quantity", "Value": f"{quantity} {primary_asset_sym}"})
            from backtesting import calculate_secondary_quantity
            if pd.notna(hedge_ratio_latest_val) and primary_price_latest_val > 0 and secondary_price_latest_val > 0:
                secondary_qty_calc = calculate_secondary_quantity(
                    quantity, hedge_ratio_latest_val, primary_price_latest_val, secondary_price_latest_val, primary_is_A
                )
                context_data.append({"Parameter": "Calculated Secondary Quantity", "Value": f"{secondary_qty_calc:.6f} {secondary_asset_sym}"})
            else:
                context_data.append({"Parameter": "Calculated Secondary Quantity", "Value": "N/A (Invalid inputs for calculation)"})
            log_latest_context(context_data)
    except Exception as e:
        print(f"[WARNING] Could not log latest trade context: {e}")
    return 0
