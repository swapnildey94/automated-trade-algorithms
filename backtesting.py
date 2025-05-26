# backtesting.py
"""
Backtesting, trade signal generation, and optimization functions.
"""
import numpy as np
import pandas as pd

def calculate_secondary_quantity(primary_quantity, hedge_ratio, primary_price, secondary_price, primary_is_asset_a):
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
    trade_log_list = []
    active_trade = None
    slippage = global_config['slippage']
    trading_fee_rate = global_config['trading_fee']
    required_cols = ['close_a', 'close_b', 'hedge_ratio', 'symbol_a', 'symbol_b']
    if not all(col in data_with_prices_hedge_ratio.columns for col in required_cols):
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
