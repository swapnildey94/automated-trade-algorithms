# optimization.py
"""
Parameter optimization for statistical arbitrage backtesting.
"""
import numpy as np
import streamlit as st
from backtesting import generate_trade_signals, generate_trade_log

def run_backtest_for_optimization(params_dict, data_with_z_precalculated, base_config, quantity, primary_is_A):
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
                valid_combinations += 1
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
                if iteration_count % 10 == 0 or iteration_count == valid_combinations:
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
    if iteration_count > 0:
        progress_bar.progress(1.0)
    else:
        progress_bar.empty()
    if best_pnl == -np.inf:
        st.warning("Optimization did not find any profitable trades with the tested parameter combinations.")
        return None, -np.inf
    return best_params, best_pnl
