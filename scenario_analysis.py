# scenario_analysis.py
"""
Scenario analysis and what-if analysis functions for statistical arbitrage.
"""
import pandas as pd
import numpy as np
from regression import calculate_hedge_ratio, calculate_zscore
from backtesting import generate_trade_signals, generate_trade_log, calculate_secondary_quantity

def run_scenario_analysis(
    asset_a_data, asset_b_data, asset_a_symbol, asset_b_symbol, config, quantity, primary_is_A
):
    """
    Run scenario/what-if analysis for a given pair and config.
    Returns: dict with results (trade log, stats, etc.)
    """
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
        return {'error': 'No overlapping data found for the selected assets and timeframe.'}
    data_with_hedge = calculate_hedge_ratio(merged_df, config['lookback_period'], config['timeframe'])
    data_with_z = calculate_zscore(data_with_hedge, config['lookback_period'], config['timeframe'])
    if data_with_z.empty or data_with_z['zscore'].isnull().all():
        return {'error': 'Could not calculate Z-scores. Check data or lookback period.'}
    trade_signals = generate_trade_signals(data_with_z, config)
    detailed_trade_log = generate_trade_log(trade_signals, data_with_z, config, quantity, primary_is_A)
    return {
        'merged_df': merged_df,
        'data_with_hedge': data_with_hedge,
        'data_with_z': data_with_z,
        'trade_signals': trade_signals,
        'detailed_trade_log': detailed_trade_log
    }
