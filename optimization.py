# optimization.py
"""
Parameter optimization for statistical arbitrage backtesting.
Using Backtrader's built-in optimization capabilities.
"""
import numpy as np
import pandas as pd
import streamlit as st
import backtrader as bt
import itertools
import concurrent.futures
from backtesting import PairTradingStrategy, PairTradingData, run_backtest_with_metrics

class OptimizationResult:
    """Class to store optimization results for each parameter combination"""
    def __init__(self, params, metrics):
        self.params = params
        self.metrics = metrics
        
        # Extract key metrics
        self.total_pnl = metrics.get('total_pnl', -np.inf)
        self.sharpe_ratio = metrics.get('sharpe_ratio', -np.inf)
        self.sortino_ratio = metrics.get('sortino_ratio', -np.inf)
        self.max_drawdown = metrics.get('max_drawdown', 100.0)
        self.win_rate = metrics.get('win_rate', 0.0)
        self.total_trades = metrics.get('total_trades', 0)

def run_backtest_for_optimization(params_dict, data_with_z_precalculated, base_config, quantity, primary_is_A):
    """
    Run a backtest with specific parameters and return the total PnL.
    This function is compatible with the original optimization workflow.
    """
    iter_config = base_config.copy()
    iter_config['entry_threshold'] = params_dict['entry_threshold']
    iter_config['exit_threshold'] = params_dict['exit_threshold']
    iter_config['stop_loss_threshold'] = params_dict['stop_loss_threshold']
    
    # Run backtest with metrics
    result = run_backtest_with_metrics(
        data_df=data_with_z_precalculated,
        config_params=iter_config,
        user_primary_quantity=quantity,
        primary_is_A_selected=primary_is_A
    )
    
    # Extract metrics
    metrics = result.get('metrics', {})
    total_pnl = metrics.get('total_pnl', -np.inf)
    
    return total_pnl

def run_single_backtest(params, data, base_config, quantity, primary_is_A):
    """
    Run a single backtest with the given parameters.
    This function is designed to be used with parallel processing.
    """
    # Extract parameters
    entry_threshold = params[0]
    exit_threshold = params[1]
    stop_loss_threshold = params[2]
    
    # Create a copy of the DataFrame to avoid modifying the original
    data_copy = data.copy()
    
    # Extract symbol information
    symbol_a = data_copy['symbol_a'].iloc[0] if 'symbol_a' in data_copy.columns else "Asset A"
    symbol_b = data_copy['symbol_b'].iloc[0] if 'symbol_b' in data_copy.columns else "Asset B"
    
    # Create a copy without string columns that Backtrader can't handle
    numeric_data = data_copy.copy()
    if 'symbol_a' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_a', axis=1)
    if 'symbol_b' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_b', axis=1)
    
    # Create a Backtrader cerebro engine
    cerebro = bt.Cerebro()
    
    # Add the strategy with symbols as parameters
    cerebro.addstrategy(
        PairTradingStrategy,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss_threshold=stop_loss_threshold,
        slippage=base_config.get('slippage', 0.001),
        trading_fee=base_config.get('trading_fee', 0.001),
        primary_quantity=quantity,
        primary_is_a=primary_is_A,
        symbol_a=symbol_a,
        symbol_b=symbol_b
    )
    
    # Prepare the data feed
    data_feed = PairTradingData(
        dataname=numeric_data,
        datetime=None,  # Use index as datetime
        zscore='zscore' if 'zscore' in numeric_data.columns else None,
        hedge_ratio='hedge_ratio' if 'hedge_ratio' in numeric_data.columns else None,
        close_a='close_a' if 'close_a' in numeric_data.columns else None,
        close_b='close_b' if 'close_b' in numeric_data.columns else None,
    )
    
    # Add the data feed to cerebro
    cerebro.adddata(data_feed)
    
    # Run the backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Create parameter dictionary
    params_dict = {
        'entry_threshold': entry_threshold,
        'exit_threshold': exit_threshold,
        'stop_loss_threshold': stop_loss_threshold
    }
    
    # Return optimization result
    return OptimizationResult(
        params=params_dict,
        metrics=strategy.metrics if hasattr(strategy, 'metrics') else {}
    )

def optimize_strategy_parameters(data_with_z_precalculated, base_config, quantity, primary_is_A):
    """
    Optimize strategy parameters using parallel processing for faster results.
    Returns the best parameters and metrics based on total PnL.
    """
    st.write("Starting parameter optimization with Backtrader...")

    # --- Session state keys for caching ---
    cache_key = f"opt_results_{hash(str(base_config))}_{quantity}_{primary_is_A}"
    df_key = f"opt_results_df_{hash(str(base_config))}_{quantity}_{primary_is_A}"

    # Check if results are already cached
    if cache_key in st.session_state and df_key in st.session_state:
        results = st.session_state[cache_key]
        all_results_df = st.session_state[df_key]
    else:
        # Define parameter ranges
        entry_thresholds = np.round(np.arange(0.8, 2.1, 0.2), 2)
        exit_thresholds = np.round(np.arange(0.1, 0.8, 0.1), 2)
        stop_loss_thresholds = np.round(np.arange(1.5, 3.6, 0.2), 2)
        
        # Generate valid parameter combinations
        valid_combinations = []
        for entry_t in entry_thresholds:
            for exit_t in exit_thresholds:
                if exit_t >= entry_t:
                    continue
                for stop_loss_t in stop_loss_thresholds:
                    if stop_loss_t <= entry_t or stop_loss_t <= exit_t:
                        continue
                    valid_combinations.append((entry_t, exit_t, stop_loss_t))
        
        if not valid_combinations:
            st.warning("No valid parameter combinations to test with the defined ranges and constraints. Adjust ranges.")
            return None, -np.inf
        
        st.write(f"Testing {len(valid_combinations)} valid parameter combinations...")
        
        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Optimizing... Total valid combinations to test: {len(valid_combinations)}")
        
        # Run optimizations in parallel
        results = []
        completed = 0
        
        # Determine the number of workers based on system capabilities
        max_workers = min(8, len(valid_combinations))  # Limit to 8 workers maximum
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(
                    run_single_backtest, 
                    params, 
                    data_with_z_precalculated, 
                    base_config, 
                    quantity, 
                    primary_is_A
                ): params for params in valid_combinations
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    st.error(f"Error with parameters {params}: {e}")
                
                # Update progress
                completed += 1
                progress_bar.progress(completed / len(valid_combinations))
                
                # Update status text periodically
                if completed % 5 == 0 or completed == len(valid_combinations):
                    # Find current best result
                    if results:
                        current_best = max(results, key=lambda x: x.total_pnl)
                        current_best_pnl = f"${current_best.total_pnl:.2f}" if current_best.total_pnl != -np.inf else "N/A"
                        current_best_params = current_best.params
                        param_str = ", ".join([f"{k}: {v}" for k, v in current_best_params.items()])
                        status_text.text(f"Optimizing: {completed}/{len(valid_combinations)} | Current Best PnL: {current_best_pnl} with {param_str}")
        
        # Find the best result
        if not results:
            st.warning("Optimization did not find any profitable trades with the tested parameter combinations.")
            return None, -np.inf
        
        # Sort results by total PnL
        results.sort(key=lambda x: x.total_pnl, reverse=True)
        best_result = results[0]
        
        # --- Build DataFrame and cache ---
        all_results_df = pd.DataFrame([
            {
                'Entry Threshold': r.params['entry_threshold'],
                'Exit Threshold': r.params['exit_threshold'],
                'Stop Loss': r.params['stop_loss_threshold'],
                'Total PnL': r.total_pnl,
                'Sharpe Ratio': r.sharpe_ratio,
                'Sortino Ratio': r.sortino_ratio,
                'Max Drawdown': r.max_drawdown,
                'Win Rate': r.win_rate,
                'Trades': r.total_trades,
                'Metrics': r.metrics
            }
            for r in results
        ])
        st.session_state[cache_key] = results
        st.session_state[df_key] = all_results_df

    best_result = results[0]
    
    # Display optimization results
    st.write("### Optimization Results")
    
    # Create a DataFrame with the top results
    top_n = min(10, len(results))
    top_results = []
    for i in range(top_n):
        result = results[i]
        top_results.append({
            'Rank': i + 1,
            'Entry Threshold': result.params['entry_threshold'],
            'Exit Threshold': result.params['exit_threshold'],
            'Stop Loss': result.params['stop_loss_threshold'],
            'Total PnL': f"${result.total_pnl:.2f}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{result.sortino_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown:.2f}%",
            'Win Rate': f"{result.win_rate*100:.1f}%",
            'Trades': result.total_trades
        })
    
    # Display the top results table
    st.table(pd.DataFrame(top_results))
    
    # --- Show head and tail of all parameter results ---
    all_results_df = pd.DataFrame([
        {
            'Entry Threshold': r.params['entry_threshold'],
            'Exit Threshold': r.params['exit_threshold'],
            'Stop Loss': r.params['stop_loss_threshold'],
            'Total PnL': r.total_pnl,
            'Sharpe Ratio': r.sharpe_ratio,
            'Sortino Ratio': r.sortino_ratio,
            'Max Drawdown': r.max_drawdown,
            'Win Rate': r.win_rate,
            'Trades': r.total_trades,
            'Metrics': r.metrics  # Keep metrics for later use
        }
        for r in results
    ])

    st.write("#### Head of all parameter results")
    st.dataframe(all_results_df.head())

    st.write("#### Tail of all parameter results")
    st.dataframe(all_results_df.tail())

    st.write("#### View trades for a specific parameter set")
    selected_idx = st.number_input(
        "Select row index from the table above to view trades (0-based):",
        min_value=0, max_value=len(all_results_df)-1, value=0, step=1,
        key=f"trade_row_selector_{df_key}"
    )
    selected_params = all_results_df.iloc[selected_idx][['Entry Threshold', 'Exit Threshold', 'Stop Loss']].to_dict()
    params_dict = {
        'entry_threshold': selected_params['Entry Threshold'],
        'exit_threshold': selected_params['Exit Threshold'],
        'stop_loss_threshold': selected_params['Stop Loss']
    }
    iter_config = base_config.copy()
    iter_config.update(params_dict)
    result = run_backtest_with_metrics(
        data_df=data_with_z_precalculated,
        config_params=iter_config,
        user_primary_quantity=quantity,
        primary_is_A_selected=primary_is_A
    )
    trade_log = result.get('trade_log', pd.DataFrame())
    if not trade_log.empty:
        st.write("##### Trades for selected parameter set")
        st.dataframe(trade_log)
    else:
        st.info("No trades for this parameter set.")
    return best_result.params, best_result.total_pnl

def optimize_with_objective(data_with_z_precalculated, base_config, quantity, primary_is_A, objective='pnl'):
    """
    Optimize strategy parameters with different objective functions.
    
    Parameters:
    - objective: The optimization objective ('pnl', 'sharpe', 'sortino', 'calmar')
    
    Returns the best parameters and metrics based on the selected objective.
    """
    st.write(f"Starting parameter optimization with objective: {objective.upper()}")
    cache_key = f"optobj_results_{objective}_{hash(str(base_config))}_{quantity}_{primary_is_A}"
    df_key = f"optobj_results_df_{objective}_{hash(str(base_config))}_{quantity}_{primary_is_A}"
    if cache_key in st.session_state and df_key in st.session_state:
        results = st.session_state[cache_key]
        all_results_df = st.session_state[df_key]
    else:
        # Define parameter ranges
        entry_thresholds = np.round(np.arange(0.8, 2.1, 0.2), 2)
        exit_thresholds = np.round(np.arange(0.1, 0.8, 0.1), 2)
        stop_loss_thresholds = np.round(np.arange(1.5, 3.6, 0.2), 2)
        
        # Generate valid parameter combinations
        valid_combinations = []
        for entry_t in entry_thresholds:
            for exit_t in exit_thresholds:
                if exit_t >= entry_t:
                    continue
                for stop_loss_t in stop_loss_thresholds:
                    if stop_loss_t <= entry_t or stop_loss_t <= exit_t:
                        continue
                    valid_combinations.append((entry_t, exit_t, stop_loss_t))
        
        if not valid_combinations:
            st.warning("No valid parameter combinations to test with the defined ranges and constraints. Adjust ranges.")
            return None, None
        
        st.write(f"Testing {len(valid_combinations)} valid parameter combinations...")
        
        # Set up progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"Optimizing... Total valid combinations to test: {len(valid_combinations)}")
        
        # Run optimizations in parallel
        results = []
        completed = 0
        
        # Determine the number of workers based on system capabilities
        max_workers = min(8, len(valid_combinations))  # Limit to 8 workers maximum
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(
                    run_single_backtest, 
                    params, 
                    data_with_z_precalculated, 
                    base_config, 
                    quantity, 
                    primary_is_A
                ): params for params in valid_combinations
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    st.error(f"Error with parameters {params}: {e}")
                
                # Update progress
                completed += 1
                progress_bar.progress(completed / len(valid_combinations))
                
                # Update status text periodically
                if completed % 5 == 0 or completed == len(valid_combinations):
                    # Find current best result based on objective
                    if results:
                        if objective == 'pnl':
                            current_best = max(results, key=lambda x: x.total_pnl)
                            metric_value = f"${current_best.total_pnl:.2f}" if current_best.total_pnl != -np.inf else "N/A"
                        elif objective == 'sharpe':
                            current_best = max(results, key=lambda x: x.sharpe_ratio)
                            metric_value = f"{current_best.sharpe_ratio:.2f}" if current_best.sharpe_ratio != -np.inf else "N/A"
                        elif objective == 'sortino':
                            current_best = max(results, key=lambda x: x.sortino_ratio)
                            metric_value = f"{current_best.sortino_ratio:.2f}" if current_best.sortino_ratio != -np.inf else "N/A"
                        elif objective == 'calmar':
                            valid_results = [r for r in results if r.max_drawdown > 0]
                            if valid_results:
                                current_best = max(valid_results, key=lambda x: x.metrics.get('calmar_ratio', -np.inf))
                                metric_value = f"{current_best.metrics.get('calmar_ratio', 0):.2f}"
                            else:
                                current_best = results[0]
                                metric_value = "N/A"
                        else:
                            current_best = max(results, key=lambda x: x.total_pnl)
                            metric_value = f"${current_best.total_pnl:.2f}" if current_best.total_pnl != -np.inf else "N/A"
                        current_best_params = current_best.params
                        param_str = ", ".join([f"{k}: {v}" for k, v in current_best_params.items()])
                        status_text.text(f"Optimizing: {completed}/{len(valid_combinations)} | Current Best {objective.upper()}: {metric_value} with {param_str}")
        
        # Find the best result based on the objective
        if not results:
            st.warning("Optimization did not find any valid results with the tested parameter combinations.")
            return None, None
        
        # Sort results based on the selected objective
        if objective == 'pnl':
            results.sort(key=lambda x: x.total_pnl, reverse=True)
        elif objective == 'sharpe':
            results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        elif objective == 'sortino':
            results.sort(key=lambda x: x.sortino_ratio, reverse=True)
        elif objective == 'calmar':
            valid_results = [r for r in results if r.max_drawdown > 0]
            if valid_results:
                valid_results.sort(key=lambda x: x.metrics.get('calmar_ratio', -np.inf), reverse=True)
                results = valid_results
            else:
                results.sort(key=lambda x: x.total_pnl, reverse=True)
        else:
            results.sort(key=lambda x: x.total_pnl, reverse=True)
        
        all_results_df = pd.DataFrame([
            {
                'Entry Threshold': r.params['entry_threshold'],
                'Exit Threshold': r.params['exit_threshold'],
                'Stop Loss': r.params['stop_loss_threshold'],
                'Total PnL': r.total_pnl,
                'Sharpe Ratio': r.sharpe_ratio,
                'Sortino Ratio': r.sortino_ratio,
                'Max Drawdown': r.max_drawdown,
                'Win Rate': r.win_rate,
                'Trades': r.total_trades,
                'Metrics': r.metrics
            }
            for r in results
        ])
        st.session_state[cache_key] = results
        st.session_state[df_key] = all_results_df
    best_result = results[0]
    
    # Display optimization results
    st.write(f"### Optimization Results (Optimized for {objective.upper()})")
    
    # Create a DataFrame with the top results
    top_n = min(10, len(results))
    top_results = []
    for i in range(top_n):
        result = results[i]
        top_results.append({
            'Rank': i + 1,
            'Entry Threshold': result.params['entry_threshold'],
            'Exit Threshold': result.params['exit_threshold'],
            'Stop Loss': result.params['stop_loss_threshold'],
            'Total PnL': f"${result.total_pnl:.2f}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{result.sortino_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown:.2f}%",
            'Win Rate': f"{result.win_rate*100:.1f}%",
            'Trades': result.total_trades
        })
    
    # Display the top results table
    st.table(pd.DataFrame(top_results))
    
    # --- Show head and tail of all parameter results ---
    all_results_df = pd.DataFrame([
        {
            'Entry Threshold': r.params['entry_threshold'],
            'Exit Threshold': r.params['exit_threshold'],
            'Stop Loss': r.params['stop_loss_threshold'],
            'Total PnL': r.total_pnl,
            'Sharpe Ratio': r.sharpe_ratio,
            'Sortino Ratio': r.sortino_ratio,
            'Max Drawdown': r.max_drawdown,
            'Win Rate': r.win_rate,
            'Trades': r.total_trades,
            'Metrics': r.metrics  # Keep metrics for later use
        }
        for r in results
    ])

    st.write("#### Head of all parameter results")
    st.dataframe(all_results_df.head())

    st.write("#### Tail of all parameter results")
    st.dataframe(all_results_df.tail())

    st.write("#### View trades for a specific parameter set")
    selected_idx = st.number_input(
        "Select row index from the table above to view trades (0-based):",
        min_value=0, max_value=len(all_results_df)-1, value=0, step=1,
        key=f"trade_row_selector_{df_key}"
    )
    selected_params = all_results_df.iloc[selected_idx][['Entry Threshold', 'Exit Threshold', 'Stop Loss']].to_dict()
    params_dict = {
        'entry_threshold': selected_params['Entry Threshold'],
        'exit_threshold': selected_params['Exit Threshold'],
        'stop_loss_threshold': selected_params['Stop Loss']
    }
    iter_config = base_config.copy()
    iter_config.update(params_dict)
    result = run_backtest_with_metrics(
        data_df=data_with_z_precalculated,
        config_params=iter_config,
        user_primary_quantity=quantity,
        primary_is_A_selected=primary_is_A
    )
    trade_log = result.get('trade_log', pd.DataFrame())
    if not trade_log.empty:
        st.write("##### Trades for selected parameter set")
        st.dataframe(trade_log)
    else:
        st.info("No trades for this parameter set.")
    return best_result.params, best_result.metrics
