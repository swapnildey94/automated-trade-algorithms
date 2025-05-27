import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization import run_backtest_for_optimization, optimize_strategy_parameters
# For patching, we need to target where the names are looked up.
# optimization.py imports:
# import streamlit as st
# from backtesting import generate_trade_signals, generate_trade_log
# So, we patch 'optimization.generate_trade_signals', 'optimization.generate_trade_log', and 'optimization.st'

class TestRunBacktestForOptimization(unittest.TestCase):
    def setUp(self):
        self.params_dict = {'entry_threshold': 1.5, 'exit_threshold': 0.5, 'stop_loss_threshold': 2.0}
        # data_with_z_precalculated needs all columns required by generate_trade_log
        # i.e., close_a, close_b, hedge_ratio, symbol_a, symbol_b, zscore
        self.data_with_z_and_prices = pd.DataFrame({
            'zscore': [1.0, 1.6, 0.4, 2.1],
            'close_a': [100,101,102,103],
            'close_b': [50,51,52,53],
            'hedge_ratio': [0.5,0.5,0.5,0.5],
            'symbol_a': ['A','A','A','A'],
            'symbol_b': ['B','B','B','B'],
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=4, freq='D'))
        }).set_index('timestamp')
        
        self.base_config = {'some_base_param': 'value', 'slippage': 0.001, 'trading_fee': 0.001} # Ensure base_config has required keys for generate_trade_log if not overridden
        self.quantity = 10.0
        self.primary_is_A = True

    @patch('optimization.generate_trade_log')
    @patch('optimization.generate_trade_signals')
    def test_successful_backtest_run(self, mock_generate_signals, mock_generate_log):
        mock_signals_df = pd.DataFrame({'zscore': [1,2,3]}) 
        mock_generate_signals.return_value = mock_signals_df
        
        expected_pnl = 100.0
        mock_log_df = pd.DataFrame({'Net PnL (USD)': [50.0, 30.0, 20.0]})
        mock_generate_log.return_value = mock_log_df

        result_pnl = run_backtest_for_optimization(
            self.params_dict, self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
        )
        self.assertEqual(result_pnl, expected_pnl)

        expected_iter_config_signals = self.base_config.copy()
        expected_iter_config_signals.update(self.params_dict)
        
        mock_generate_signals.assert_called_once()
        call_args_signals, _ = mock_generate_signals.call_args
        pd.testing.assert_frame_equal(call_args_signals[0], self.data_with_z_and_prices)
        self.assertEqual(call_args_signals[1], expected_iter_config_signals)
        
        mock_generate_log.assert_called_once_with(
            signals_df=mock_signals_df,
            data_with_prices_hedge_ratio=self.data_with_z_and_prices,
            global_config=expected_iter_config_signals, # generate_trade_log receives the merged iter_config
            user_primary_quantity=self.quantity,
            primary_is_A_selected=self.primary_is_A
        )

    @patch('optimization.generate_trade_log')
    @patch('optimization.generate_trade_signals')
    def test_empty_trade_log(self, mock_generate_signals, mock_generate_log):
        mock_generate_signals.return_value = pd.DataFrame({'zscore': [1,2,3]})
        mock_generate_log.return_value = pd.DataFrame() 
        result_pnl = run_backtest_for_optimization(
            self.params_dict, self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
        )
        self.assertEqual(result_pnl, -np.inf)

    @patch('optimization.generate_trade_log')
    @patch('optimization.generate_trade_signals')
    def test_trade_log_no_pnl_column(self, mock_generate_signals, mock_generate_log):
        mock_generate_signals.return_value = pd.DataFrame({'zscore': [1,2,3]})
        mock_generate_log.return_value = pd.DataFrame({'SomeOtherColumn': [10, 20]}) # Log without PnL column
        
        # Current implementation of run_backtest_for_optimization will raise KeyError here
        with self.assertRaises(KeyError) as context:
            run_backtest_for_optimization(
                self.params_dict, self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
            )
        self.assertTrue('Net PnL (USD)' in str(context.exception))


    @patch('optimization.generate_trade_log')
    @patch('optimization.generate_trade_signals')
    def test_trade_log_pnl_column_all_nan(self, mock_generate_signals, mock_generate_log):
        mock_generate_signals.return_value = pd.DataFrame({'zscore': [1,2,3]})
        mock_log_df = pd.DataFrame({'Net PnL (USD)': [np.nan, np.nan]})
        mock_generate_log.return_value = mock_log_df
        result_pnl = run_backtest_for_optimization(
            self.params_dict, self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
        )
        self.assertEqual(result_pnl, 0.0, "Sum of NaNs PnL should be 0 as per pandas sum behavior")


class TestOptimizeStrategyParameters(unittest.TestCase):
    def setUp(self):
        self.data_with_z_and_prices = pd.DataFrame({
            'zscore': np.random.randn(100),
            'close_a': np.random.rand(100)*100+50,
            'close_b': np.random.rand(100)*50+25,
            'hedge_ratio': np.full(100, 0.5),
            'symbol_a': ['A']*100,
            'symbol_b': ['B']*100,
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
        }).set_index('timestamp')
        self.base_config = {'initial_param': 'test_val', 'slippage':0.001, 'trading_fee':0.001}
        self.quantity = 100.0
        self.primary_is_A = True
        
        # These are now fixed inside optimize_strategy_parameters, tests will use these defaults
        # self.entry_thresholds = np.arange(1.0, 2.5, 0.5) 
        # self.exit_thresholds = np.arange(0.1, 1.0, 0.2)  
        # self.stop_loss_thresholds = np.arange(2.0, 3.5, 0.5)

    @patch('optimization.st') 
    @patch('optimization.run_backtest_for_optimization')
    def test_finds_best_parameters(self, mock_run_backtest, mock_st_ops): # Renamed mock_st
        mock_progress_bar = MagicMock(); mock_st_ops.progress.return_value = mock_progress_bar
        mock_status_text = MagicMock(); mock_st_ops.empty.return_value = mock_status_text

        def pnl_side_effect(params_dict, data, base_cfg, qty, is_A):
            et, xt, slt = params_dict['entry_threshold'], params_dict['exit_threshold'], params_dict['stop_loss_threshold']
            # Using default ranges from optimization.py:
            # entry_thresholds = np.round(np.arange(0.8, 2.1, 0.2), 2) -> [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
            # exit_thresholds = np.round(np.arange(0.1, 0.8, 0.1), 2) -> [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            # stop_loss_thresholds = np.round(np.arange(1.5, 3.6, 0.2), 2) -> [1.5, 1.7, ..., 3.5]
            if et == 1.6 and xt == 0.4 and slt == 2.1: return 200.0 # Example best
            if et == 1.2 and xt == 0.2 and slt == 1.9: return 150.0
            return 50.0 
        mock_run_backtest.side_effect = pnl_side_effect

        best_params, best_pnl = optimize_strategy_parameters(
            self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
        )
        self.assertIsNotNone(best_params)
        self.assertAlmostEqual(best_params['entry_threshold'], 1.6)
        self.assertAlmostEqual(best_params['exit_threshold'], 0.4)
        # Stop loss must be > entry. For entry=1.6, min stop is 1.7 from default list.
        # For exit=0.4, stop_loss must be > 0.4.
        # The function defines stop_loss_thresholds as np.round(np.arange(1.5, 3.6, 0.2), 2)
        # So, if entry is 1.6, valid stops are [1.7, 1.9, 2.1, ...].
        # The side effect for 200.0 has slt=2.1, which is valid for entry=1.6 and exit=0.4.
        self.assertAlmostEqual(best_params['stop_loss_threshold'], 2.1)
        self.assertEqual(best_pnl, 200.0)
        mock_st_ops.progress.assert_called()
        mock_status_text.text.assert_called()

    @patch('optimization.st')
    @patch('optimization.run_backtest_for_optimization')
    def test_no_profitable_trades(self, mock_run_backtest, mock_st_ops):
        mock_run_backtest.return_value = -np.inf 
        mock_progress_bar = MagicMock(); mock_st_ops.progress.return_value = mock_progress_bar
        mock_status_text = MagicMock(); mock_st_ops.empty.return_value = mock_status_text

        best_params, best_pnl = optimize_strategy_parameters(
             self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
        )
        self.assertIsNone(best_params)
        self.assertEqual(best_pnl, -np.inf)
        mock_st_ops.warning.assert_called_with("Optimization did not find any profitable trades with the tested parameter combinations.")

    @patch('optimization.st')
    @patch('optimization.np.arange', side_effect=lambda start, stop, step: np.array([])) # Mock arange to create no params
    @patch('optimization.run_backtest_for_optimization')
    def test_no_valid_parameter_combinations(self, mock_run_backtest, mock_np_arange, mock_st_ops):
        # This test now mocks np.arange to ensure no parameter combinations are generated
        mock_progress_bar = MagicMock(); mock_st_ops.progress.return_value = mock_progress_bar
        mock_status_text = MagicMock(); mock_st_ops.empty.return_value = mock_status_text
        
        best_params, best_pnl = optimize_strategy_parameters(
            self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
        )
        mock_run_backtest.assert_not_called() 
        self.assertIsNone(best_params)
        self.assertEqual(best_pnl, -np.inf)
        # The warning message might change if valid_combinations is 0 before loops start
        # The function calculates valid_combinations first. If this is 0, it warns.
        # If np.arange returns empty, valid_combinations will be 0.
        mock_st_ops.warning.assert_called_with("No valid parameter combinations to test with the defined ranges and constraints. Adjust ranges.")


    @patch('optimization.st')
    @patch('optimization.run_backtest_for_optimization')
    def test_parameter_filtering_verified_by_calls(self, mock_run_backtest, mock_st_ops):
        mock_run_backtest.return_value = 10.0 
        mock_progress_bar = MagicMock(); mock_st_ops.progress.return_value = mock_progress_bar
        mock_status_text = MagicMock(); mock_st_ops.empty.return_value = mock_status_text

        optimize_strategy_parameters(
            self.data_with_z_and_prices, self.base_config, self.quantity, self.primary_is_A
        )
        
        self.assertGreater(mock_run_backtest.call_count, 0, "run_backtest_for_optimization should be called for valid combinations")

        # Check some calls to ensure filters are applied (exit < entry, stop > entry, stop > exit)
        # Default ranges from function:
        # entry_thresholds = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        # exit_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        # stop_loss_thresholds = [1.5, 1.7, ..., 3.5]
        
        for call_args_item in mock_run_backtest.call_args_list:
            params_dict = call_args_item[0][0] # First positional arg of run_backtest, which is params_dict
            entry_t = params_dict['entry_threshold']
            exit_t = params_dict['exit_threshold']
            stop_loss_t = params_dict['stop_loss_threshold']
            
            self.assertLess(exit_t, entry_t, f"Filter fail: exit_t ({exit_t}) >= entry_t ({entry_t})")
            self.assertGreater(stop_loss_t, entry_t, f"Filter fail: stop_loss_t ({stop_loss_t}) <= entry_t ({entry_t})")
            self.assertGreater(stop_loss_t, exit_t, f"Filter fail: stop_loss_t ({stop_loss_t}) <= exit_t ({exit_t})")

if __name__ == '__main__':
    unittest.main()
