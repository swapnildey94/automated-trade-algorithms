import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call

# Add parent directory to sys.path to allow direct import of scenario_analysis and its dependencies
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import the function to be tested
try:
    from scenario_analysis import run_scenario_analysis
except ImportError as e:
    print(f"Initial ImportError (expected if dependencies not yet installed): {e}")
    run_scenario_analysis = None

# We will mock functions from regression and backtesting, so no need to import them directly for the test execution
# but good to note them for patching targets:
# from regression import calculate_hedge_ratio, calculate_zscore
# from backtesting import generate_trade_signals, generate_trade_log


class TestRunScenarioAnalysis(unittest.TestCase):
    def setUp(self):
        # Sample data for Asset A
        self.asset_a_data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'close': [100, 101, 102, 103],
            'log_price': np.log([100, 101, 102, 103])
        }).set_index('timestamp')
        
        # Sample data for Asset B
        self.asset_b_data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
            'close': [50, 51, 52, 53],
            'log_price': np.log([50, 51, 52, 53])
        }).set_index('timestamp')

        self.asset_a_symbol = "AssetA"
        self.asset_b_symbol = "AssetB"

        self.config = {
            'lookback_period': 20, # Days for hedge_ratio/zscore lookback
            'timeframe': '1d',     # Timeframe for data
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss_threshold': 3.0,
            'slippage': 0.001,      # For generate_trade_log (global_config part)
            'trading_fee': 0.001 # For generate_trade_log (global_config part)
        }
        self.quantity = 10.0 # Units of primary asset
        self.primary_is_A = True
        
        # Expected merged_df structure (before renaming in the function)
        self.expected_merged_initial = pd.merge(
            self.asset_a_data, self.asset_b_data, 
            on='timestamp', suffixes=('_a', '_b'), how='inner'
        )


    @patch('scenario_analysis.generate_trade_log')
    @patch('scenario_analysis.generate_trade_signals')
    @patch('scenario_analysis.calculate_zscore')
    @patch('scenario_analysis.calculate_hedge_ratio')
    @patch('scenario_analysis.pd.merge')
    def test_successful_run(self, mock_pd_merge, mock_calc_hr, mock_calc_z, mock_gen_signals, mock_gen_log):
        # --- Setup Mocks ---
        # 1. pd.merge
        # The actual merged_df created inside the function will have columns like a_close, a_log_price
        # before renaming. The mock should return this.
        # However, the function renames columns 'a_close' to 'close_a', etc.
        # The dataframe passed to calculate_hedge_ratio will have these renamed columns.
        
        # Let's define what merged_df looks like AFTER renaming and adding symbol columns,
        # as this is the structure passed to calculate_hedge_ratio.
        expected_df_for_hr_input = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'close_a': [100, 101], 'log_price_a': np.log([100, 101]),
            'close_b': [50, 51], 'log_price_b': np.log([50, 51]),
            'symbol_a': [self.asset_a_symbol]*2, 'symbol_b': [self.asset_b_symbol]*2
        }).set_index('timestamp')
        
        # pd.merge is called with asset_a_data[['close', 'log_price']].add_prefix('a_')
        # and asset_b_data...
        # So, the mock_pd_merge should return a df with columns like 'a_close', 'a_log_price', 'b_close', 'b_log_price'
        mock_merged_df_initial_index = pd.to_datetime(['2023-01-01', '2023-01-02'])
        mock_merged_df_initial_index.name = 'timestamp' # Ensure index name matches
        mock_merged_df_initial = pd.DataFrame({
            'a_close': [100, 101], 'a_log_price': np.log([100, 101]),
            'b_close': [50, 51], 'b_log_price': np.log([50, 51])
        }, index=mock_merged_df_initial_index)
        mock_pd_merge.return_value = mock_merged_df_initial

        # 2. calculate_hedge_ratio
        mock_df_with_hr = expected_df_for_hr_input.copy() # Start with the structure it receives
        mock_df_with_hr['hedge_ratio'] = 0.5
        mock_df_with_hr['spread'] = mock_df_with_hr['log_price_a'] - mock_df_with_hr['hedge_ratio'] * mock_df_with_hr['log_price_b']
        mock_calc_hr.return_value = mock_df_with_hr

        # 3. calculate_zscore
        mock_df_with_z = mock_df_with_hr.copy()
        mock_df_with_z['zscore'] = [-2.0, 2.0]
        mock_calc_z.return_value = mock_df_with_z
        
        # 4. generate_trade_signals
        mock_signals = pd.DataFrame({'signal': [1.0, -1.0]}, index=mock_df_with_z.index)
        mock_gen_signals.return_value = mock_signals

        # 5. generate_trade_log
        mock_log = pd.DataFrame({'Net PnL (USD)': [100.0]})
        mock_gen_log.return_value = mock_log

        # --- Call the function ---
        results = run_scenario_analysis(
            self.asset_a_data, self.asset_b_data, 
            self.asset_a_symbol, self.asset_b_symbol,
            self.config, self.quantity, self.primary_is_A
        )

        # --- Assertions ---
        # 1. pd.merge call
        mock_pd_merge.assert_called_once()
        # Check arguments of pd.merge (simplified check for suffixes and how='inner')
        args, kwargs = mock_pd_merge.call_args
        pd.testing.assert_frame_equal(args[0], self.asset_a_data[['close', 'log_price']].add_prefix('a_'))
        pd.testing.assert_frame_equal(args[1], self.asset_b_data[['close', 'log_price']].add_prefix('b_'))
        self.assertEqual(kwargs['left_index'], True)
        self.assertEqual(kwargs['right_index'], True)
        self.assertEqual(kwargs['how'], 'inner')

        # 2. calculate_hedge_ratio call
        mock_calc_hr.assert_called_once()
        args_hr, _ = mock_calc_hr.call_args
        # The first argument to calc_hr is the dataframe *after* merge, rename, and symbol columns addition
        pd.testing.assert_frame_equal(args_hr[0], expected_df_for_hr_input)
        self.assertEqual(args_hr[1], self.config['lookback_period'])
        self.assertEqual(args_hr[2], self.config['timeframe'])

        # 3. calculate_zscore call
        mock_calc_z.assert_called_once_with(mock_df_with_hr, self.config['lookback_period'], self.config['timeframe'])
        
        # 4. generate_trade_signals call
        mock_gen_signals.assert_called_once_with(mock_df_with_z, self.config)

        # 5. generate_trade_log call
        mock_gen_log.assert_called_once_with(mock_signals, mock_df_with_z, self.config, self.quantity, self.primary_is_A)

        # 6. Returned dictionary content
        self.assertIn('merged_df', results)
        # results['merged_df'] is the one after renaming and adding symbol columns
        pd.testing.assert_frame_equal(results['merged_df'], expected_df_for_hr_input)
        pd.testing.assert_frame_equal(results['data_with_hedge'], mock_df_with_hr)
        pd.testing.assert_frame_equal(results['data_with_z'], mock_df_with_z)
        pd.testing.assert_frame_equal(results['trade_signals'], mock_signals)
        pd.testing.assert_frame_equal(results['detailed_trade_log'], mock_log)
        self.assertNotIn('error', results)


    @patch('scenario_analysis.generate_trade_log')
    @patch('scenario_analysis.generate_trade_signals')
    @patch('scenario_analysis.calculate_zscore')
    @patch('scenario_analysis.calculate_hedge_ratio')
    @patch('scenario_analysis.pd.merge')
    def test_error_no_overlapping_data(self, mock_pd_merge, mock_calc_hr, mock_calc_z, mock_gen_signals, mock_gen_log):
        mock_pd_merge.return_value = pd.DataFrame() # Simulate empty merge

        results = run_scenario_analysis(
            self.asset_a_data, self.asset_b_data, self.asset_a_symbol, self.asset_b_symbol,
            self.config, self.quantity, self.primary_is_A
        )
        
        self.assertEqual(results, {'error': 'No overlapping data found for the selected assets and timeframe.'})
        mock_calc_hr.assert_not_called()
        mock_calc_z.assert_not_called()
        mock_gen_signals.assert_not_called()
        mock_gen_log.assert_not_called()

    @patch('scenario_analysis.generate_trade_log')
    @patch('scenario_analysis.generate_trade_signals')
    @patch('scenario_analysis.calculate_zscore')
    @patch('scenario_analysis.calculate_hedge_ratio')
    @patch('scenario_analysis.pd.merge')
    def test_error_zscore_calc_failure_empty_df(self, mock_pd_merge, mock_calc_hr, mock_calc_z, mock_gen_signals, mock_gen_log):
        mock_merged_df_initial = pd.DataFrame({'a_close': [100], 'b_close': [50]}, index=[pd.to_datetime('2023-01-01')])
        mock_pd_merge.return_value = mock_merged_df_initial
        
        # data_with_hedge is the renamed and symbol-added version of merged_df
        # For this test, its content beyond being non-empty for calc_hr doesn't deeply matter
        mock_df_with_hr = pd.DataFrame({'log_price_a': [1], 'log_price_b': [1], 'symbol_a':['A'], 'symbol_b':['B']}, index=[pd.to_datetime('2023-01-01')])
        mock_calc_hr.return_value = mock_df_with_hr
        
        mock_calc_z.return_value = pd.DataFrame() # Z-score calc returns empty df

        results = run_scenario_analysis(
            self.asset_a_data, self.asset_b_data, self.asset_a_symbol, self.asset_b_symbol,
            self.config, self.quantity, self.primary_is_A
        )
        self.assertEqual(results, {'error': 'Could not calculate Z-scores. Check data or lookback period.'})
        mock_gen_signals.assert_not_called()
        mock_gen_log.assert_not_called()

    @patch('scenario_analysis.generate_trade_log')
    @patch('scenario_analysis.generate_trade_signals')
    @patch('scenario_analysis.calculate_zscore')
    @patch('scenario_analysis.calculate_hedge_ratio')
    @patch('scenario_analysis.pd.merge')
    def test_error_zscore_calc_failure_all_nan_zscore(self, mock_pd_merge, mock_calc_hr, mock_calc_z, mock_gen_signals, mock_gen_log):
        mock_merged_df_initial = pd.DataFrame({'a_close': [100], 'b_close': [50]}, index=[pd.to_datetime('2023-01-01')])
        mock_pd_merge.return_value = mock_merged_df_initial
        
        mock_df_with_hr = pd.DataFrame({'log_price_a': [1], 'log_price_b': [1], 'symbol_a':['A'], 'symbol_b':['B']}, index=[pd.to_datetime('2023-01-01')])
        mock_calc_hr.return_value = mock_df_with_hr
        
        df_all_nan_zscore = mock_df_with_hr.copy()
        df_all_nan_zscore['zscore'] = np.nan
        mock_calc_z.return_value = df_all_nan_zscore # Z-score column is all NaN

        results = run_scenario_analysis(
            self.asset_a_data, self.asset_b_data, self.asset_a_symbol, self.asset_b_symbol,
            self.config, self.quantity, self.primary_is_A
        )
        self.assertEqual(results, {'error': 'Could not calculate Z-scores. Check data or lookback period.'})
        mock_gen_signals.assert_not_called()
        mock_gen_log.assert_not_called()

if __name__ == '__main__':
    if run_scenario_analysis is None:
        print("scenario_analysis.py function not imported. Cannot run tests directly.")
    else:
        # Tests are designed to be run with pytest
        print("Tests are designed to be run with pytest.")
        pass
