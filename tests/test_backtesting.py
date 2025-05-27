import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_almost_equal
from unittest.mock import patch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtesting import calculate_secondary_quantity, generate_trade_signals, generate_trade_log

class TestCalculateSecondaryQuantity(unittest.TestCase):
    # Actual signature from backtesting.py:
    # calculate_secondary_quantity(primary_quantity_units, hedge_ratio, primary_price, secondary_price, primary_is_asset_a)

    def test_primary_is_asset_a(self):
        # 10 units of Asset A, A_price=$100, B_price=$50, HR=0.5
        # Primary Notional (A) = 10 * 100 = $1000
        # Secondary Notional Target (B) = $1000 * 0.5 = $500
        # Secondary Quantity (B units) = $500 / $50 = 10 units
        self.assertAlmostEqual(calculate_secondary_quantity(10, 0.5, 100, 50, True), 10.0)

        # 5 units of Asset A, A_price=$50, B_price=$200, HR=0.8
        # Primary Notional (A) = 5 * 50 = $250
        # Secondary Notional Target (B) = $250 * 0.8 = $200
        # Secondary Quantity (B units) = $200 / $200 = 1 unit
        self.assertAlmostEqual(calculate_secondary_quantity(5, 0.8, 50, 200, True), 1.0)

    def test_primary_is_asset_b(self): # primary_is_asset_a = False
        # 20 units of Asset B, B_price=$50, A_price=$100, HR=0.5
        # Primary Notional (B) = 20 * 50 = $1000
        # Secondary Notional Target (A) = $1000 / 0.5 = $2000
        # Secondary Quantity (A units) = $2000 / $100 = 20 units
        self.assertAlmostEqual(calculate_secondary_quantity(20, 0.5, 50, 100, False), 20.0)

        # 5 units of Asset B, B_price=$200, A_price=$50, HR=0.8
        # Primary Notional (B) = 5 * 200 = $1000
        # Secondary Notional Target (A) = $1000 / 0.8 = $1250
        # Secondary Quantity (A units) = $1250 / $50 = 25 units
        self.assertAlmostEqual(calculate_secondary_quantity(5, 0.8, 200, 50, False), 25.0)
        
    def test_zero_prices_or_hedge_ratio(self):
        self.assertEqual(calculate_secondary_quantity(10, 0.5, 0, 50, True), 0, "Primary price 0")
        self.assertEqual(calculate_secondary_quantity(10, 0.5, 100, 0, True), 0, "Secondary price 0")
        self.assertEqual(calculate_secondary_quantity(10, 0.5, 0, 100, False), 0, "Primary price 0 (B)")
        self.assertEqual(calculate_secondary_quantity(10, 0.5, 50, 0, False), 0, "Secondary price 0 (A)")
        self.assertEqual(calculate_secondary_quantity(10, 0, 100, 50, True), 0, "HR 0, Primary A")
        self.assertEqual(calculate_secondary_quantity(10, 0, 50, 100, False), 0, "HR 0, Primary B (special case in func)")

    def test_output_is_always_positive(self):
        self.assertAlmostEqual(calculate_secondary_quantity(10, -0.5, 100, 50, True), 10.0) 
        self.assertAlmostEqual(calculate_secondary_quantity(20, -0.5, 50, 100, False), 20.0)
        self.assertAlmostEqual(calculate_secondary_quantity(10, 0.5, 100, -50, True), 10.0)


class TestGenerateTradeSignals(unittest.TestCase):
    def setUp(self):
        self.config_params = {'entry_threshold': 2.0, 'exit_threshold': 0.5, 'stop_loss_threshold': 3.0}
        self.zscore_data_template = pd.DataFrame({
            'zscore': np.full(20, 0.0), 
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=20, freq='D'))
        }).set_index('timestamp')

    def test_no_signals(self):
        data = self.zscore_data_template.copy()
        data.loc[:,'zscore'] = [0.1,0.2,0.3,0.1,-0.1,-0.2,-0.3,0.0] * (len(data)//8) + [0.0]*(len(data)%8)
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertTrue((signals_df['signal'] == 0).all())
        self.assertTrue((signals_df['position'] == 0).all())

    def test_entry_buy_spread(self): 
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = -1.8 
        data.loc[data.index[1], 'zscore'] = -2.1 # Entry signal here
        data.loc[data.index[2], 'zscore'] = -1.5 # Should carry position (z > prev_z, but not > -exit_thresh)
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['signal'].iloc[1], 1.0)
        self.assertEqual(signals_df['position'].iloc[1], 1)
        self.assertEqual(signals_df['buy_signal_z'].iloc[1], -2.1)
        self.assertEqual(signals_df['position'].iloc[2], 1, "Position should carry to index 2")

    def test_entry_sell_spread(self): 
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = 1.8 
        data.loc[data.index[1], 'zscore'] = 2.1  # Entry signal here
        data.loc[data.index[2], 'zscore'] = 1.5  # Should carry position (z < prev_z, but not < exit_thresh)
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['signal'].iloc[1], -1.0)
        self.assertEqual(signals_df['position'].iloc[1], -1)
        self.assertEqual(signals_df['sell_signal_z'].iloc[1], 2.1)
        self.assertEqual(signals_df['position'].iloc[2], -1, "Position should carry to index 2")

    def test_exit_from_buy_spread(self): 
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = -1.8; data.loc[data.index[1], 'zscore'] = -2.1 
        data.loc[data.index[2], 'zscore'] = -0.6 
        data.loc[data.index[3], 'zscore'] = -0.4 
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['position'].iloc[1], 1) 
        self.assertEqual(signals_df['position'].iloc[2], 1) 
        self.assertEqual(signals_df['signal'].iloc[3], 2.0) 
        self.assertEqual(signals_df['position'].iloc[3], 0)
        self.assertEqual(signals_df['exit_signal_z'].iloc[3], -0.4)

    def test_exit_from_sell_spread(self):
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = 1.8; data.loc[data.index[1], 'zscore'] = 2.1
        data.loc[data.index[2], 'zscore'] = 0.6 
        data.loc[data.index[3], 'zscore'] = 0.4 
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['position'].iloc[1], -1)
        self.assertEqual(signals_df['position'].iloc[2], -1)
        self.assertEqual(signals_df['signal'].iloc[3], 2.0)
        self.assertEqual(signals_df['position'].iloc[3], 0)
        self.assertEqual(signals_df['exit_signal_z'].iloc[3], 0.4)

    def test_stop_loss_from_buy_spread(self):
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = -1.8; data.loc[data.index[1], 'zscore'] = -2.1 
        data.loc[data.index[2], 'zscore'] = -2.9 
        data.loc[data.index[3], 'zscore'] = -3.1 
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['position'].iloc[1], 1)
        self.assertEqual(signals_df['position'].iloc[2], 1) 
        self.assertEqual(signals_df['signal'].iloc[3], 2.0) 
        self.assertEqual(signals_df['position'].iloc[3], 0)
        self.assertEqual(signals_df['exit_signal_z'].iloc[3], -3.1)

    def test_stop_loss_from_sell_spread(self): 
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = 1.8; data.loc[data.index[1], 'zscore'] = 2.1
        data.loc[data.index[2], 'zscore'] = 2.9 
        data.loc[data.index[3], 'zscore'] = 3.1 
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['position'].iloc[1], -1)
        self.assertEqual(signals_df['position'].iloc[2], -1)
        self.assertEqual(signals_df['signal'].iloc[3], 2.0)
        self.assertEqual(signals_df['position'].iloc[3], 0)
        self.assertEqual(signals_df['exit_signal_z'].iloc[3], 3.1)

    def test_nan_zscore_handling(self): 
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = -1.8; data.loc[data.index[1], 'zscore'] = -2.2 
        data.loc[data.index[2], 'zscore'] = np.nan 
        data.loc[data.index[3], 'zscore'] = -1.5   
        data.loc[data.index[4], 'zscore'] = -0.4   
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['position'].iloc[1], 1)
        self.assertEqual(signals_df['signal'].iloc[2], 0)   
        self.assertEqual(signals_df['position'].iloc[2], 1) 
        self.assertEqual(signals_df['signal'].iloc[3], 0)   
        self.assertEqual(signals_df['position'].iloc[3], 1)
        self.assertEqual(signals_df['signal'].iloc[4], 2.0)
        self.assertEqual(signals_df['position'].iloc[4], 0)

    def test_prev_z_nan_handling_at_start(self): 
        data = self.zscore_data_template.copy()
        data.loc[data.index[0], 'zscore'] = np.nan 
        data.loc[data.index[1], 'zscore'] = -1.8   
        data.loc[data.index[2], 'zscore'] = -2.2   
        signals_df = generate_trade_signals(data, self.config_params)
        self.assertEqual(signals_df['position'].iloc[0], 0); self.assertEqual(signals_df['signal'].iloc[0], 0)
        self.assertEqual(signals_df['position'].iloc[1], 0); self.assertEqual(signals_df['signal'].iloc[1], 0)
        self.assertEqual(signals_df['signal'].iloc[2], 1.0); self.assertEqual(signals_df['position'].iloc[2], 1)

class TestGenerateTradeLog(unittest.TestCase):
    def setUp(self):
        self.global_config = {'slippage': 0.001, 'trading_fee': 0.001}
        self.user_primary_qty_units = 10.0 
        self.primary_is_A = True

        idx = pd.to_datetime(pd.date_range(start='2023-01-01', periods=10, freq='D'))
        self.data_prices_hr = pd.DataFrame({
            'close_a': [100,101,102,103,104,100,98,99,100,102.0],
            'close_b': [50,51,52,53,54,50,49,48,50,51.0],
            'hedge_ratio': [0.5]*10,
            'zscore': [1.0,1.5,2.1,0.8,0.4,-1.0,-1.5,-2.2,-1.0,0.0],
            'symbol_a': ['SYMA']*10, 'symbol_b': ['SYMB']*10,
        }, index=idx)
        
        self.signals_template_df = pd.DataFrame({ 
            'zscore': self.data_prices_hr['zscore'].copy(), # Ensure zscore column is present
            'signal': np.zeros(10), 'position': np.zeros(10),
            'buy_signal_z': np.nan, 'sell_signal_z': np.nan, 'exit_signal_z': np.nan,
        }, index=idx)

    def test_no_trades(self):
        log = generate_trade_log(self.signals_template_df, self.data_prices_hr, self.global_config, self.user_primary_qty_units, self.primary_is_A)
        self.assertTrue(log.empty)

    def test_missing_columns(self):
        signals = self.signals_template_df.copy(); signals.loc[signals.index[1],'signal']=1.0
        for col in ['close_a', 'close_b', 'hedge_ratio', 'symbol_a', 'symbol_b']:
            data_missing = self.data_prices_hr.drop(columns=[col])
            log = generate_trade_log(signals, data_missing, self.global_config, self.user_primary_qty_units, self.primary_is_A)
            self.assertTrue(log.empty, f"Empty log expected if {col} is missing")

    @patch('backtesting.calculate_secondary_quantity')
    def test_buy_spread_cycle(self, mock_calc_sec_qty):
        entry_idx, exit_idx = 1, 3
        pa_entry = self.data_prices_hr['close_a'].iloc[entry_idx]
        pb_entry = self.data_prices_hr['close_b'].iloc[entry_idx]
        hr_entry = self.data_prices_hr['hedge_ratio'].iloc[entry_idx]
        # Expected secondary qty based on actual function: (10 * 101 * 0.5) / 51 = 9.90196...
        expected_qty_b = calculate_secondary_quantity(self.user_primary_qty_units, hr_entry, pa_entry, pb_entry, True)
        mock_calc_sec_qty.return_value = expected_qty_b

        signals = self.signals_template_df.copy()
        signals.loc[signals.index[entry_idx], 'signal'] = 1.0; signals.loc[signals.index[entry_idx], 'buy_signal_z'] = -2.05
        signals.loc[signals.index[exit_idx], 'signal'] = 2.0; signals.loc[signals.index[exit_idx], 'exit_signal_z'] = -0.05
        
        log = generate_trade_log(signals, self.data_prices_hr, self.global_config, self.user_primary_qty_units, True)
        self.assertEqual(len(log), 1); trade = log.iloc[0]
        self.assertEqual(trade['Trade Type'], 'Buy Spread (L SYMA, S SYMB)')
        mock_calc_sec_qty.assert_called_once_with(self.user_primary_qty_units, hr_entry, pa_entry, pb_entry, True)
        self.assertAlmostEqual(trade['Qty A'], self.user_primary_qty_units) # Corrected key
        self.assertAlmostEqual(trade['Qty B'], mock_calc_sec_qty.return_value) # Corrected key
        self.assertIsNotNone(trade['Net PnL (USD)'])

    @patch('backtesting.calculate_secondary_quantity')
    def test_sell_spread_trade_cycle(self, mock_calc_sec_qty):
        entry_idx, exit_idx = 1, 3
        pa_entry = self.data_prices_hr['close_a'].iloc[entry_idx]
        pb_entry = self.data_prices_hr['close_b'].iloc[entry_idx]
        hr_entry = self.data_prices_hr['hedge_ratio'].iloc[entry_idx]
        expected_qty_b = calculate_secondary_quantity(self.user_primary_qty_units, hr_entry, pa_entry, pb_entry, True)
        mock_calc_sec_qty.return_value = expected_qty_b

        signals = self.signals_template_df.copy()
        signals.loc[signals.index[entry_idx], 'signal'] = -1.0 
        signals.loc[signals.index[entry_idx], 'sell_signal_z'] = 2.1
        signals.loc[signals.index[exit_idx], 'signal'] = 2.0 
        signals.loc[signals.index[exit_idx], 'exit_signal_z'] = 0.4
        
        log = generate_trade_log(signals, self.data_prices_hr, self.global_config, self.user_primary_qty_units, True)
        self.assertEqual(len(log), 1); trade = log.iloc[0]
        self.assertEqual(trade['Trade Type'], 'Sell Spread (S SYMA, L SYMB)')
        self.assertAlmostEqual(trade['Qty A'], self.user_primary_qty_units)
        self.assertAlmostEqual(trade['Qty B'], expected_qty_b)

    @patch('backtesting.calculate_secondary_quantity')
    def test_primary_is_A_false_buy_spread(self, mock_calc_sec_qty):
        entry_idx, exit_idx = 1, 3
        user_qty_b_units = 20.0 # This is now units of B
        pa_entry = self.data_prices_hr['close_a'].iloc[entry_idx] # Secondary price
        pb_entry = self.data_prices_hr['close_b'].iloc[entry_idx] # Primary price
        hr_entry = self.data_prices_hr['hedge_ratio'].iloc[entry_idx]
        # Expected secondary qty (A) = (20 * 51 / 0.5) / 101 = 20.198...
        expected_qty_a = calculate_secondary_quantity(user_qty_b_units, hr_entry, pb_entry, pa_entry, False)
        mock_calc_sec_qty.return_value = expected_qty_a

        signals = self.signals_template_df.copy()
        signals.loc[signals.index[entry_idx], 'signal'] = 1.0 
        signals.loc[signals.index[exit_idx], 'signal'] = 2.0 
        
        log = generate_trade_log(signals, self.data_prices_hr, self.global_config, user_qty_b_units, False) # primary_is_A = False
        self.assertEqual(len(log), 1); trade = log.iloc[0]
        self.assertAlmostEqual(trade['Qty B'], user_qty_b_units)
        self.assertAlmostEqual(trade['Qty A'], expected_qty_a)
        self.assertIsNotNone(trade['Net PnL (USD)'])
        # Check PnL % basis
        primary_notional_at_entry = abs(user_qty_b_units * pb_entry)
        self.assertAlmostEqual(trade['PnL % (Primary Notional)'], (trade['Net PnL (USD)'] / primary_notional_at_entry) * 100)


    @patch('backtesting.calculate_secondary_quantity')
    def test_eod_closure(self, mock_calc_sec_qty):
        mock_calc_sec_qty.return_value = 9.9 
        signals = self.signals_template_df.iloc[:5].copy()
        data = self.data_prices_hr.iloc[:5].copy()
        signals.loc[signals.index[1], 'signal'] = 1.0 
        
        log = generate_trade_log(signals, data, self.global_config, self.user_primary_qty_units, True)
        self.assertEqual(len(log), 1); trade = log.iloc[0]
        self.assertTrue(" (Closed at EOD)" in trade['Trade Type'])
        self.assertEqual(trade['Exit Timestamp'], data.index[-1])
        self.assertEqual(trade['Exit Z-score'], data['zscore'].iloc[-1])

    def test_skip_trade_conditions(self):
        signals = self.signals_template_df.copy(); signals.loc[signals.index[1], 'signal'] = 1.0
        
        data_nan_hr = self.data_prices_hr.copy(); data_nan_hr.loc[data_nan_hr.index[1], 'hedge_ratio'] = np.nan
        self.assertTrue(generate_trade_log(signals, data_nan_hr, self.global_config, self.user_primary_qty_units, True).empty, "NaN HR")
        
        data_zero_pa = self.data_prices_hr.copy(); data_zero_pa.loc[data_zero_pa.index[1], 'close_a'] = 0
        self.assertTrue(generate_trade_log(signals, data_zero_pa, self.global_config, self.user_primary_qty_units, True).empty, "Zero Price A")
        
        data_zero_pb = self.data_prices_hr.copy(); data_zero_pb.loc[data_zero_pb.index[1], 'close_b'] = 0
        self.assertTrue(generate_trade_log(signals, data_zero_pb, self.global_config, self.user_primary_qty_units, True).empty, "Zero Price B")

        self.assertTrue(generate_trade_log(signals, self.data_prices_hr, self.global_config, 0, True).empty, "Zero Primary Qty Units")

        with patch('backtesting.calculate_secondary_quantity', return_value=0) as m:
            self.assertTrue(generate_trade_log(signals, self.data_prices_hr, self.global_config, self.user_primary_qty_units, True).empty, "Calc Sec Qty returns 0")
            m.assert_called()

    def test_zero_slippage_fees(self):
        config_no_costs = {'slippage': 0.0, 'trading_fee': 0.0}
        signals = self.signals_template_df.copy()
        entry_idx, exit_idx = 1, 3
        signals.loc[signals.index[entry_idx], 'signal'] = 1.0
        signals.loc[signals.index[exit_idx], 'signal'] = 2.0
        
        pa_entry = self.data_prices_hr['close_a'].iloc[entry_idx]
        pb_entry = self.data_prices_hr['close_b'].iloc[entry_idx]
        hr_entry = self.data_prices_hr['hedge_ratio'].iloc[entry_idx]
        pa_exit = self.data_prices_hr['close_a'].iloc[exit_idx]
        pb_exit = self.data_prices_hr['close_b'].iloc[exit_idx]

        qty_a_exp = self.user_primary_qty_units
        qty_b_exp = calculate_secondary_quantity(qty_a_exp, hr_entry, pa_entry, pb_entry, True)

        log = generate_trade_log(signals, self.data_prices_hr, config_no_costs, self.user_primary_qty_units, True)
        self.assertEqual(len(log), 1); trade = log.iloc[0]

        pnl_a_exp = qty_a_exp * (pa_exit - pa_entry)
        pnl_b_exp = qty_b_exp * (pb_entry - pb_exit) 
        
        self.assertAlmostEqual(trade['PnL Asset A (USD)'], pnl_a_exp)
        self.assertAlmostEqual(trade['PnL Asset B (USD)'], pnl_b_exp)
        self.assertAlmostEqual(trade['Total Fees (USD)'], 0.0)
        self.assertAlmostEqual(trade['Net PnL (USD)'], pnl_a_exp + pnl_b_exp)

if __name__ == '__main__':
    unittest.main()
