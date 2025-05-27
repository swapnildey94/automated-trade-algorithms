import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_almost_equal

# Add parent directory to sys.path to allow direct import of regression
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regression import calculate_hedge_ratio, calculate_zscore

class TestRegressionFunctions(unittest.TestCase):

    def setUp(self):
        self.data_size = 100
        # Using linspace for more predictable data, adding some noise for realism
        log_price_a_base = np.log(np.linspace(50, 150, self.data_size))
        self.data = pd.DataFrame({
            'log_price_a': log_price_a_base + np.random.randn(self.data_size) * 0.01,
            'log_price_b': log_price_a_base + np.random.randn(self.data_size) * 0.05 + np.log(5), # Correlated but different scale/offset
            'timestamp': pd.to_datetime(np.arange(self.data_size), unit='D', origin='2023-01-01')
        })
        self.data = self.data.set_index('timestamp')

    # --- Tests for calculate_hedge_ratio ---

    def test_calculate_hedge_ratio_valid_input(self):
        lookback_days = 20
        df_hr = calculate_hedge_ratio(self.data.copy(), lookback_period_days=lookback_days, timeframe_selection='1d')
        
        self.assertIn('hedge_ratio', df_hr.columns)
        self.assertIn('spread', df_hr.columns)
        self.assertIn('ols_intercept', df_hr.columns)
        self.assertEqual(df_hr.shape[0], self.data_size)

        lookback_points = lookback_days # For '1d'
        # Hedge ratio is calculated from index `lookback_points` onwards.
        # Indices 0 to `lookback_points - 1` should be NaN.
        self.assertTrue(df_hr['hedge_ratio'].iloc[0:lookback_points].isnull().all())
        if len(df_hr) > lookback_points: # Ensure data is long enough
            self.assertFalse(df_hr['hedge_ratio'].iloc[lookback_points:].isnull().any(), "Hedge ratios should be calculated after initial lookback period.")
            self.assertFalse(df_hr['spread'].iloc[lookback_points:].isnull().any(), "Spreads should be calculated after initial lookback period.")
            self.assertFalse(df_hr['ols_intercept'].iloc[lookback_points:].isnull().any(), "Intercepts should be calculated after initial lookback period.")
            # Plausibility check for hedge_ratio (can be broad)
            self.assertTrue(all(-10 < hr < 10 for hr in df_hr['hedge_ratio'].dropna()), "Hedge ratio seems out of plausible range.")

    def test_calculate_hedge_ratio_lookback_points_calculation(self):
        # Test '1d'
        days_1d = 5
        points_1d = days_1d * 1 
        df_1d = calculate_hedge_ratio(self.data.iloc[:points_1d+5].copy(), lookback_period_days=days_1d, timeframe_selection='1d')
        self.assertTrue(df_1d['hedge_ratio'].iloc[0:points_1d].isnull().all())
        if len(df_1d) > points_1d: self.assertFalse(pd.isna(df_1d['hedge_ratio'].iloc[points_1d]))

        # Test '1h'
        days_1h = 2
        points_1h = days_1h * 24
        index_1h = pd.date_range(start='2023-01-01', periods=points_1h + 5, freq='h')
        data_1h = pd.DataFrame({'log_price_a': np.random.rand(len(index_1h)), 'log_price_b': np.random.rand(len(index_1h))}, index=index_1h)
        df_1h = calculate_hedge_ratio(data_1h, lookback_period_days=days_1h, timeframe_selection='1h')
        self.assertTrue(df_1h['hedge_ratio'].iloc[0:points_1h].isnull().all())
        if len(df_1h) > points_1h: self.assertFalse(pd.isna(df_1h['hedge_ratio'].iloc[points_1h]))

        # Test '15m' with lookback_points adjusted to min 2
        days_15m_short = 1 # results in 1 * 24 * 4 = 96 points
        points_15m_short = days_15m_short * 24 * 4
        index_15m_short = pd.date_range(start='2023-01-01', periods=points_15m_short + 5, freq='15min')
        data_15m_short = pd.DataFrame({'log_price_a': np.random.rand(len(index_15m_short)), 'log_price_b': np.random.rand(len(index_15m_short))}, index=index_15m_short)
        df_15m_short = calculate_hedge_ratio(data_15m_short, lookback_period_days=days_15m_short, timeframe_selection='15m')
        self.assertTrue(df_15m_short['hedge_ratio'].iloc[0:points_15m_short].isnull().all())
        if len(df_15m_short) > points_15m_short: self.assertFalse(pd.isna(df_15m_short['hedge_ratio'].iloc[points_15m_short]))
        
        # Test function's internal minimum lookback_points of 2
        # lookback_period_days = 1, timeframe='1d' -> lookback_points = 1, adjusted to 2.
        df_min_lb = calculate_hedge_ratio(self.data.iloc[:5].copy(), lookback_period_days=1, timeframe_selection='1d')
        internal_min_lb = 2
        self.assertTrue(df_min_lb['hedge_ratio'].iloc[0:internal_min_lb].isnull().all())
        if len(df_min_lb) > internal_min_lb: self.assertFalse(pd.isna(df_min_lb['hedge_ratio'].iloc[internal_min_lb]))


    def test_calculate_hedge_ratio_insufficient_data(self):
        # Data length 5, lookback_points 10. Loop range(10, 5) is empty.
        short_data = self.data.iloc[:5].copy()
        df_hr = calculate_hedge_ratio(short_data, lookback_period_days=10, timeframe_selection='1d')
        self.assertTrue(df_hr['hedge_ratio'].isnull().all())
        self.assertTrue(df_hr['spread'].isnull().all())
        self.assertTrue(df_hr['ols_intercept'].isnull().all())

    def test_calculate_hedge_ratio_with_nans_in_input(self):
        data_with_nans = self.data.copy()
        nan_indices = data_with_nans.index[5:10] # NaNs at original indices 5,6,7,8,9
        data_with_nans.loc[nan_indices, 'log_price_a'] = np.nan
        
        lookback_days = 5
        df_hr = calculate_hedge_ratio(data_with_nans.copy(), lookback_period_days=lookback_days, timeframe_selection='1d')
        lookback_points = lookback_days # for '1d'

        # First calc at index `lookback_points` (iloc[5])
        # Window for iloc[5] (data indices 0-4): No NaNs. Valid.
        if len(df_hr) > lookback_points: self.assertFalse(pd.isna(df_hr['hedge_ratio'].iloc[lookback_points]))
        
        # Window for iloc[6] (data indices 1-5): data.iloc[5] is NaN. Result NaN.
        if len(df_hr) > lookback_points + 1: self.assertTrue(pd.isna(df_hr['hedge_ratio'].iloc[lookback_points + 1]))
        
        # Window for iloc[9] (data indices 4-8): includes NaN from original index 5. Result NaN.
        if len(df_hr) > 9: self.assertTrue(pd.isna(df_hr['hedge_ratio'].iloc[9]))
        # Window for iloc[10] (data indices 5-9): includes original NaNs 5-9. Result NaN.
        if len(df_hr) > 10: self.assertTrue(pd.isna(df_hr['hedge_ratio'].iloc[10]))
        # Window for iloc[14] (data indices 9-13): includes original NaN at index 9. Result NaN.
        if len(df_hr) > 14: self.assertTrue(pd.isna(df_hr['hedge_ratio'].iloc[14]))
        # Window for iloc[15] (data indices 10-14): Original NaNs (5-9) are out. Valid.
        if len(df_hr) > 15: self.assertFalse(pd.isna(df_hr['hedge_ratio'].iloc[15]))

    def test_calculate_hedge_ratio_series_too_short_for_ols(self):
        # lookback_points becomes 2 internally for these cases
        # Data length 1. Loop range(2,1) -> empty. All NaN.
        df_hr_1 = calculate_hedge_ratio(self.data.iloc[:1].copy(), lookback_period_days=1, timeframe_selection='1d')
        self.assertTrue(df_hr_1['hedge_ratio'].isnull().all())

        # Data length 2. Loop range(2,2) -> empty. All NaN.
        df_hr_2 = calculate_hedge_ratio(self.data.iloc[:2].copy(), lookback_period_days=1, timeframe_selection='1d')
        self.assertTrue(df_hr_2['hedge_ratio'].isnull().all())
        
        # Data length 3. Loop range(2,3) -> i=2. Window data[0:2]. Result at index 2.
        df_hr_3 = calculate_hedge_ratio(self.data.iloc[:3].copy(), lookback_period_days=1, timeframe_selection='1d')
        self.assertTrue(df_hr_3['hedge_ratio'].iloc[0:2].isnull().all())
        if len(df_hr_3) > 2: self.assertFalse(pd.isna(df_hr_3['hedge_ratio'].iloc[2]))
        
        # Data length 0
        df_hr_empty = calculate_hedge_ratio(pd.DataFrame(columns=['log_price_a', 'log_price_b']), lookback_period_days=5, timeframe_selection='1d')
        self.assertTrue(df_hr_empty.empty or df_hr_empty['hedge_ratio'].isnull().all())

    # --- Tests for calculate_zscore ---

    def test_calculate_zscore_valid_input(self):
        df_s = self.data.copy()
        df_s['spread'] = df_s['log_price_a'] - df_s['log_price_b'] 
        
        days = 20
        df_z = calculate_zscore(df_s, lookback_period_days=days, timeframe_selection='1d')
        
        self.assertIn('spread_ma', df_z.columns); self.assertIn('spread_std', df_z.columns); self.assertIn('zscore', df_z.columns)
        self.assertEqual(df_z.shape[0], self.data_size)

        lb_points = days * 1 # for '1d'
        min_p = max(1, lb_points // 2)
        first_idx = min_p - 1
        
        if first_idx > 0: self.assertTrue(df_z['zscore'].iloc[0:first_idx].isnull().all())
        if len(df_z) > first_idx:
            self.assertFalse(pd.isna(df_z['zscore'].iloc[first_idx]))
            self.assertFalse(df_z['zscore'].iloc[first_idx:].isnull().any())

    def test_calculate_zscore_lookback_points_calculation(self):
        df_s = self.data[['log_price_a']].copy()
        df_s['spread'] = df_s['log_price_a'] - df_s['log_price_a'].shift(1)
        df_s.dropna(inplace=True)

        days_1d = 5; timeframe_1d = '1d'; lb_1d = days_1d * 1; min_p_1d = max(1, lb_1d//2); first_idx_1d = min_p_1d-1
        df_z_1d = calculate_zscore(df_s.copy(), days_1d, timeframe_1d)
        if first_idx_1d > 0: self.assertTrue(df_z_1d['zscore'].iloc[0:first_idx_1d].isnull().all())
        if len(df_z_1d) > first_idx_1d: self.assertFalse(pd.isna(df_z_1d['zscore'].iloc[first_idx_1d]))

        days_1h = 2; timeframe_1h = '1h'; lb_1h = days_1h * 24; min_p_1h = max(1, lb_1h//2); first_idx_1h = min_p_1h-1
        idx_1h = pd.date_range(start='2023-01-01', periods=lb_1h + 5, freq='h')
        data_1h_s = pd.DataFrame({'spread': np.random.rand(len(idx_1h))}, index=idx_1h)
        df_z_1h = calculate_zscore(data_1h_s, days_1h, timeframe_1h)
        if first_idx_1h > 0: self.assertTrue(df_z_1h['zscore'].iloc[0:first_idx_1h].isnull().all())
        if len(df_z_1h) > first_idx_1h: self.assertFalse(pd.isna(df_z_1h['zscore'].iloc[first_idx_1h]))

    def test_calculate_zscore_min_periods(self):
        days = 10
        df_s = pd.DataFrame({'spread': np.arange(30.0)}, index=pd.date_range('2023-01-01', periods=30, freq='D'))
        df_z = calculate_zscore(df_s, lookback_period_days=days, timeframe_selection='1d')
        
        lb_pts = days * 1; min_p = max(1, lb_pts//2); first_idx = min_p-1
        if first_idx > 0:
            self.assertTrue(df_z['spread_ma'].iloc[0:first_idx].isnull().all())
            self.assertTrue(df_z['spread_std'].iloc[0:first_idx].isnull().all())
            self.assertTrue(df_z['zscore'].iloc[0:first_idx].isnull().all())
        if len(df_z) > first_idx:
            self.assertFalse(pd.isna(df_z['spread_ma'].iloc[first_idx]))
            self.assertFalse(pd.isna(df_z['spread_std'].iloc[first_idx])) # Std of linear sequence is not NaN
            self.assertFalse(pd.isna(df_z['zscore'].iloc[first_idx]))

    def test_calculate_zscore_constant_spread(self):
        days = 5; val = 5.0
        df_s = pd.DataFrame({'spread': [val]*20}, index=pd.date_range('2023-01-01', periods=20, freq='D'))
        df_z = calculate_zscore(df_s, days, '1d')

        lb_pts = days*1; min_p = max(1, lb_pts//2); first_idx = min_p-1
        
        if len(df_z) > first_idx:
            assert_series_equal(df_z['spread_ma'].iloc[first_idx:], pd.Series([val]*(20-first_idx), name='ma', index=df_z.index[first_idx:]), check_dtype=False, atol=1e-9, check_names=False)
            
            std_expected = pd.Series([0.0]*(20-first_idx), name='std', index=df_z.index[first_idx:])
            if min_p == 1 and first_idx == 0 and len(std_expected)>0: std_expected.iloc[0] = np.nan # std of 1 element is NaN
            assert_series_equal(df_z['spread_std'].iloc[first_idx:], std_expected, check_dtype=False, atol=1e-9, check_names=False)
            
            # Z-score is NaN if std is 0 or (if min_p=1) std is NaN for the first point
            self.assertTrue(df_z['zscore'].iloc[first_idx:].isnull().all())

    def test_calculate_zscore_trending_spread(self):
        days = 5
        df_s = pd.DataFrame({'spread': np.arange(20.0)}, index=pd.date_range('2023-01-01', periods=20, freq='D'))
        df_z = calculate_zscore(df_s, days, '1d')
        
        lb_pts = days*1; min_p = max(1, lb_pts//2); first_idx = min_p-1
        if len(df_z) > first_idx:
            expected_ma = df_s['spread'].rolling(window=lb_pts, min_periods=min_p).mean().iloc[first_idx]
            self.assertAlmostEqual(df_z['spread_ma'].iloc[first_idx], expected_ma)
            self.assertFalse(df_z['zscore'].iloc[first_idx:].isnull().any())

    def test_calculate_zscore_with_nans_in_spread(self):
        days = 5
        df_s_nan = pd.DataFrame({'spread': np.arange(20.0)}, index=pd.date_range('2023-01-01', periods=20, freq='D'))
        df_s_nan.loc[df_s_nan.index[5:8], 'spread'] = np.nan # NaNs at index 5,6,7
        df_z = calculate_zscore(df_s_nan, days, '1d')

        lb_pts = days*1; min_p = max(1, lb_pts//2); first_idx = min_p-1 # =1 for lb_pts=5
        
        # Before NaNs start affecting the window significantly for min_periods
        if len(df_z) > 4 : self.assertFalse(pd.isna(df_z['zscore'].iloc[4])) # Window [0:5] (data indices 0..4), no NaNs

        # Window for iloc[5] is data_idx[1]..[5](NaN). Current spread value df_s_nan.iloc[5] is NaN. Z-score is NaN.
        if len(df_z) > 5: self.assertTrue(pd.isna(df_z['zscore'].iloc[5]))
        # Window for iloc[6] is data_idx[2]..[6](NaN). Current spread value df_s_nan.iloc[6] is NaN. Z-score is NaN.
        if len(df_z) > 6: self.assertTrue(pd.isna(df_z['zscore'].iloc[6]))
        # Window for iloc[7] is data_idx[3]..[7](NaN). Current spread value df_s_nan.iloc[7] is NaN. Z-score is NaN.
        if len(df_z) > 7: self.assertTrue(pd.isna(df_z['zscore'].iloc[7]))
        
        # Window for iloc[8] is data_idx[4]..[8]. Current spread value df_s_nan.iloc[8] is NOT NaN. Rolling MA/STD valid. Z-score NOT NaN.
        if len(df_z) > 8: self.assertFalse(pd.isna(df_z['zscore'].iloc[8]))
        
        # Window for iloc[9] is data_idx[5](NaN)..data_idx[9]. Current spread value df_s_nan.iloc[9] is NOT NaN. Rolling MA/STD valid. Z-score NOT NaN.
        if len(df_z) > 9: self.assertFalse(pd.isna(df_z['zscore'].iloc[9]))

        # Window for iloc[10] is data_idx[6](NaN)..data_idx[10]. Current spread value df_s_nan.iloc[10] is NOT NaN. Rolling MA/STD valid. Z-score NOT NaN.
        if len(df_z) > 10: self.assertFalse(pd.isna(df_z['zscore'].iloc[10]))

        # After NaNs are out of any window that could affect this point for MA/STD, and current spread value is not NaN
        if len(df_z) > 12: self.assertFalse(pd.isna(df_z['zscore'].iloc[12])) # Window for MA/STD is [8:13], spread.iloc[12] is not NaN.

if __name__ == '__main__':
    unittest.main()
