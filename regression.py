# regression.py
"""
Statistical regression and z-score calculation functions.
"""
import numpy as np
import statsmodels.api as sm

def calculate_hedge_ratio(data_to_process, lookback_period_days, timeframe_selection):
    points_per_day = {
        '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2,
        '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1, '1w': 1/7, 'D': 1, 'W': 1/7
    }
    lookback_points = int(lookback_period_days * points_per_day.get(timeframe_selection, 1))
    if lookback_points < 2: lookback_points = 2
    data = data_to_process.copy()
    data['hedge_ratio'] = np.nan
    data['spread'] = np.nan
    data['ols_intercept'] = np.nan
    for i in range(lookback_points, len(data)):
        y_series = data['log_price_a'].iloc[i-lookback_points:i]
        X_series = data['log_price_b'].iloc[i-lookback_points:i]
        if y_series.isnull().any() or X_series.isnull().any() or len(y_series) < 2 or len(X_series) < 2:
            data.loc[data.index[i], 'hedge_ratio'] = np.nan
            data.loc[data.index[i], 'ols_intercept'] = np.nan
            continue
        X_with_const = sm.add_constant(X_series, prepend=True)
        try:
            model = sm.OLS(y_series, X_with_const).fit()
            if len(model.params) > 1:
                intercept = model.params.iloc[0]
                beta = model.params.iloc[1]
                data.loc[data.index[i], 'ols_intercept'] = intercept
                data.loc[data.index[i], 'hedge_ratio'] = beta
                data.loc[data.index[i], 'spread'] = data['log_price_a'].iloc[i] - beta * data['log_price_b'].iloc[i]
            else:
                data.loc[data.index[i], 'hedge_ratio'] = np.nan
                data.loc[data.index[i], 'ols_intercept'] = np.nan
        except Exception:
            data.loc[data.index[i], 'hedge_ratio'] = np.nan
            data.loc[data.index[i], 'ols_intercept'] = np.nan
    return data

def calculate_zscore(data_with_spread, lookback_period_days, timeframe_selection):
    points_per_day = {
        '1m': 24 * 60, '5m': 24 * 12, '15m': 24 * 4, '30m': 24 * 2,
        '1h': 24, '2h': 12, '4h': 6, '6h': 4, '12h': 2, '1d': 1, '1w': 1/7, 'D': 1, 'W': 1/7
    }
    lookback_points = int(lookback_period_days * points_per_day.get(timeframe_selection, 1))
    if lookback_points < 2: lookback_points = 2
    result_df = data_with_spread.copy()
    result_df['spread_ma'] = result_df['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).mean()
    result_df['spread_std'] = result_df['spread'].rolling(window=lookback_points, min_periods=max(1, lookback_points//2)).std()
    result_df['zscore'] = (result_df['spread'] - result_df['spread_ma']) / result_df['spread_std']
    return result_df
