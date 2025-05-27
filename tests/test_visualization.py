import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import plotly.graph_objects as go # For type checking
import plotly.express as px # For type checking

# Add parent directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization import plot_price_history, plot_correlation

class TestPlotPriceHistory(unittest.TestCase):
    def setUp(self):
        self.merged_df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'close_a': [100, 101, 102],
            'close_b': [50, 51, 52]
        }).set_index('timestamp')
        self.asset_a_symbol = "AssetA"
        self.asset_b_symbol = "AssetB"

    @patch('visualization.make_subplots')
    @patch('visualization.go.Scatter') # Patch go.Scatter specifically
    def test_plot_price_history_structure_and_data(self, mock_go_scatter, mock_make_subplots):
        # Mock the figure object that make_subplots would return, to allow method chaining
        mock_figure = MagicMock(spec=go.Figure)
        mock_make_subplots.return_value = mock_figure

        # Call the function
        fig = plot_price_history(self.merged_df, self.asset_a_symbol, self.asset_b_symbol)

        # 1. Verify it returns a Plotly Figure object (or our mock of it)
        self.assertEqual(fig, mock_figure)

        # 2. Check make_subplots call
        mock_make_subplots.assert_called_once_with(specs=[[{"secondary_y": True}]])

        # 3. Check go.Scatter calls (two traces)
        self.assertEqual(mock_go_scatter.call_count, 2)
        
        # Call arguments for go.Scatter: call_args_list[0] is first call, call_args_list[1] is second
        # First call (Asset A)
        args_a, kwargs_a = mock_go_scatter.call_args_list[0]
        pd.testing.assert_index_equal(kwargs_a['x'], self.merged_df.index)
        pd.testing.assert_series_equal(kwargs_a['y'], self.merged_df['close_a'], check_names=False)
        self.assertEqual(kwargs_a['name'], self.asset_a_symbol)
        self.assertEqual(kwargs_a['line'], dict(color='blue'))
        
        # Second call (Asset B)
        args_b, kwargs_b = mock_go_scatter.call_args_list[1]
        pd.testing.assert_index_equal(kwargs_b['x'], self.merged_df.index)
        pd.testing.assert_series_equal(kwargs_b['y'], self.merged_df['close_b'], check_names=False)
        self.assertEqual(kwargs_b['name'], self.asset_b_symbol)
        self.assertEqual(kwargs_b['line'], dict(color='red'))

        # 4. Check add_trace calls on the figure mock
        self.assertEqual(mock_figure.add_trace.call_count, 2)
        mock_figure.add_trace.assert_any_call(mock_go_scatter.return_value, secondary_y=False)
        mock_figure.add_trace.assert_any_call(mock_go_scatter.return_value, secondary_y=True)
        
        # 5. Check layout updates
        mock_figure.update_layout.assert_called_once_with(
            title_text=f'Price History: {self.asset_a_symbol} and {self.asset_b_symbol}',
            xaxis_title='Date',
            legend_title_text='Assets'
        )
        
        # 6. Check y-axes updates
        self.assertEqual(mock_figure.update_yaxes.call_count, 2)
        mock_figure.update_yaxes.assert_any_call(title_text=f"<b>{self.asset_a_symbol} Price</b>", secondary_y=False, color='blue')
        mock_figure.update_yaxes.assert_any_call(title_text=f"<b>{self.asset_b_symbol} Price</b>", secondary_y=True, color='red')


class TestPlotCorrelation(unittest.TestCase):
    def setUp(self):
        self.asset_a_symbol = "TokenA"
        self.asset_b_symbol = "TokenB"
        self.timeframe = "1D"
        
        # Data for successful plot
        self.merged_df_success = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'close_a': [100, 102, 101, 103, 105], # % changes: NaN, 2, -0.98, 1.98, 1.94
            'close_b': [50, 51, 50, 52, 53]      # % changes: NaN, 2, -1.96, 4, 1.92
        }).set_index('timestamp')

        # Data for insufficient data warning (less than 2 points after pct_change and dropna)
        self.merged_df_insufficient = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01']),
            'close_a': [100],
            'close_b': [50]
        }).set_index('timestamp')
        
        self.merged_df_empty = pd.DataFrame(columns=['close_a', 'close_b'])


    @patch('visualization.st')
    @patch('visualization.px.scatter') # Patching plotly.express.scatter
    def test_successful_plot_generation(self, mock_px_scatter, mock_st_viz):
        mock_figure = MagicMock(spec=go.Figure) # Mock the figure object px.scatter would return
        mock_px_scatter.return_value = mock_figure

        plot_correlation(self.merged_df_success, self.asset_a_symbol, self.asset_b_symbol, self.timeframe)

        # 1. Verify pct_change calculations (indirectly through px.scatter call)
        # Expected data for scatter plot after pct_change().fillna(0).dropna()
        # The first row of pct_change() is NaN, then filled with 0. dropna() does nothing after fillna(0) if no other NaNs.
        # So, the first row of zeros should be included.
        expected_a_pct_change = self.merged_df_success['close_a'].pct_change().fillna(0) * 100
        expected_b_pct_change = self.merged_df_success['close_b'].pct_change().fillna(0) * 100
        expected_plot_corr_df = pd.DataFrame({
            'a_pct_change': expected_a_pct_change.values,
            'b_pct_change': expected_b_pct_change.values
        })
        # Correlation is calculated on the df passed to px.scatter, which includes the 0s.
        expected_correlation = expected_plot_corr_df['a_pct_change'].corr(expected_plot_corr_df['b_pct_change'])

        # 2. Assert px.scatter called correctly
        mock_px_scatter.assert_called_once()
        call_args, call_kwargs = mock_px_scatter.call_args
        
        # Check the DataFrame passed to px.scatter
        pd.testing.assert_frame_equal(call_args[0].reset_index(drop=True), expected_plot_corr_df.reset_index(drop=True), atol=1e-5) # Compare values
        
        self.assertEqual(call_kwargs['x'], 'a_pct_change')
        self.assertEqual(call_kwargs['y'], 'b_pct_change')
        self.assertEqual(call_kwargs['title'], f'{self.asset_a_symbol} vs {self.asset_b_symbol} {self.timeframe} % Change Correlation (ρ = {expected_correlation:.2f})')
        self.assertEqual(call_kwargs['labels'], {'a_pct_change': f'{self.asset_a_symbol} % Change', 'b_pct_change': f'{self.asset_b_symbol} % Change'})
        self.assertEqual(call_kwargs['trendline'], 'ols')
        self.assertEqual(call_kwargs['template'], 'plotly_white')

        # 3. Assert st.plotly_chart called
        mock_st_viz.plotly_chart.assert_called_once_with(mock_figure, use_container_width=True)
        
        # 4. Ensure st.warning NOT called
        mock_st_viz.warning.assert_not_called()

    @patch('visualization.st')
    @patch('visualization.px.scatter')
    def test_insufficient_data_for_correlation(self, mock_px_scatter, mock_st_viz):
        # Test with merged_df_insufficient (1 row, so pct_change leads to 0 non-NaN rows)
        plot_correlation(self.merged_df_insufficient, self.asset_a_symbol, self.asset_b_symbol, self.timeframe)
        mock_st_viz.warning.assert_called_once_with("Not enough data points to calculate and plot price percentage change correlation.")
        mock_px_scatter.assert_not_called()
        mock_st_viz.plotly_chart.assert_not_called()

        mock_st_viz.warning.reset_mock() # Reset for the next scenario

        # Test with empty df
        plot_correlation(self.merged_df_empty, self.asset_a_symbol, self.asset_b_symbol, self.timeframe)
        mock_st_viz.warning.assert_called_once_with("Not enough data points to calculate and plot price percentage change correlation.")
        mock_px_scatter.assert_not_called()
        mock_st_viz.plotly_chart.assert_not_called()


if __name__ == '__main__':
    unittest.main()
