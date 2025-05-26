# visualization.py
"""
Visualization functions for Streamlit and Plotly charts.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd

def plot_price_history(merged_df, asset_a_symbol, asset_b_symbol):
    fig_prices_plotly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_prices_plotly.add_trace(
        go.Scatter(x=merged_df.index, y=merged_df['close_a'], name=asset_a_symbol, line=dict(color='blue')),
        secondary_y=False,
    )
    fig_prices_plotly.add_trace(
        go.Scatter(x=merged_df.index, y=merged_df['close_b'], name=asset_b_symbol, line=dict(color='red')),
        secondary_y=True,
    )
    fig_prices_plotly.update_layout(title_text=f'Price History: {asset_a_symbol} and {asset_b_symbol}', xaxis_title='Date', legend_title_text='Assets')
    fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_a_symbol} Price</b>", secondary_y=False, color='blue')
    fig_prices_plotly.update_yaxes(title_text=f"<b>{asset_b_symbol} Price</b>", secondary_y=True, color='red')
    return fig_prices_plotly

def plot_correlation(merged_df, asset_a_symbol, asset_b_symbol, timeframe):
    correlation_df = merged_df.copy()
    correlation_df['a_pct_change'] = correlation_df['close_a'].pct_change().fillna(0) * 100
    correlation_df['b_pct_change'] = correlation_df['close_b'].pct_change().fillna(0) * 100
    plot_corr_df = correlation_df[['a_pct_change', 'b_pct_change']].dropna()
    if not plot_corr_df.empty and len(plot_corr_df) > 1:
        correlation_value = plot_corr_df['a_pct_change'].corr(plot_corr_df['b_pct_change'])
        fig_correlation = px.scatter(
            plot_corr_df,
            x='a_pct_change',
            y='b_pct_change',
            title=f'{asset_a_symbol} vs {asset_b_symbol} {timeframe} % Change Correlation (ρ = {correlation_value:.2f})',
            labels={'a_pct_change': f'{asset_a_symbol} % Change', 'b_pct_change': f'{asset_b_symbol} % Change'},
            trendline='ols',
            template='plotly_white'
        )
        fig_correlation.update_layout(xaxis_title=f'{asset_a_symbol} % Change', yaxis_title=f'{asset_b_symbol} % Change')
        st.plotly_chart(fig_correlation, use_container_width=True)
    else:
        st.warning("Not enough data points to calculate and plot price percentage change correlation.")
