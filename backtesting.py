# backtesting.py
"""
Backtesting, trade signal generation, and optimization functions.
Using Backtrader library for improved backtesting capabilities.
"""
import numpy as np
import pandas as pd
import backtrader as bt
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import io
import base64
from matplotlib.figure import Figure
import math

# Keep the original function for compatibility
def calculate_secondary_quantity(primary_quantity, hedge_ratio, primary_price, secondary_price, primary_is_asset_a):
    if primary_price == 0 or secondary_price == 0: return 0
    primary_notional = primary_quantity * primary_price
    if primary_is_asset_a:
        secondary_notional_target = primary_notional * hedge_ratio
        secondary_asset_quantity = secondary_notional_target / secondary_price if secondary_price != 0 else 0
    else:
        if hedge_ratio == 0: return 0
        secondary_notional_target = primary_notional / hedge_ratio
        secondary_asset_quantity = secondary_notional_target / secondary_price if secondary_price != 0 else 0
    return abs(secondary_asset_quantity)

# Custom Pair Trading Strategy for Backtrader
class PairTradingStrategy(bt.Strategy):
    params = (
        ('entry_threshold', 2.0),
        ('exit_threshold', 0.5),
        ('stop_loss_threshold', 3.0),
        ('slippage', 0.001),
        ('trading_fee', 0.001),
        ('primary_quantity', 1.0),
        ('primary_is_a', True),
        ('symbol_a', 'Asset A'),
        ('symbol_b', 'Asset B'),
        ('risk_free_rate', 0.02),  # Annual risk-free rate for Sharpe ratio calculation
    )
    
    def __init__(self):
        # Store Z-score data
        self.zscore = self.datas[0].zscore
        self.hedge_ratio = self.datas[0].hedge_ratio
        self.price_a = self.datas[0].close_a
        self.price_b = self.datas[0].close_b
        # Get symbols from parameters
        self.symbol_a = self.p.symbol_a
        self.symbol_b = self.p.symbol_b
        
        # Initialize position tracking
        self.position_type = 0  # 0: no position, 1: long spread, -1: short spread
        self.qty_a = 0
        self.qty_b = 0
        self.entry_price_a = 0
        self.entry_price_b = 0
        self.entry_zscore = 0
        self.entry_hedge_ratio = 0
        self.entry_time = None
        
        # Trade tracking
        self.trades = []
        
        # Performance metrics tracking
        self.daily_returns = []
        self.equity_curve = [1.0]  # Start with 1.0 (100%)
        self.max_drawdown = 0.0
        self.peak_equity = 1.0
        self.trade_durations = []
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.total_trades = 0
        self.current_drawdown = 0.0
        
        # Store dates for return calculation
        self.dates = []
        
    def next(self):
        # Skip if we don't have valid data
        if np.isnan(self.zscore[0]) or np.isnan(self.zscore[-1]):
            return
            
        # Store current date for performance tracking
        current_date = self.datetime.datetime()
        self.dates.append(current_date)
            
        # Current and previous Z-scores
        z = self.zscore[0]
        prev_z = self.zscore[-1]
        
        # Track daily returns and equity curve (if we have at least one trade)
        if len(self.trades) > 0:
            # Calculate current equity based on open positions and closed trades
            current_equity = 1.0 + (self.total_pnl / 100.0)  # Convert to percentage
            
            # Update equity curve
            self.equity_curve.append(current_equity)
            
            # Calculate daily return
            if len(self.equity_curve) >= 2:
                daily_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1.0
                self.daily_returns.append(daily_return)
            
            # Update drawdown metrics
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # No position - check for entry signals
        if self.position_type == 0:
            # Buy spread signal (long A, short B)
            if prev_z >= -self.p.entry_threshold and z < -self.p.entry_threshold:
                self._enter_long_spread()
                
            # Sell spread signal (short A, long B)
            elif prev_z <= self.p.entry_threshold and z > self.p.entry_threshold:
                self._enter_short_spread()
                
        # Long spread position - check for exit signals
        elif self.position_type == 1:
            # Exit conditions
            if ((prev_z <= -self.p.exit_threshold and z > -self.p.exit_threshold) or 
                (prev_z >= -self.p.stop_loss_threshold and z < -self.p.stop_loss_threshold and 
                 self.p.stop_loss_threshold > self.p.entry_threshold)):
                self._exit_position('Exit Long Spread')
                
        # Short spread position - check for exit signals
        elif self.position_type == -1:
            # Exit conditions
            if ((prev_z >= self.p.exit_threshold and z < self.p.exit_threshold) or 
                (prev_z <= self.p.stop_loss_threshold and z > self.p.stop_loss_threshold and 
                 self.p.stop_loss_threshold > self.p.entry_threshold)):
                self._exit_position('Exit Short Spread')
    
    def _enter_long_spread(self):
        # Calculate quantities
        if self.p.primary_is_a:
            self.qty_a = self.p.primary_quantity
            self.qty_b = calculate_secondary_quantity(
                self.qty_a, self.hedge_ratio[0], self.price_a[0], self.price_b[0], True
            )
        else:
            self.qty_b = self.p.primary_quantity
            self.qty_a = calculate_secondary_quantity(
                self.qty_b, self.hedge_ratio[0], self.price_b[0], self.price_a[0], False
            )
            
        # Skip if quantities are invalid
        if self.qty_a == 0 or self.qty_b == 0:
            return
            
        # Record entry details
        self.position_type = 1
        self.entry_price_a = self.price_a[0]
        self.entry_price_b = self.price_b[0]
        self.entry_zscore = self.zscore[0]
        self.entry_hedge_ratio = self.hedge_ratio[0]
        self.entry_time = self.datetime.datetime()
        
    def _enter_short_spread(self):
        # Calculate quantities
        if self.p.primary_is_a:
            self.qty_a = self.p.primary_quantity
            self.qty_b = calculate_secondary_quantity(
                self.qty_a, self.hedge_ratio[0], self.price_a[0], self.price_b[0], True
            )
        else:
            self.qty_b = self.p.primary_quantity
            self.qty_a = calculate_secondary_quantity(
                self.qty_b, self.hedge_ratio[0], self.price_b[0], self.price_a[0], False
            )
            
        # Skip if quantities are invalid
        if self.qty_a == 0 or self.qty_b == 0:
            return
            
        # Record entry details
        self.position_type = -1
        self.entry_price_a = self.price_a[0]
        self.entry_price_b = self.price_b[0]
        self.entry_zscore = self.zscore[0]
        self.entry_hedge_ratio = self.hedge_ratio[0]
        self.entry_time = self.datetime.datetime()
        
    def _exit_position(self, exit_reason):
        # Skip if no position
        if self.position_type == 0:
            return
            
        # Record trade details
        exit_price_a = self.price_a[0]
        exit_price_b = self.price_b[0]
        exit_zscore = self.zscore[0]
        exit_time = self.datetime.datetime()
        
        # Calculate P&L with slippage and fees
        pnl_a_gross, pnl_b_gross, fee_a, fee_b = 0, 0, 0, 0
        
        if self.position_type == 1:  # Long spread (long A, short B)
            # Asset A (long)
            eff_entry_a = self.entry_price_a * (1 + self.p.slippage)
            eff_exit_a = exit_price_a * (1 - self.p.slippage)
            pnl_a_gross = (eff_exit_a - eff_entry_a) * self.qty_a
            fee_a = (self.qty_a * abs(eff_entry_a) + self.qty_a * abs(eff_exit_a)) * self.p.trading_fee
            
            # Asset B (short)
            eff_entry_b_sell = self.entry_price_b * (1 - self.p.slippage)
            eff_exit_b_buy = exit_price_b * (1 + self.p.slippage)
            pnl_b_gross = (eff_entry_b_sell - eff_exit_b_buy) * self.qty_b
            fee_b = (self.qty_b * abs(eff_entry_b_sell) + self.qty_b * abs(eff_exit_b_buy)) * self.p.trading_fee
            
            trade_type = f'Buy Spread (L {self.symbol_a}, S {self.symbol_b})'
            
        else:  # Short spread (short A, long B)
            # Asset A (short)
            eff_entry_a_sell = self.entry_price_a * (1 - self.p.slippage)
            eff_exit_a_buy = exit_price_a * (1 + self.p.slippage)
            pnl_a_gross = (eff_entry_a_sell - eff_exit_a_buy) * self.qty_a
            fee_a = (self.qty_a * abs(eff_entry_a_sell) + self.qty_a * abs(eff_exit_a_buy)) * self.p.trading_fee
            
            # Asset B (long)
            eff_entry_b = self.entry_price_b * (1 + self.p.slippage)
            eff_exit_b = exit_price_b * (1 - self.p.slippage)
            pnl_b_gross = (eff_exit_b - eff_entry_b) * self.qty_b
            fee_b = (self.qty_b * abs(eff_entry_b) + self.qty_b * abs(eff_exit_b)) * self.p.trading_fee
            
            trade_type = f'Sell Spread (S {self.symbol_a}, L {self.symbol_b})'
            
        # Calculate net P&L
        pnl_a_net = pnl_a_gross - fee_a
        pnl_b_net = pnl_b_gross - fee_b
        total_fees = fee_a + fee_b
        net_pnl = pnl_a_net + pnl_b_net
        
        # Calculate P&L as percentage of primary notional
        initial_primary_notional = 0
        if self.p.primary_is_a:
            initial_primary_notional = abs(self.qty_a * self.entry_price_a)
        else:
            initial_primary_notional = abs(self.qty_b * self.entry_price_b)
            
        pnl_pct = (net_pnl / initial_primary_notional) * 100 if initial_primary_notional != 0 else 0
        
        # Calculate trade duration
        trade_duration = (exit_time - self.entry_time).total_seconds() / 86400  # Convert to days
        self.trade_durations.append(trade_duration)
        
        # Update trade statistics
        self.total_trades += 1
        self.total_pnl += net_pnl
        
        if net_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Add trade to log
        self.trades.append({
            'Entry Timestamp': self.entry_time,
            'Exit Timestamp': exit_time,
            'Trade Type': trade_type,
            'Entry Z-score': self.entry_zscore,
            'Exit Z-score': exit_zscore,
            'Entry Price A (Orig)': self.entry_price_a,
            'Entry Price B (Orig)': self.entry_price_b,
            'Exit Price A (Orig)': exit_price_a,
            'Exit Price B (Orig)': exit_price_b,
            'Entry Hedge Ratio': self.entry_hedge_ratio,
            'Qty A': self.qty_a,
            'Qty B': self.qty_b,
            'Primary is A': self.p.primary_is_a,
            'Symbol A': self.symbol_a,
            'Symbol B': self.symbol_b,
            'PnL Asset A (USD)': pnl_a_net,
            'PnL Asset B (USD)': pnl_b_net,
            'Total Fees (USD)': total_fees,
            'Net PnL (USD)': net_pnl,
            'PnL % (Primary Notional)': pnl_pct,
            'Trade Duration (Days)': trade_duration
        })
        
        # Reset position
        self.position_type = 0
        self.qty_a = 0
        self.qty_b = 0
        
    def stop(self):
        # Close any open positions at the end of the backtest
        if self.position_type != 0:
            self._exit_position('End of Backtest')
            
        # Calculate performance metrics
        self.metrics = {}
        
        # Basic metrics
        self.metrics['total_trades'] = self.total_trades
        self.metrics['win_count'] = self.win_count
        self.metrics['loss_count'] = self.loss_count
        self.metrics['win_rate'] = self.win_count / self.total_trades if self.total_trades > 0 else 0
        self.metrics['total_pnl'] = self.total_pnl
        self.metrics['max_drawdown'] = self.max_drawdown * 100  # Convert to percentage
        
        # Average trade metrics
        self.metrics['avg_trade_pnl'] = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
        self.metrics['avg_trade_duration'] = sum(self.trade_durations) / len(self.trade_durations) if self.trade_durations else 0
        
        # Calculate Sharpe ratio
        if len(self.daily_returns) > 0:
            avg_daily_return = sum(self.daily_returns) / len(self.daily_returns)
            std_daily_return = np.std(self.daily_returns) if len(self.daily_returns) > 1 else 0
            
            if std_daily_return > 0:
                daily_risk_free = (1 + self.p.risk_free_rate) ** (1/252) - 1  # Convert annual to daily
                self.metrics['sharpe_ratio'] = (avg_daily_return - daily_risk_free) / std_daily_return * np.sqrt(252)  # Annualize
            else:
                self.metrics['sharpe_ratio'] = 0
        else:
            self.metrics['sharpe_ratio'] = 0
            
        # Calculate Sortino ratio (downside risk only)
        if len(self.daily_returns) > 0:
            avg_daily_return = sum(self.daily_returns) / len(self.daily_returns)
            downside_returns = [r for r in self.daily_returns if r < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 1 else 0
            
            if downside_deviation > 0:
                daily_risk_free = (1 + self.p.risk_free_rate) ** (1/252) - 1  # Convert annual to daily
                self.metrics['sortino_ratio'] = (avg_daily_return - daily_risk_free) / downside_deviation * np.sqrt(252)  # Annualize
            else:
                self.metrics['sortino_ratio'] = 0
        else:
            self.metrics['sortino_ratio'] = 0
            
        # Calculate Calmar ratio (return / max drawdown)
        if self.max_drawdown > 0 and len(self.equity_curve) > 1:
            total_return = self.equity_curve[-1] - self.equity_curve[0]
            self.metrics['calmar_ratio'] = total_return / self.max_drawdown
        else:
            self.metrics['calmar_ratio'] = 0

# Custom data feed for pair trading
class PairTradingData(bt.feeds.PandasData):
    lines = ('zscore', 'hedge_ratio', 'close_a', 'close_b')
    params = (
        ('zscore', None),
        ('hedge_ratio', None),
        ('close_a', None),
        ('close_b', None),
    )

# Main function to generate trade signals using Backtrader
def generate_trade_signals(data_df, config_params):
    """
    Generate trade signals using Backtrader.
    Returns a DataFrame with the same format as the original function for compatibility.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = data_df.copy()
    
    # Extract symbol information
    symbol_a = data['symbol_a'].iloc[0] if 'symbol_a' in data.columns else "Asset A"
    symbol_b = data['symbol_b'].iloc[0] if 'symbol_b' in data.columns else "Asset B"
    
    # Create a copy without string columns that Backtrader can't handle
    numeric_data = data.copy()
    if 'symbol_a' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_a', axis=1)
    if 'symbol_b' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_b', axis=1)
    
    # Create a Backtrader cerebro engine
    cerebro = bt.Cerebro()
    
    # Add the strategy with symbols as parameters
    cerebro.addstrategy(
        PairTradingStrategy,
        entry_threshold=config_params['entry_threshold'],
        exit_threshold=config_params['exit_threshold'],
        stop_loss_threshold=config_params['stop_loss_threshold'],
        slippage=config_params.get('slippage', 0.001),
        trading_fee=config_params.get('trading_fee', 0.001),
        symbol_a=symbol_a,
        symbol_b=symbol_b
    )
    
    # Prepare the data feed with only numeric columns
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
    
    # Create a signals DataFrame with the same format as the original function
    signals = pd.DataFrame(index=data_df.index)
    signals['zscore'] = data_df['zscore']
    signals['signal'] = 0.0
    signals['position'] = 0
    
    # Extract signals from the strategy's trades
    for trade in strategy.trades:
        entry_idx = data_df.index.get_indexer([trade['Entry Timestamp']], method='nearest')[0]
        exit_idx = data_df.index.get_indexer([trade['Exit Timestamp']], method='nearest')[0]
        
        # Set entry signal
        if 'Buy Spread' in trade['Trade Type']:
            signals.loc[signals.index[entry_idx], 'signal'] = 1.0
            for i in range(entry_idx, exit_idx + 1):
                if i < len(signals):
                    signals.loc[signals.index[i], 'position'] = 1
        else:  # Sell Spread
            signals.loc[signals.index[entry_idx], 'signal'] = -1.0
            for i in range(entry_idx, exit_idx + 1):
                if i < len(signals):
                    signals.loc[signals.index[i], 'position'] = -1
        
        # Set exit signal
        signals.loc[signals.index[exit_idx], 'signal'] = 2.0
    
    # Add signal visualization columns
    signals['buy_signal_z'] = np.where(signals['signal'] == 1.0, signals['zscore'], np.nan)
    signals['sell_signal_z'] = np.where(signals['signal'] == -1.0, signals['zscore'], np.nan)
    signals['exit_signal_z'] = np.where(signals['signal'] == 2.0, signals['zscore'], np.nan)
    
    return signals

# Function to generate trade log from Backtrader results
def generate_trade_log(signals_df, data_with_prices_hedge_ratio, global_config, user_primary_quantity, primary_is_A_selected):
    """
    Generate a trade log using Backtrader.
    Returns a DataFrame with the same format as the original function for compatibility.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = data_with_prices_hedge_ratio.copy()
    
    # Extract symbol information
    symbol_a = data['symbol_a'].iloc[0] if 'symbol_a' in data.columns else "Asset A"
    symbol_b = data['symbol_b'].iloc[0] if 'symbol_b' in data.columns else "Asset B"
    
    # Create a copy without string columns that Backtrader can't handle
    numeric_data = data.copy()
    if 'symbol_a' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_a', axis=1)
    if 'symbol_b' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_b', axis=1)
    
    # Create a Backtrader cerebro engine
    cerebro = bt.Cerebro()
    
    # Add the strategy with symbols as parameters
    cerebro.addstrategy(
        PairTradingStrategy,
        entry_threshold=global_config['entry_threshold'],
        exit_threshold=global_config['exit_threshold'],
        stop_loss_threshold=global_config['stop_loss_threshold'],
        slippage=global_config.get('slippage', 0.001),
        trading_fee=global_config.get('trading_fee', 0.001),
        primary_quantity=user_primary_quantity,
        primary_is_a=primary_is_A_selected,
        symbol_a=symbol_a,
        symbol_b=symbol_b
    )
    
    # Prepare the data feed with only numeric columns
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
    
    # Convert strategy trades to DataFrame
    if strategy.trades:
        return pd.DataFrame(strategy.trades)
    else:
        return pd.DataFrame()

# Visualization functions
def plot_equity_curve(strategy):
    """
    Generate an equity curve plot from a backtest strategy.
    Returns a base64 encoded image that can be displayed in Streamlit.
    """
    if not hasattr(strategy, 'equity_curve') or len(strategy.equity_curve) < 2:
        return None
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot equity curve
    dates = strategy.dates if hasattr(strategy, 'dates') and len(strategy.dates) == len(strategy.equity_curve) else range(len(strategy.equity_curve))
    ax.plot(dates, strategy.equity_curve, label='Equity Curve', color='blue')
    
    # Add drawdown shading
    if hasattr(strategy, 'max_drawdown') and strategy.max_drawdown > 0:
        ax.axhline(y=strategy.peak_equity, color='green', linestyle='--', alpha=0.5, label=f'Peak Equity: {strategy.peak_equity:.2f}')
        ax.axhline(y=strategy.peak_equity * (1 - strategy.max_drawdown), color='red', linestyle='--', alpha=0.5, 
                  label=f'Max Drawdown: {strategy.max_drawdown*100:.2f}%')
    
    ax.set_title('Equity Curve')
    ax.set_xlabel('Time')
    ax.set_ylabel('Equity (Starting = 1.0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def plot_drawdown(strategy):
    """
    Generate a drawdown plot from a backtest strategy.
    Returns a base64 encoded image that can be displayed in Streamlit.
    """
    if not hasattr(strategy, 'equity_curve') or len(strategy.equity_curve) < 2:
        return None
    
    # Calculate drawdowns
    drawdowns = []
    peak = strategy.equity_curve[0]
    
    for equity in strategy.equity_curve:
        if equity > peak:
            peak = equity
            drawdowns.append(0)
        else:
            drawdown = (peak - equity) / peak if peak > 0 else 0
            drawdowns.append(drawdown)
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot drawdowns
    dates = strategy.dates if hasattr(strategy, 'dates') and len(strategy.dates) == len(drawdowns) else range(len(drawdowns))
    ax.fill_between(dates, 0, [d * 100 for d in drawdowns], color='red', alpha=0.3)
    ax.plot(dates, [d * 100 for d in drawdowns], color='red', label='Drawdown %')
    
    ax.set_title('Drawdown Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def plot_trade_distribution(strategy):
    """
    Generate a trade distribution plot from a backtest strategy.
    Returns a base64 encoded image that can be displayed in Streamlit.
    """
    if not hasattr(strategy, 'trades') or len(strategy.trades) < 1:
        return None
    
    # Extract PnL values from trades
    pnl_values = [trade['Net PnL (USD)'] for trade in strategy.trades]
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Create histogram
    n, bins, patches = ax.hist(pnl_values, bins=20, alpha=0.7, color='blue')
    
    # Color positive and negative bars differently
    for i in range(len(patches)):
        if bins[i] < 0:
            patches[i].set_facecolor('red')
        else:
            patches[i].set_facecolor('green')
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    ax.set_title('Trade P&L Distribution')
    ax.set_xlabel('P&L (USD)')
    ax.set_ylabel('Number of Trades')
    ax.grid(True, alpha=0.3)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def run_backtest_with_metrics(data_df, config_params, user_primary_quantity, primary_is_A_selected):
    """
    Run a complete backtest and return both the trade log and performance metrics.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = data_df.copy()
    
    # Extract symbol information
    symbol_a = data['symbol_a'].iloc[0] if 'symbol_a' in data.columns else "Asset A"
    symbol_b = data['symbol_b'].iloc[0] if 'symbol_b' in data.columns else "Asset B"
    
    # Create a copy without string columns that Backtrader can't handle
    numeric_data = data.copy()
    if 'symbol_a' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_a', axis=1)
    if 'symbol_b' in numeric_data.columns:
        numeric_data = numeric_data.drop('symbol_b', axis=1)
    
    # Create a Backtrader cerebro engine
    cerebro = bt.Cerebro()
    
    # Add the strategy with symbols as parameters
    cerebro.addstrategy(
        PairTradingStrategy,
        entry_threshold=config_params['entry_threshold'],
        exit_threshold=config_params['exit_threshold'],
        stop_loss_threshold=config_params['stop_loss_threshold'],
        slippage=config_params.get('slippage', 0.001),
        trading_fee=config_params.get('trading_fee', 0.001),
        primary_quantity=user_primary_quantity,
        primary_is_a=primary_is_A_selected,
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
    
    # Return both the trade log and the metrics
    trade_log = pd.DataFrame(strategy.trades) if strategy.trades else pd.DataFrame()
    
    return {
        'trade_log': trade_log,
        'metrics': strategy.metrics if hasattr(strategy, 'metrics') else {},
        'equity_curve_plot': plot_equity_curve(strategy),
        'drawdown_plot': plot_drawdown(strategy),
        'trade_distribution_plot': plot_trade_distribution(strategy)
    }
