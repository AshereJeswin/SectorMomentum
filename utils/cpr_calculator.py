import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_cpr(df):
    """
    Calculate Central Pivot Range (CPR) for the given data.
    
    Parameters:
    - df: DataFrame with OHLC data
    
    Returns:
    - DataFrame with CPR values
    """
    if df is None or df.empty:
        return None
    
    # Create a copy to avoid modifying original data
    result = df.copy()
    
    # Calculate Pivot Point (PP)
    result['PP'] = (result['High'] + result['Low'] + result['Close']) / 3
    
    # Calculate Bottom Central Pivot (BC)
    result['BC'] = (result['High'] + result['Low']) / 2
    
    # Calculate Top Central Pivot (TC)
    result['TC'] = (result['PP'] - result['BC']) + result['PP']
    
    # Calculate additional support and resistance levels
    result['R3'] = result['High'] + 2 * (result['PP'] - result['Low'])
    result['R2'] = result['PP'] + (result['High'] - result['Low'])
    result['R1'] = 2 * result['PP'] - result['Low']
    
    result['S1'] = 2 * result['PP'] - result['High']
    result['S2'] = result['PP'] - (result['High'] - result['Low'])
    result['S3'] = result['Low'] - 2 * (result['High'] - result['PP'])
    
    # Calculate CPR Width - a measure of volatility
    result['CPR_Width'] = result['TC'] - result['BC']
    
    # Calculate relative position of price to CPR
    result['Price_Position'] = np.where(
        result['Close'] > result['TC'], "Above CPR",
        np.where(result['Close'] < result['BC'], "Below CPR", "Within CPR")
    )
    
    return result

def plot_cpr(df_with_cpr, symbol, timeframe="Daily"):
    """
    Create candlestick chart with CPR levels
    
    Parameters:
    - df_with_cpr: DataFrame with OHLC and CPR data
    - symbol: Stock symbol
    - timeframe: Timeframe of the data (e.g., "Daily", "Weekly")
    
    Returns:
    - Plotly figure object
    """
    if df_with_cpr is None or df_with_cpr.empty:
        return None
    
    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2],
        subplot_titles=(f"{symbol} - {timeframe} Chart with CPR", "Volume")
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_with_cpr.index,
            open=df_with_cpr['Open'],
            high=df_with_cpr['High'],
            low=df_with_cpr['Low'],
            close=df_with_cpr['Close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    colors = ['#DC2626' if row['Open'] > row['Close'] else '#16A34A' for _, row in df_with_cpr.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df_with_cpr.index,
            y=df_with_cpr['Volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add CPR levels
    date_range = df_with_cpr.index.tolist()
    
    # Get the last day's CPR values
    last_idx = len(df_with_cpr) - 1
    
    if last_idx >= 0:
        # Add Pivot Point line
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['PP'],
                mode='lines',
                name='Pivot Point (PP)',
                line=dict(color='#2962FF', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add Top Central Pivot line
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['TC'],
                mode='lines',
                name='Top Central Pivot (TC)',
                line=dict(color='#16A34A', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add Bottom Central Pivot line
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['BC'],
                mode='lines',
                name='Bottom Central Pivot (BC)',
                line=dict(color='#DC2626', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add resistance levels
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['R1'],
                mode='lines',
                name='Resistance 1 (R1)',
                line=dict(color='#16A34A', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['R2'],
                mode='lines',
                name='Resistance 2 (R2)',
                line=dict(color='#16A34A', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['R3'],
                mode='lines',
                name='Resistance 3 (R3)',
                line=dict(color='#16A34A', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        # Add support levels
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['S1'],
                mode='lines',
                name='Support 1 (S1)',
                line=dict(color='#DC2626', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['S2'],
                mode='lines',
                name='Support 2 (S2)',
                line=dict(color='#DC2626', width=1, dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=date_range,
                y=df_with_cpr['S3'],
                mode='lines',
                name='Support 3 (S3)',
                line=dict(color='#DC2626', width=1, dash='dot')
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} with Central Pivot Range (CPR)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Style volume subplot
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Update Y-axis to show complete CPR range
    all_levels = ['Close', 'PP', 'TC', 'BC', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']
    y_values = [val for col in all_levels for val in df_with_cpr[col].dropna().values]
    
    if y_values:
        y_min = min(y_values) * 0.98
        y_max = max(y_values) * 1.02
        
        fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    
    return fig

def get_cpr_analysis(df_with_cpr):
    """
    Generate analysis text based on CPR levels
    
    Parameters:
    - df_with_cpr: DataFrame with OHLC and CPR data
    
    Returns:
    - String with CPR analysis
    """
    if df_with_cpr is None or df_with_cpr.empty:
        return "No data available for CPR analysis."
    
    # Get the most recent data point
    latest = df_with_cpr.iloc[-1]
    
    # Basic analysis
    current_price = latest['Close']
    pp = latest['PP']
    tc = latest['TC']
    bc = latest['BC']
    cpr_width = latest['CPR_Width']
    
    # Calculate width percentage relative to price
    width_percentage = (cpr_width / current_price) * 100
    
    # Determine trend based on position relative to CPR
    if current_price > tc:
        trend = "Bullish"
        strength = "Strong" if current_price > latest['R1'] else "Moderate"
    elif current_price < bc:
        trend = "Bearish"
        strength = "Strong" if current_price < latest['S1'] else "Moderate"
    else:
        trend = "Neutral/Consolidating"
        strength = "Undecided"
        
    # Check if price is testing CPR levels
    if abs(current_price - tc) / current_price < 0.002:
        level_test = "Price is currently testing the Top Central Pivot (TC), which may act as resistance."
    elif abs(current_price - bc) / current_price < 0.002:
        level_test = "Price is currently testing the Bottom Central Pivot (BC), which may act as support."
    elif abs(current_price - pp) / current_price < 0.002:
        level_test = "Price is currently testing the Pivot Point (PP), which is a key level for determining intraday direction."
    else:
        level_test = ""
    
    # Build the analysis text
    analysis = f"""
    ### CPR Analysis Summary

    **Current Status**: {trend} ({strength})
    **CPR Width**: {cpr_width:.2f} ({width_percentage:.2f}% of price)
    
    **Key Levels**:
    - Top Central Pivot (TC): {tc:.2f}
    - Pivot Point (PP): {pp:.2f}
    - Bottom Central Pivot (BC): {bc:.2f}
    
    **Price Position**: {latest['Price_Position']}
    
    **Interpretation**:
    """
    
    if width_percentage > 2.0:
        analysis += "- Wide CPR indicates higher volatility and potential for significant price movement.\n"
    elif width_percentage < 0.5:
        analysis += "- Narrow CPR suggests consolidation and lower volatility.\n"
    
    if trend == "Bullish":
        analysis += "- Price above CPR suggests bullish sentiment. TC now acts as first support.\n"
        if strength == "Strong":
            analysis += "- Strong bullish momentum with price above R1. Watch for continuation.\n"
    elif trend == "Bearish":
        analysis += "- Price below CPR suggests bearish sentiment. BC now acts as first resistance.\n"
        if strength == "Strong":
            analysis += "- Strong bearish momentum with price below S1. Watch for continuation.\n"
    else:
        analysis += "- Price within CPR indicates indecision. Watch for breakout in either direction.\n"
    
    if level_test:
        analysis += f"- {level_test}\n"
    
    # Next day outlook
    analysis += "\n**Next Session Outlook**:\n"
    
    if trend == "Bullish":
        analysis += f"- Potential upside targets: R1 ({latest['R1']:.2f}), R2 ({latest['R2']:.2f})\n"
        analysis += f"- Key support levels: TC ({tc:.2f}), PP ({pp:.2f})\n"
    elif trend == "Bearish":
        analysis += f"- Potential downside targets: S1 ({latest['S1']:.2f}), S2 ({latest['S2']:.2f})\n"
        analysis += f"- Key resistance levels: BC ({bc:.2f}), PP ({pp:.2f})\n"
    else:
        analysis += f"- Upside breakout target: TC ({tc:.2f}), R1 ({latest['R1']:.2f})\n"
        analysis += f"- Downside breakout target: BC ({bc:.2f}), S1 ({latest['S1']:.2f})\n"
    
    return analysis
