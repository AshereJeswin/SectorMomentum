import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_fetcher import get_stock_data, get_sector_symbols, get_market_indices
from utils.cpr_calculator import calculate_cpr

# Page configuration
st.set_page_config(
    page_title="Sector Analysis | Stock Market Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = "Technology"

# Define available sectors
sectors = [
    "Technology", 
    "Finance", 
    "Healthcare", 
    "Consumer", 
    "Energy", 
    "Industrials"
]

# Page Header
st.title("📊 Sector Analysis")
st.markdown("Analyze stock performance and momentum across different market sectors")

# Market Overview (Top Section)
st.subheader("Market Indices Performance")

# Fetch market indices data
market_indices = get_market_indices()

# Display market indices in a grid
cols = st.columns(len(market_indices))
for i, (index_name, data) in enumerate(market_indices.items()):
    if data:
        with cols[i]:
            # Make sure we have scalar values for the metrics
            daily_change = float(data["daily_change"]) if data["daily_change"] is not None else 0
            price = float(data["price"]) if data["price"] is not None else 0
                
            color = "#16A34A" if daily_change >= 0 else "#DC2626"
            st.metric(
                label=index_name,
                value=f"${price:.2f}",
                delta=f"{daily_change:.2f}%",
                delta_color="normal"
            )

# Main content
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Analysis Controls")
    
    # Sector selection
    selected_sector = st.selectbox(
        "Select Sector",
        options=sectors,
        index=sectors.index(st.session_state.selected_sector) if st.session_state.selected_sector in sectors else 0
    )
    st.session_state.selected_sector = selected_sector
    
    # Time period selection
    time_period = st.selectbox(
        "Time Period",
        options=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        index=1
    )
    
    # Map selection to yfinance period
    period_mapping = {
        "1 Week": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    yf_period = period_mapping[time_period]
    
    # Performance metric
    performance_metric = st.selectbox(
        "Performance Metric",
        options=["Percentage Change", "Price", "Volume", "Volatility"],
        index=0
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        # Moving average options
        show_ma = st.checkbox("Show Moving Averages", value=True)
        
        if show_ma:
            ma_periods = st.multiselect(
                "MA Periods",
                options=[5, 10, 20, 50, 100, 200],
                default=[20, 50]
            )
        
        # Technical indicators
        show_rsi = st.checkbox("Show RSI", value=False)
        show_macd = st.checkbox("Show MACD", value=False)
        
        # CPR analysis
        show_cpr = st.checkbox("Show CPR Analysis", value=True)

# Get symbols for the selected sector
sector_symbols = get_sector_symbols(selected_sector)

if not sector_symbols:
    st.error(f"No symbols available for {selected_sector} sector")
else:
    # Fetch stock data for all symbols in the sector
    sector_data = {}
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(sector_symbols):
        sector_data[symbol] = get_stock_data(symbol, period=yf_period)
        progress_bar.progress((i + 1) / len(sector_symbols))
    
    progress_bar.empty()
    
    # Calculate performance metrics for all symbols
    performance_df = []
    
    for symbol, df in sector_data.items():
        if df is not None and not df.empty:
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            pct_change = ((end_price - start_price) / start_price) * 100
            
            # Calculate volatility (standard deviation of daily returns)
            daily_returns = df['Close'].pct_change().dropna()
            volatility = daily_returns.std() * 100
            
            # Calculate average volume
            avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
            
            # Calculate momentum (rate of change over the last 5 periods)
            momentum = ((df['Close'].iloc[-1] - df['Close'].iloc[-6 if len(df) > 5 else 0]) / 
                       df['Close'].iloc[-6 if len(df) > 5 else 0]) * 100 if len(df) > 1 else 0
            
            # Add to performance dataframe
            performance_df.append({
                'Symbol': symbol,
                'Current Price': end_price,
                'Change (%)': pct_change,
                'Volume': avg_volume,
                'Volatility (%)': volatility,
                'Momentum': momentum
            })
    
    if performance_df:
        performance_df = pd.DataFrame(performance_df)
        
        # Sort by the selected performance metric
        if performance_metric == "Percentage Change":
            performance_df = performance_df.sort_values('Change (%)', ascending=False)
        elif performance_metric == "Price":
            performance_df = performance_df.sort_values('Current Price', ascending=False)
        elif performance_metric == "Volume":
            performance_df = performance_df.sort_values('Volume', ascending=False)
        elif performance_metric == "Volatility":
            performance_df = performance_df.sort_values('Volatility (%)', ascending=False)
        
        # Display sector performance in two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader(f"{selected_sector} Sector Performance")
            
            # Format the display dataframe
            display_df = performance_df.copy()
            display_df['Current Price'] = display_df['Current Price'].map('${:,.2f}'.format)
            display_df['Change (%)'] = display_df['Change (%)'].map('{:+.2f}%'.format)
            display_df['Volatility (%)'] = display_df['Volatility (%)'].map('{:.2f}%'.format)
            display_df['Momentum'] = display_df['Momentum'].map('{:+.2f}%'.format)
            display_df['Volume'] = display_df['Volume'].map('{:,.0f}'.format)
            
            st.dataframe(display_df, use_container_width=True)
        
        with col2:
            st.subheader("Performance Comparison")
            
            # Create bar chart for the selected metric
            if performance_metric == "Percentage Change":
                fig = px.bar(
                    performance_df, 
                    x='Symbol', 
                    y='Change (%)',
                    color='Change (%)',
                    color_continuous_scale=['#DC2626', '#FFFFFF', '#16A34A'],
                    title=f"{time_period} Percentage Change by Symbol"
                )
            elif performance_metric == "Price":
                fig = px.bar(
                    performance_df, 
                    x='Symbol', 
                    y='Current Price',
                    color='Change (%)',
                    color_continuous_scale=['#DC2626', '#FFFFFF', '#16A34A'],
                    title=f"Current Price by Symbol"
                )
            elif performance_metric == "Volume":
                fig = px.bar(
                    performance_df, 
                    x='Symbol', 
                    y='Volume',
                    color='Volume',
                    color_continuous_scale='Blues',
                    title=f"Average Volume by Symbol"
                )
            elif performance_metric == "Volatility":
                fig = px.bar(
                    performance_df, 
                    x='Symbol', 
                    y='Volatility (%)',
                    color='Volatility (%)',
                    color_continuous_scale='Reds',
                    title=f"Volatility by Symbol"
                )
            
            # Update layout
            fig.update_layout(
                height=400,
                xaxis_title="Symbol",
                yaxis_title=performance_metric,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display momentum analysis
        st.subheader("Sector Momentum Analysis")
        
        # Create a scatter plot of change vs. momentum
        momentum_fig = px.scatter(
            performance_df,
            x='Change (%)',
            y='Momentum',
            size='Volume',
            color='Symbol',
            hover_name='Symbol',
            text='Symbol',
            title="Momentum vs. Performance Analysis",
            size_max=60,
            height=500
        )
        
        # Add quadrant lines
        momentum_fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
        momentum_fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
        
        # Add annotations for quadrants
        momentum_fig.add_annotation(x=performance_df['Change (%)'].max()/2, y=performance_df['Momentum'].max()/2,
                            text="Strong Momentum (Buy)",
                            showarrow=False,
                            font=dict(size=12, color="#16A34A"))
        momentum_fig.add_annotation(x=performance_df['Change (%)'].min()/2, y=performance_df['Momentum'].max()/2,
                            text="Improving (Potential Buy)",
                            showarrow=False,
                            font=dict(size=12, color="#16A34A"))
        momentum_fig.add_annotation(x=performance_df['Change (%)'].max()/2, y=performance_df['Momentum'].min()/2,
                            text="Weakening (Caution)",
                            showarrow=False,
                            font=dict(size=12, color="#DC2626"))
        momentum_fig.add_annotation(x=performance_df['Change (%)'].min()/2, y=performance_df['Momentum'].min()/2,
                            text="Weak (Avoid/Sell)",
                            showarrow=False,
                            font=dict(size=12, color="#DC2626"))
        
        # Update layout
        momentum_fig.update_layout(
            xaxis_title="Overall Change (%)",
            yaxis_title="Momentum (Rate of Change)",
            template="plotly_white"
        )
        
        st.plotly_chart(momentum_fig, use_container_width=True)
        
        # Individual Stock Analysis
        st.markdown("---")
        st.subheader("Individual Stock Analysis")
        
        # Stock selector
        selected_symbol = st.selectbox(
            "Select Stock for Detailed Analysis",
            options=sector_symbols,
            index=0
        )
        
        # Get data for selected symbol
        df = sector_data.get(selected_symbol)
        
        if df is not None and not df.empty:
            # Create tabs for different analyses
            tabs = st.tabs(["Price Chart", "Technical Indicators", "CPR Analysis"])
            
            with tabs[0]:  # Price Chart
                # Create candlestick chart
                fig = go.Figure()
                
                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="Price"
                    )
                )
                
                # Add volume as a bar chart on the same figure
                colors = ['#DC2626' if row['Open'] > row['Close'] else '#16A34A' for _, row in df.iterrows()]
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        marker_color=colors,
                        name="Volume",
                        yaxis="y2"
                    )
                )
                
                # Add moving averages if selected
                if show_ma:
                    for period in ma_periods:
                        if len(df) > period:
                            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df[f'MA{period}'],
                                    mode='lines',
                                    name=f'{period}-day MA',
                                    line=dict(width=1.5)
                                )
                            )
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_symbol} Price Chart - {time_period}",
                    yaxis_title="Price",
                    xaxis_title="Date",
                    height=600,
                    template="plotly_white",
                    xaxis_rangeslider_visible=False,
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add price statistics
                if not df.empty:
                    stats_cols = st.columns(5)
                    with stats_cols[0]:
                        st.metric("Current", f"${df['Close'].iloc[-1]:.2f}")
                    with stats_cols[1]:
                        st.metric("Open", f"${df['Open'].iloc[-1]:.2f}")
                    with stats_cols[2]:
                        st.metric("High", f"${df['High'].iloc[-1]:.2f}")
                    with stats_cols[3]:
                        st.metric("Low", f"${df['Low'].iloc[-1]:.2f}")
                    with stats_cols[4]:
                        day_change = ((df['Close'].iloc[-1] - df['Open'].iloc[-1]) / df['Open'].iloc[-1]) * 100
                        st.metric("Day Change", f"{day_change:.2f}%")
            
            with tabs[1]:  # Technical Indicators
                # Add technical indicators
                technical_cols = st.columns(2)
                
                with technical_cols[0]:
                    # Calculate and display RSI
                    if show_rsi:
                        st.subheader("Relative Strength Index (RSI)")
                        
                        # Calculate RSI
                        delta = df['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        
                        # Calculate RS and RSI
                        rs = gain / loss
                        df['RSI'] = 100 - (100 / (1 + rs))
                        
                        # Create RSI chart
                        rsi_fig = go.Figure()
                        
                        rsi_fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['RSI'],
                                mode='lines',
                                name='RSI',
                                line=dict(color='#2962FF', width=1.5)
                            )
                        )
                        
                        # Add reference lines
                        rsi_fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="#DC2626")
                        rsi_fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="#16A34A")
                        rsi_fig.add_hline(y=50, line_width=1, line_dash="dash", line_color="gray")
                        
                        # Update layout
                        rsi_fig.update_layout(
                            title=f"{selected_symbol} - RSI (14)",
                            yaxis_title="RSI Value",
                            xaxis_title="Date",
                            height=300,
                            template="plotly_white",
                            yaxis=dict(range=[0, 100])
                        )
                        
                        st.plotly_chart(rsi_fig, use_container_width=True)
                        
                        # RSI interpretation
                        latest_rsi = df['RSI'].iloc[-1]
                        if latest_rsi > 70:
                            st.warning(f"RSI: {latest_rsi:.2f} - Potentially Overbought")
                        elif latest_rsi < 30:
                            st.success(f"RSI: {latest_rsi:.2f} - Potentially Oversold")
                        else:
                            st.info(f"RSI: {latest_rsi:.2f} - Neutral")
                
                with technical_cols[1]:
                    # Calculate and display MACD
                    if show_macd:
                        st.subheader("MACD (Moving Average Convergence Divergence)")
                        
                        # Ensure MACD is calculated
                        if 'MACD' not in df.columns:
                            df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                            df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                            df['MACD'] = df['EMA12'] - df['EMA26']
                            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
                            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
                        
                        # Create MACD chart
                        macd_fig = go.Figure()
                        
                        # Add MACD line
                        macd_fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['MACD'],
                                mode='lines',
                                name='MACD',
                                line=dict(color='#2962FF', width=1.5)
                            )
                        )
                        
                        # Add Signal line
                        macd_fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['Signal_Line'],
                                mode='lines',
                                name='Signal Line',
                                line=dict(color='#DC2626', width=1.5)
                            )
                        )
                        
                        # Add Histogram
                        colors = ['#16A34A' if val >= 0 else '#DC2626' for val in df['MACD_Histogram']]
                        
                        macd_fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=df['MACD_Histogram'],
                                name='Histogram',
                                marker_color=colors
                            )
                        )
                        
                        # Add zero line
                        macd_fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="gray")
                        
                        # Update layout
                        macd_fig.update_layout(
                            title=f"{selected_symbol} - MACD (12,26,9)",
                            yaxis_title="MACD Value",
                            xaxis_title="Date",
                            height=300,
                            template="plotly_white",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(macd_fig, use_container_width=True)
                        
                        # MACD interpretation
                        latest_macd = df['MACD'].iloc[-1]
                        latest_signal = df['Signal_Line'].iloc[-1]
                        
                        if latest_macd > latest_signal:
                            st.success(f"MACD ({latest_macd:.3f}) > Signal ({latest_signal:.3f}) - Bullish Signal")
                        else:
                            st.warning(f"MACD ({latest_macd:.3f}) < Signal ({latest_signal:.3f}) - Bearish Signal")
                
                # Momentum indicators
                st.subheader("Momentum Indicators")
                
                # Calculate momentum indicators
                df['ROC'] = df['Close'].pct_change(periods=10) * 100
                
                # Price Rate of Change
                roc_fig = go.Figure()
                
                roc_fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ROC'],
                        mode='lines',
                        name='Rate of Change (10)',
                        line=dict(color='#2962FF', width=1.5)
                    )
                )
                
                # Add zero line
                roc_fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
                
                # Update layout
                roc_fig.update_layout(
                    title=f"{selected_symbol} - Price Rate of Change (10-day)",
                    yaxis_title="ROC Value",
                    xaxis_title="Date",
                    height=300,
                    template="plotly_white"
                )
                
                st.plotly_chart(roc_fig, use_container_width=True)
                
                # ROC interpretation
                latest_roc = df['ROC'].iloc[-1]
                if latest_roc > 5:
                    st.success(f"ROC: {latest_roc:.2f}% - Strong Positive Momentum")
                elif latest_roc > 0:
                    st.info(f"ROC: {latest_roc:.2f}% - Positive Momentum")
                elif latest_roc > -5:
                    st.warning(f"ROC: {latest_roc:.2f}% - Negative Momentum")
                else:
                    st.error(f"ROC: {latest_roc:.2f}% - Strong Negative Momentum")
            
            with tabs[2]:  # CPR Analysis
                if show_cpr:
                    st.subheader("Central Pivot Range (CPR) Analysis")
                    
                    # Calculate CPR for the dataframe
                    df_cpr = calculate_cpr(df)
                    
                    if df_cpr is not None:
                        latest = df_cpr.iloc[-1]
                        
                        # Display CPR values in columns
                        cpr_cols = st.columns(3)
                        with cpr_cols[0]:
                            st.metric("Top Central Pivot (TC)", f"${latest['TC']:.2f}")
                        with cpr_cols[1]:
                            st.metric("Pivot Point (PP)", f"${latest['PP']:.2f}")
                        with cpr_cols[2]:
                            st.metric("Bottom Central Pivot (BC)", f"${latest['BC']:.2f}")
                        
                        # Display support and resistance levels
                        st.subheader("Support and Resistance Levels")
                        level_cols = st.columns(6)
                        with level_cols[0]:
                            st.metric("R3", f"${latest['R3']:.2f}")
                        with level_cols[1]:
                            st.metric("R2", f"${latest['R2']:.2f}")
                        with level_cols[2]:
                            st.metric("R1", f"${latest['R1']:.2f}")
                        with level_cols[3]:
                            st.metric("S1", f"${latest['S1']:.2f}")
                        with level_cols[4]:
                            st.metric("S2", f"${latest['S2']:.2f}")
                        with level_cols[5]:
                            st.metric("S3", f"${latest['S3']:.2f}")
                        
                        # CPR width analysis
                        cpr_width = latest['CPR_Width']
                        width_percentage = (cpr_width / latest['Close']) * 100
                        
                        st.metric("CPR Width", f"${cpr_width:.2f} ({width_percentage:.2f}% of price)")
                        
                        # Display interpretation
                        st.subheader("CPR Interpretation")
                        
                        current_price = latest['Close']
                        if current_price > latest['TC']:
                            st.success(f"Price (${current_price:.2f}) is above the CPR - Bullish")
                            st.markdown("- When price is above CPR, the TC acts as the first support level")
                            st.markdown("- Look for buying opportunities on pullbacks to the TC")
                        elif current_price < latest['BC']:
                            st.error(f"Price (${current_price:.2f}) is below the CPR - Bearish")
                            st.markdown("- When price is below CPR, the BC acts as the first resistance level")
                            st.markdown("- Look for selling opportunities on rallies to the BC")
                        else:
                            st.info(f"Price (${current_price:.2f}) is within the CPR - Neutral/Consolidating")
                            st.markdown("- When price is within the CPR, the market is in a state of equilibrium")
                            st.markdown("- Watch for breakouts above TC (bullish) or below BC (bearish)")
                        
                        # CPR width analysis
                        if width_percentage > 2.0:
                            st.info("Wide CPR indicates higher volatility and potential for significant price movement")
                        elif width_percentage < 0.5:
                            st.info("Narrow CPR suggests consolidation and lower volatility")
                        
                        # Add price position relative to key levels
                        st.subheader("Price Position Relative to Key Levels")
                        
                        level_status = {}
                        level_status["R3"] = "Above" if current_price > latest['R3'] else "Below"
                        level_status["R2"] = "Above" if current_price > latest['R2'] else "Below"
                        level_status["R1"] = "Above" if current_price > latest['R1'] else "Below"
                        level_status["TC"] = "Above" if current_price > latest['TC'] else "Below"
                        level_status["PP"] = "Above" if current_price > latest['PP'] else "Below"
                        level_status["BC"] = "Above" if current_price > latest['BC'] else "Below"
                        level_status["S1"] = "Above" if current_price > latest['S1'] else "Below"
                        level_status["S2"] = "Above" if current_price > latest['S2'] else "Below"
                        level_status["S3"] = "Above" if current_price > latest['S3'] else "Below"
                        
                        level_df = pd.DataFrame({
                            "Level": list(level_status.keys()),
                            "Value": [
                                f"${latest['R3']:.2f}", f"${latest['R2']:.2f}", f"${latest['R1']:.2f}",
                                f"${latest['TC']:.2f}", f"${latest['PP']:.2f}", f"${latest['BC']:.2f}",
                                f"${latest['S1']:.2f}", f"${latest['S2']:.2f}", f"${latest['S3']:.2f}"
                            ],
                            "Position": list(level_status.values())
                        })
                        
                        st.dataframe(level_df, use_container_width=True)
        else:
            st.error(f"No data available for {selected_symbol}")
    else:
        st.error("No performance data available for the selected sector")

# Footer
st.markdown("---")
st.markdown("Powered by yfinance, Streamlit, and Plotly | Data for educational purposes only")

# Disclaimer
with st.expander("Disclaimer"):
    st.warning(
        "This application is for educational and informational purposes only. "
        "The data and analysis provided should not be considered as financial advice. "
        "Past performance is not indicative of future results. "
        "Always conduct your own research or consult with a financial advisor before making investment decisions."
    )
