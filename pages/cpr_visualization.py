import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_fetcher import get_stock_data, get_sector_symbols, get_company_info
from utils.cpr_calculator import calculate_cpr, plot_cpr, get_cpr_analysis

# Page configuration
st.set_page_config(
    page_title="CPR Visualization | Stock Market Analysis Platform",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = "Technology"
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "AAPL"

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
st.title("📏 Central Pivot Range (CPR) Visualization")
st.markdown("Analyze stock price action using Central Pivot Range (CPR) technique")

# Sidebar for controls
with st.sidebar:
    st.header("CPR Analysis Controls")
    
    # Sector selection
    selected_sector = st.selectbox(
        "Select Sector",
        options=sectors,
        index=sectors.index(st.session_state.selected_sector) if st.session_state.selected_sector in sectors else 0,
        key="cpr_sector_selector"
    )
    st.session_state.selected_sector = selected_sector
    
    # Get symbols for the selected sector
    sector_symbols = get_sector_symbols(selected_sector)
    
    # Symbol selection
    if sector_symbols:
        default_idx = sector_symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in sector_symbols else 0
        selected_symbol = st.selectbox(
            "Select Symbol",
            options=sector_symbols,
            index=default_idx,
            key="cpr_symbol_selector"
        )
        st.session_state.selected_symbol = selected_symbol
    else:
        st.error(f"No symbols available for {selected_sector} sector")
        selected_symbol = None
    
    # Time period selection
    time_period = st.selectbox(
        "Time Period",
        options=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        index=1,
        key="cpr_time_period"
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
    
    # CPR Display options
    st.subheader("Display Options")
    
    show_pivot = st.checkbox("Show Pivot Point (PP)", value=True)
    show_tc_bc = st.checkbox("Show TC and BC", value=True)
    show_r_levels = st.checkbox("Show Resistance Levels", value=True)
    show_s_levels = st.checkbox("Show Support Levels", value=True)
    show_analysis = st.checkbox("Show CPR Analysis", value=True)

# Main content
if selected_symbol:
    # Fetch stock data
    df = get_stock_data(selected_symbol, period=yf_period)
    
    if df is not None and not df.empty:
        # Fetch company info
        company_info = get_company_info(selected_symbol)
        
        # Display company name if available
        if company_info and 'name' in company_info:
            st.subheader(f"{company_info['name']} ({selected_symbol})")
            
            # Create two columns for company info and latest price
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                if 'sector' in company_info and 'industry' in company_info:
                    st.markdown(f"**Sector:** {company_info['sector']} | **Industry:** {company_info['industry']}")
                
                if 'market_cap' in company_info and company_info['market_cap'] != 'N/A':
                    market_cap = company_info['market_cap']
                    if isinstance(market_cap, (int, float)):
                        if market_cap >= 1e12:
                            market_cap_str = f"${market_cap/1e12:.2f}T"
                        elif market_cap >= 1e9:
                            market_cap_str = f"${market_cap/1e9:.2f}B"
                        elif market_cap >= 1e6:
                            market_cap_str = f"${market_cap/1e6:.2f}M"
                        else:
                            market_cap_str = f"${market_cap:,.0f}"
                        st.markdown(f"**Market Cap:** {market_cap_str}")
            
            with info_col2:
                # Display latest price information
                latest_price = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[0]
                price_change = latest_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
                
                price_color = "#16A34A" if price_change >= 0 else "#DC2626"
                
                st.markdown(f"**Latest Price:** <span style='color:{price_color};font-weight:bold;'>${latest_price:.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Change:** <span style='color:{price_color};font-weight:bold;'>{price_change:+.2f} ({price_change_pct:+.2f}%)</span>", unsafe_allow_html=True)
        else:
            st.subheader(f"{selected_symbol}")
        
        # Calculate CPR
        df_cpr = calculate_cpr(df)
        
        if df_cpr is not None:
            # Create tabs for different views
            tabs = st.tabs(["CPR Chart", "Raw Data", "Analysis"])
            
            with tabs[0]:  # CPR Chart tab
                # Plot CPR
                fig = plot_cpr(df_cpr, selected_symbol, time_period)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to create CPR chart")
            
            with tabs[1]:  # Raw Data tab
                st.subheader("CPR Data Table")
                
                # Select columns to display
                display_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                if show_pivot:
                    display_columns.append('PP')
                
                if show_tc_bc:
                    display_columns.extend(['TC', 'BC'])
                
                if show_r_levels:
                    display_columns.extend(['R1', 'R2', 'R3'])
                
                if show_s_levels:
                    display_columns.extend(['S1', 'S2', 'S3'])
                
                # Format the dataframe for display
                display_df = df_cpr[display_columns].copy()
                
                # Convert date index to column
                display_df = display_df.reset_index()
                display_df = display_df.rename(columns={'index': 'Date'})
                
                # Format date column
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Display the dataframe
                st.dataframe(display_df, use_container_width=True)
                
                # Option to download the data
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{selected_symbol}_CPR_data.csv",
                    mime="text/csv"
                )
            
            with tabs[2]:  # Analysis tab
                if show_analysis:
                    st.subheader("CPR Analysis")
                    
                    # Get CPR analysis text
                    analysis_text = get_cpr_analysis(df_cpr)
                    
                    st.markdown(analysis_text)
                    
                    # Latest CPR values
                    latest = df_cpr.iloc[-1]
                    
                    # Display current values in a grid
                    st.subheader("Current CPR Values")
                    
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
                    
                    # CPR width trend
                    st.subheader("CPR Width Trend")
                    
                    # Calculate the CPR width for all days
                    df_cpr['CPR_Width_Pct'] = (df_cpr['CPR_Width'] / df_cpr['Close']) * 100
                    
                    # Create a line chart for CPR width trend
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_cpr.index,
                            y=df_cpr['CPR_Width_Pct'],
                            mode='lines+markers',
                            name='CPR Width (%)',
                            line=dict(color='#2962FF', width=2)
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title="CPR Width Trend (as % of Price)",
                        xaxis_title="Date",
                        yaxis_title="Width (%)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # CPR Width interpretation
                    avg_width = df_cpr['CPR_Width_Pct'].mean()
                    latest_width = df_cpr['CPR_Width_Pct'].iloc[-1]
                    
                    if latest_width > avg_width * 1.5:
                        st.info(f"Current CPR width ({latest_width:.2f}%) is significantly wider than average ({avg_width:.2f}%), suggesting higher volatility")
                    elif latest_width < avg_width * 0.5:
                        st.info(f"Current CPR width ({latest_width:.2f}%) is significantly narrower than average ({avg_width:.2f}%), suggesting lower volatility")
                    else:
                        st.info(f"Current CPR width ({latest_width:.2f}%) is close to average ({avg_width:.2f}%)")
        else:
            st.error("Failed to calculate CPR values")
    else:
        st.error(f"No data available for {selected_symbol}")
else:
    st.warning("Please select a symbol to view CPR analysis")

# CPR Educational Section
with st.expander("What is Central Pivot Range (CPR)?"):
    st.markdown("""
    ### Central Pivot Range (CPR)
    
    Central Pivot Range (CPR) is a technical analysis indicator that helps identify key support and resistance levels. It consists of three main levels:
    
    1. **Pivot Point (PP)**: The arithmetic average of the high, low, and close of the previous period.
    2. **Bottom Central Pivot (BC)**: The average of the high and low of the previous period.
    3. **Top Central Pivot (TC)**: A level derived from PP and BC, calculated as (PP - BC) + PP.
    
    The area between TC and BC is known as the Central Pivot Range.
    
    ### Calculation
    
    The formulas used to calculate CPR levels are:
    
    - **PP (Pivot Point)**: (High + Low + Close) / 3
    - **BC (Bottom Central Pivot)**: (High + Low) / 2
    - **TC (Top Central Pivot)**: (PP - BC) + PP
    
    Additional support and resistance levels can be calculated:
    
    - **R3**: High + 2 * (PP - Low)
    - **R2**: PP + (High - Low)
    - **R1**: 2 * PP - Low
    - **S1**: 2 * PP - High
    - **S2**: PP - (High - Low)
    - **S3**: Low - 2 * (High - PP)
    
    ### Interpretation
    
    - **Price above CPR**: Bullish sentiment. TC acts as first support.
    - **Price below CPR**: Bearish sentiment. BC acts as first resistance.
    - **Price within CPR**: Market is in equilibrium or consolidation. Watch for breakouts.
    
    **CPR Width**:
    - **Wide CPR**: Indicates high volatility.
    - **Narrow CPR**: Indicates low volatility or consolidation.
    
    ### Trading Strategies
    
    1. **Breakout Strategy**: Enter long when price breaks above TC, or short when price breaks below BC.
    2. **Reversal Strategy**: Look for reversals when price tests TC from above or BC from below.
    3. **Range-Bound Strategy**: When price stays within the CPR, trade between TC and BC levels.
    
    CPR is particularly effective in daily charts but can be applied to any timeframe.
    """)

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
