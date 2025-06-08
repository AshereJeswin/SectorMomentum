import streamlit as st
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Stock Market Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("FINANCIAL_API_KEY", "")
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = "Technology"
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "AAPL"

# Define sector-symbol mapping
sectors = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "Finance": ["JPM", "BAC", "GS", "MS", "WFC"],
    "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV"],
    "Consumer": ["PG", "KO", "PEP", "WMT", "MCD"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Industrials": ["GE", "BA", "CAT", "HON", "MMM"],
}

# Function to fetch stock data for the main dashboard
@st.cache_data(ttl=3600)
def fetch_market_overview():
    # Fetch data for major indices
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "Russell 2000": "^RUT"
    }
    
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    for name, ticker in indices.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                # Get scalar values using .iloc[0] to avoid deprecation warnings
                last_price = float(df['Close'].iloc[-1].iloc[0]) if isinstance(df['Close'].iloc[-1], pd.Series) else float(df['Close'].iloc[-1])
                prev_price = float(df['Close'].iloc[-2].iloc[0]) if len(df) > 1 and isinstance(df['Close'].iloc[-2], pd.Series) else float(df['Close'].iloc[-2]) if len(df) > 1 else None
                change = ((last_price - prev_price) / prev_price * 100) if prev_price else None
                
                # Extract volume as a scalar value
                volume = None
                if 'Volume' in df:
                    volume = float(df['Volume'].iloc[-1].iloc[0]) if isinstance(df['Volume'].iloc[-1], pd.Series) else float(df['Volume'].iloc[-1])
                    
                data[name] = {
                    "price": last_price,
                    "change": change,
                    "volume": volume
                }
            else:
                data[name] = {"price": None, "change": None, "volume": None}
        except Exception as e:
            st.error(f"Error fetching data for {name}: {e}")
            data[name] = {"price": None, "change": None, "volume": None}
    
    return data

# Main page header
st.title("📊 Stock Market Analysis Platform")

# Market overview section
st.subheader("Market Overview")
market_data = fetch_market_overview()

# Create a layout with multiple columns for market overview
cols = st.columns(len(market_data))
for i, (index_name, data) in enumerate(market_data.items()):
    with cols[i]:
        st.metric(
            label=index_name,
            value=f"${data['price']:.2f}" if data['price'] else "N/A",
            delta=f"{data['change']:.2f}%" if data['change'] else "N/A"
        )

# Main dashboard section
st.markdown("---")
st.subheader("Sector Performance")

# Selector for sector
selected_sector = st.selectbox(
    "Select Sector",
    options=list(sectors.keys()),
    key="sector_selector"
)
st.session_state.selected_sector = selected_sector

# Fetch data for selected sector symbols
@st.cache_data(ttl=3600)
def fetch_sector_data(symbols, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = {}
    
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if not df.empty:
                data[symbol] = df
            else:
                st.warning(f"No data available for {symbol}")
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
    
    return data

sector_symbols = sectors[selected_sector]
sector_data = fetch_sector_data(sector_symbols)

# Display sector data in a table
if sector_data:
    # Create a DataFrame for sector performance
    performance_data = []
    for symbol, df in sector_data.items():
        if not df.empty:
            # Get values and convert properly to avoid deprecation warnings
            current_price = float(df['Close'].iloc[-1].iloc[0]) if isinstance(df['Close'].iloc[-1], pd.Series) else float(df['Close'].iloc[-1])
            prev_close_value = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[0]
            prev_close = float(prev_close_value.iloc[0]) if isinstance(prev_close_value, pd.Series) else float(prev_close_value)
            daily_change = ((current_price - prev_close) / prev_close * 100)
            
            start_price_value = df['Close'].iloc[0]
            start_price = float(start_price_value.iloc[0]) if isinstance(start_price_value, pd.Series) else float(start_price_value)
            period_change = ((current_price - start_price) / start_price * 100)
            
            avg_volume_value = df['Volume'].mean() if 'Volume' in df else 0
            avg_volume = float(avg_volume_value) if avg_volume_value is not None else 0
                
            performance_data.append({
                "Symbol": symbol,
                "Current Price": f"${current_price:.2f}",
                "Daily Change": f"{daily_change:.2f}%",
                "30-Day Change": f"{period_change:.2f}%",
                "Avg Volume": f"{int(avg_volume):,}" if avg_volume else "N/A"
            })
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
    else:
        st.warning("No performance data available for the selected sector.")
else:
    st.warning("No data available for the selected sector.")

# Quick links to other pages
st.markdown("---")
st.subheader("Detailed Analysis Tools")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📊 [Sector Analysis](Sector_Analysis)")
with col2:
    st.info("📏 [CPR Visualization](CPR_Visualization)")
with col3:
    st.info("📰 [Sentiment Analysis](Sentiment_Analysis)")

# Footer
st.markdown("---")
st.markdown("Powered by yfinance, Streamlit, and NLTK | Data for educational purposes only")

# Disclaimer
with st.expander("Disclaimer"):
    st.warning(
        "This application is for educational and informational purposes only. "
        "The data and analysis provided should not be considered as financial advice. "
        "Past performance is not indicative of future results. "
        "Always conduct your own research or consult with a financial advisor before making investment decisions."
    )
