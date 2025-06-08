import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import os
import requests
import time
from newsapi import NewsApiClient

# Define a cache decorator for API calls with rate limiting
def rate_limited_cache(ttl=3600, max_calls=5, period=60):
    """
    Cache decorator with rate limiting to prevent API abuse
    - ttl: Time to live for cached data in seconds
    - max_calls: Maximum number of calls allowed in the period
    - period: Time period in seconds
    """
    def decorator(func):
        cache = {}
        call_times = []
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            current_time = time.time()
            
            # Clean up old call times
            call_times[:] = [t for t in call_times if current_time - t < period]
            
            # Check if we're within rate limits
            if len(call_times) >= max_calls:
                oldest_call = min(call_times)
                sleep_time = period - (current_time - oldest_call)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Check cache validity
            if key in cache and current_time - cache[key]["timestamp"] < ttl:
                return cache[key]["data"]
            
            # Make the actual call
            call_times.append(time.time())
            result = func(*args, **kwargs)
            
            # Update cache
            cache[key] = {
                "data": result,
                "timestamp": time.time()
            }
            
            return result
        
        return wrapper
    
    return decorator

@st.cache_data(ttl=3600)
def get_stock_data(symbol, period="1mo", interval="1d"):
    """
    Fetch stock data for a given symbol
    
    Parameters:
    - symbol: Stock ticker symbol
    - period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    - interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
    - DataFrame with stock data
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None
            
        # Calculate additional metrics
        if len(data) > 1:
            # Calculate daily returns
            data['Daily_Return'] = data['Close'].pct_change() * 100
            
            # Calculate moving averages
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate Bollinger Bands
            data['MA20_std'] = data['Close'].rolling(window=20).std()
            data['Upper_Band'] = data['MA20'] + (data['MA20_std'] * 2)
            data['Lower_Band'] = data['MA20'] - (data['MA20_std'] * 2)
            
            # Calculate MACD
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

@st.cache_data(ttl=7200)
def get_sector_symbols(sector_name):
    """
    Get list of symbols for a given sector
    
    Parameters:
    - sector_name: Name of the sector
    
    Returns:
    - List of stock symbols
    """
    # Predefined sector mappings
    sectors = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "ADBE", "CRM", "INTC", "CSCO"],
        "Finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP", "V", "MA"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "BMY", "ABT", "TMO", "MDT"],
        "Consumer": ["PG", "KO", "PEP", "WMT", "MCD", "COST", "NKE", "SBUX", "DIS", "HD"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "MPC", "KMI"],
        "Industrials": ["GE", "BA", "CAT", "HON", "MMM", "UPS", "LMT", "RTX", "DE", "EMR"],
    }
    
    return sectors.get(sector_name, [])

@rate_limited_cache(ttl=86400, max_calls=10, period=60)
def get_company_info(symbol):
    """
    Get company information for a given symbol
    
    Parameters:
    - symbol: Stock ticker symbol
    
    Returns:
    - Dictionary with company information
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Extract relevant information
        company_info = {
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "website": info.get("website", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "description": info.get("longBusinessSummary", "N/A")
        }
        
        return company_info
    except Exception as e:
        st.error(f"Error fetching company info for {symbol}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_market_news(limit=5):
    """
    Get latest market news using NewsAPI
    
    Parameters:
    - limit: Number of news items to return
    
    Returns:
    - List of news items
    """
    try:
        # Get API key from environment or session state
        api_key = os.getenv("NEWS_API_KEY", "")
        if not api_key and "news_api_key" in st.session_state:
            api_key = st.session_state.news_api_key
        
        if not api_key:
            # If no API key is available, provide sample news
            st.warning("News API key is not set. Please go to Settings page to set your API key.")
            sample_news = [
                {
                    "title": "Please add a News API key in Settings to get real market news",
                    "url": "https://newsapi.org",
                    "publishedAt": datetime.now().isoformat(),
                    "source": {"name": "NewsAPI"},
                    "description": "This is a placeholder. Go to Settings to add your News API key."
                }
            ]
            return sample_news
        
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=api_key)
        
        # Get top business headlines
        top_headlines = newsapi.get_top_headlines(
            category='business',
            language='en',
            page_size=limit
        )
        
        # Format news data
        news_items = []
        for article in top_headlines.get('articles', []):
            news_items.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', {}),
                'urlToImage': article.get('urlToImage', '')
            })
        
        return news_items
    except Exception as e:
        st.error(f"Error fetching market news: {e}")
        return []

@st.cache_data(ttl=86400)
def get_market_indices():
    """
    Get data for major market indices
    
    Returns:
    - DataFrame with index data
    """
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "Russell 2000": "^RUT",
        "VIX": "^VIX"
    }
    
    data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    for name, ticker in indices.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                # Extract values properly, avoiding Series to float conversion warnings
                last_price = float(df['Close'].iloc[-1].iloc[0]) if isinstance(df['Close'].iloc[-1], pd.Series) else float(df['Close'].iloc[-1])
                
                prev_price_value = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[0]
                prev_price = float(prev_price_value.iloc[0]) if isinstance(prev_price_value, pd.Series) else float(prev_price_value)
                daily_change = ((last_price - prev_price) / prev_price * 100)
                
                start_price_value = df['Close'].iloc[0]
                start_price = float(start_price_value.iloc[0]) if isinstance(start_price_value, pd.Series) else float(start_price_value)
                period_change = ((last_price - start_price) / start_price * 100)
                
                # Handle volume
                volume = None
                if 'Volume' in df:
                    volume_value = df['Volume'].iloc[-1]
                    volume = float(volume_value.iloc[0]) if isinstance(volume_value, pd.Series) else float(volume_value)
                
                data[name] = {
                    "price": last_price,
                    "daily_change": daily_change,
                    "weekly_change": period_change,
                    "volume": volume
                }
            else:
                data[name] = None
        except Exception as e:
            st.error(f"Error fetching data for {name}: {e}")
            data[name] = None
    
    return data
