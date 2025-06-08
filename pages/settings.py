import streamlit as st
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Settings | Stock Market Analysis Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("FINANCIAL_API_KEY", "")
if 'news_api_key' not in st.session_state:
    st.session_state.news_api_key = os.getenv("NEWS_API_KEY", "")
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Page header
st.title("⚙️ Settings")
st.markdown("Configure your stock market analysis platform")

# Create tabs for different settings
tabs = st.tabs(["API Keys", "Appearance", "About"])

with tabs[0]:  # API Keys tab
    st.subheader("API Key Management")
    
    # Information about API keys
    st.info(
        "This application uses several financial data APIs to provide market data and analysis. "
        "Some features may require API keys for full functionality."
    )
    
    # Financial API key
    st.markdown("### Financial Data API")
    financial_api_key = st.text_input(
        "Enter your Financial API Key",
        value=st.session_state.api_key,
        type="password",
        help="This key is used for accessing additional financial data beyond what's provided by yfinance"
    )
    
    # News API key
    st.markdown("### News API")
    news_api_key = st.text_input(
        "Enter your News API Key",
        value=st.session_state.news_api_key,
        type="password",
        help="This key is used for fetching market news for sentiment analysis. Get your free key at https://newsapi.org"
    )
    
    # Save both API keys
    if st.button("Save API Keys"):
        st.session_state.api_key = financial_api_key
        st.session_state.news_api_key = news_api_key
        st.success("API keys saved successfully!")
    
    # Information about free APIs
    st.markdown("### Free Data Sources")
    st.markdown(
        "This platform primarily uses **yfinance** which provides free access to Yahoo Finance data. "
        "No API key is required for basic functionality, but rate limits may apply."
    )
    
    # Links to API providers
    st.markdown("### Additional API Providers")
    st.markdown(
        """
        For more comprehensive data, consider obtaining API keys from these providers:
        
        - [News API](https://newsapi.org/) - News data for market sentiment analysis
        - [Alpha Vantage](https://www.alphavantage.co/) - Stock data, technical indicators, forex
        - [Financial Modeling Prep](https://financialmodelingprep.com/) - Financial statements, ratios, real-time data
        - [IEX Cloud](https://iexcloud.io/) - Comprehensive financial data platform
        """
    )

with tabs[1]:  # Appearance tab
    st.subheader("Appearance Settings")
    
    # Theme selection
    st.markdown("### Theme")
    selected_theme = st.radio(
        "Select Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "light" else 1
    )
    
    # Save theme setting
    if selected_theme.lower() != st.session_state.theme:
        st.session_state.theme = selected_theme.lower()
        st.success(f"Theme changed to {selected_theme}. Some changes may require a page reload.")
    
    # Chart color preferences
    st.markdown("### Chart Colors")
    st.markdown("Default colors used in charts and visualizations:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.color_picker("Primary Color", "#2962FF", disabled=True)
    with col2:
        st.color_picker("Positive/Green", "#16A34A", disabled=True)
    with col3:
        st.color_picker("Negative/Red", "#DC2626", disabled=True)
    
    # Note about customization
    st.info(
        "Note: Full theme customization is available by modifying the `.streamlit/config.toml` file directly. "
        "This interface provides basic theme switching."
    )

with tabs[2]:  # About tab
    st.subheader("About the Stock Market Analysis Platform")
    
    st.markdown(
        """
        This platform provides tools for stock market analysis, with a focus on:
        
        - **Sector-wise Momentum Analysis**: Track and compare performance across market sectors
        - **Central Pivot Range (CPR)**: Visualize key support and resistance levels using CPR technique
        - **Sentiment Analysis**: Gauge market sentiment from news and media sources
        
        ### Technologies Used
        
        - **Streamlit**: Web application framework
        - **Pandas**: Data manipulation and analysis
        - **Plotly**: Interactive data visualizations
        - **yfinance**: Market data access
        - **NLTK**: Natural language processing for sentiment analysis
        
        ### Disclaimer
        
        This application is for educational and informational purposes only. The data and analysis 
        provided should not be considered as financial advice. Past performance is not indicative of 
        future results. Always conduct your own research or consult with a financial advisor before 
        making investment decisions.
        
        ### Credits
        
        Developed using open-source tools and libraries including Streamlit, Pandas, Plotly, yfinance, and NLTK.
        """
    )
    
    # Version information
    st.markdown("### Version Information")
    st.markdown("Stock Market Analysis Platform v1.0.0")
    
    # Contact information
    st.markdown("### Contact & Support")
    st.markdown(
        "For questions, suggestions, or support requests, please contact the developer at example@email.com"
    )

# Footer
st.markdown("---")
st.markdown("Powered by yfinance, Streamlit, and Plotly | Data for educational purposes only")
