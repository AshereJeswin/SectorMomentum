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
from utils.data_fetcher import get_sector_symbols, get_stock_data, get_market_news
from utils.sentiment_analyzer import analyze_news_sentiment, analyze_sector_sentiment, get_sentiment_color

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis | Stock Market Analysis Platform",
    page_icon="📰",
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
st.title("📰 Sentiment Analysis")
st.markdown("Analyze market sentiment through news and social media")

# Latest Market Headlines
st.subheader("Latest Market Headlines")

# Fetch market news
market_news = get_market_news(limit=5)

if market_news:
    # Display news in columns
    cols = st.columns(len(market_news))
    
    for i, news in enumerate(cols):
        with news:
            # Format date
            if 'providerPublishTime' in market_news[i]:
                published_time = datetime.fromtimestamp(market_news[i]['providerPublishTime']).strftime('%Y-%m-%d %H:%M')
            else:
                published_time = "Unknown date"
            
            # Display news card
            st.markdown(f"##### {market_news[i].get('title', 'No title')}")
            st.caption(f"{published_time} | {market_news[i].get('publisher', 'Unknown source')}")
            
            # Add link to full article
            if 'link' in market_news[i]:
                st.markdown(f"[Read full article]({market_news[i]['link']})")
else:
    st.info("No market news available")

# Divider
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Sentiment Analysis Controls")
    
    # Sector selection
    selected_sector = st.selectbox(
        "Select Sector",
        options=sectors,
        index=sectors.index(st.session_state.selected_sector) if st.session_state.selected_sector in sectors else 0,
        key="sentiment_sector_selector"
    )
    st.session_state.selected_sector = selected_sector
    
    # Time period for analysis
    time_period = st.selectbox(
        "Analysis Timeframe",
        options=["Last 24 hours", "Last 3 days", "Last week", "Last 2 weeks", "Last month"],
        index=2
    )
    
    # Map selection to days
    days_mapping = {
        "Last 24 hours": 1,
        "Last 3 days": 3,
        "Last week": 7,
        "Last 2 weeks": 14,
        "Last month": 30
    }
    analysis_days = days_mapping[time_period]
    
    # Analysis type
    analysis_type = st.radio(
        "Analysis Type",
        options=["Sector Overview", "Individual Symbol"],
        index=0
    )

# Get symbols for the selected sector
sector_symbols = get_sector_symbols(selected_sector)

if not sector_symbols:
    st.error(f"No symbols available for {selected_sector} sector")
else:
    # Perform sector sentiment analysis or individual symbol analysis
    if analysis_type == "Sector Overview":
        st.subheader(f"{selected_sector} Sector Sentiment Overview")
        
        # Create a progress bar
        progress_text = f"Analyzing sentiment for {len(sector_symbols)} symbols in {selected_sector} sector..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Analyze sentiment for all symbols in the sector
        sector_sentiment = {}
        for i, symbol in enumerate(sector_symbols):
            # Update progress
            progress_percentage = (i + 1) / len(sector_symbols)
            progress_bar.progress(progress_percentage, text=f"Analyzing {symbol}... ({i+1}/{len(sector_symbols)})")
            
            # Get news and analyze sentiment
            news_df = analyze_news_sentiment(symbol, analysis_days)
            
            if news_df is not None and not news_df.empty:
                # Calculate sentiment summary
                positive_count = (news_df['Sentiment'] == 'Positive').sum()
                neutral_count = (news_df['Sentiment'] == 'Neutral').sum()
                negative_count = (news_df['Sentiment'] == 'Negative').sum()
                total_count = len(news_df)
                
                positive_pct = (positive_count / total_count * 100) if total_count > 0 else 0
                neutral_pct = (neutral_count / total_count * 100) if total_count > 0 else 0
                negative_pct = (negative_count / total_count * 100) if total_count > 0 else 0
                
                avg_compound = news_df['Compound'].mean() if total_count > 0 else 0
                
                # Determine overall sentiment
                if positive_pct > 60:
                    overall_sentiment = 'Very Positive'
                elif positive_pct > 40:
                    overall_sentiment = 'Positive'
                elif negative_pct > 60:
                    overall_sentiment = 'Very Negative'
                elif negative_pct > 40:
                    overall_sentiment = 'Negative'
                else:
                    overall_sentiment = 'Neutral'
                
                # Save sentiment data
                sector_sentiment[symbol] = {
                    'total_news': total_count,
                    'positive_pct': positive_pct,
                    'neutral_pct': neutral_pct,
                    'negative_pct': negative_pct,
                    'avg_compound': avg_compound,
                    'overall_sentiment': overall_sentiment
                }
            else:
                # No news available
                sector_sentiment[symbol] = {
                    'total_news': 0,
                    'positive_pct': 0,
                    'neutral_pct': 0,
                    'negative_pct': 0,
                    'avg_compound': 0,
                    'overall_sentiment': 'No Data'
                }
        
        # Clear progress bar
        progress_bar.empty()
        
        # Create dataframe from sector sentiment
        sentiment_data = []
        for symbol, data in sector_sentiment.items():
            sentiment_data.append({
                'Symbol': symbol,
                'News Count': data['total_news'],
                'Overall Sentiment': data['overall_sentiment'],
                'Positive (%)': data['positive_pct'],
                'Neutral (%)': data['neutral_pct'],
                'Negative (%)': data['negative_pct'],
                'Compound Score': data['avg_compound']
            })
        
        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data)
            
            # Sort by compound score (descending)
            sentiment_df = sentiment_df.sort_values('Compound Score', ascending=False)
            
            # Display sector sentiment overview
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Display sentiment table
                st.dataframe(sentiment_df, use_container_width=True)
            
            with col2:
                # Create a horizontal bar chart of compound scores
                fig = px.bar(
                    sentiment_df, 
                    y='Symbol', 
                    x='Compound Score',
                    color='Compound Score',
                    color_continuous_scale=["#DC2626", "#FFFFFF", "#16A34A"],
                    title="Sentiment Score by Symbol",
                    height=500
                )
                
                fig.update_layout(
                    xaxis_title="Compound Sentiment Score",
                    yaxis_title="Symbol",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create sentiment distribution chart
            st.subheader("Sentiment Distribution by Symbol")
            
            # Prepare data for stacked bar chart
            chart_data = sentiment_df[['Symbol', 'Positive (%)', 'Neutral (%)', 'Negative (%)']].copy()
            
            # Melt the dataframe for easier plotting
            chart_data_melted = pd.melt(
                chart_data,
                id_vars=['Symbol'],
                value_vars=['Positive (%)', 'Neutral (%)', 'Negative (%)'],
                var_name='Sentiment',
                value_name='Percentage'
            )
            
            # Create stacked bar chart
            fig = px.bar(
                chart_data_melted,
                x='Symbol',
                y='Percentage',
                color='Sentiment',
                color_discrete_map={
                    'Positive (%)': '#16A34A',
                    'Neutral (%)': '#64748B',
                    'Negative (%)': '#DC2626'
                },
                title="Sentiment Distribution by Symbol",
                height=400
            )
            
            fig.update_layout(
                xaxis_title="Symbol",
                yaxis_title="Percentage (%)",
                template="plotly_white",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sector sentiment summary
            st.subheader("Sector Sentiment Summary")
            
            # Calculate overall sector sentiment
            sector_avg_compound = sentiment_df['Compound Score'].mean()
            sector_positive_avg = sentiment_df['Positive (%)'].mean()
            sector_neutral_avg = sentiment_df['Neutral (%)'].mean()
            sector_negative_avg = sentiment_df['Negative (%)'].mean()
            
            # Determine sector sentiment
            if sector_avg_compound > 0.25:
                sector_sentiment_label = "Very Positive"
                sector_sentiment_color = "#16A34A"
            elif sector_avg_compound > 0.05:
                sector_sentiment_label = "Positive"
                sector_sentiment_color = "#4ADE80"
            elif sector_avg_compound < -0.25:
                sector_sentiment_label = "Very Negative"
                sector_sentiment_color = "#DC2626"
            elif sector_avg_compound < -0.05:
                sector_sentiment_label = "Negative"
                sector_sentiment_color = "#EF4444"
            else:
                sector_sentiment_label = "Neutral"
                sector_sentiment_color = "#64748B"
            
            # Display sector sentiment metrics
            cols = st.columns(4)
            with cols[0]:
                st.metric(
                    "Overall Sector Sentiment",
                    sector_sentiment_label,
                    delta=f"{sector_avg_compound:.2f} Score"
                )
            with cols[1]:
                st.metric(
                    "Positive Sentiment",
                    f"{sector_positive_avg:.1f}%"
                )
            with cols[2]:
                st.metric(
                    "Neutral Sentiment",
                    f"{sector_neutral_avg:.1f}%"
                )
            with cols[3]:
                st.metric(
                    "Negative Sentiment",
                    f"{sector_negative_avg:.1f}%"
                )
            
            # Display most positive and negative stocks
            st.subheader("Sentiment Extremes")
            
            # Get most positive and negative stocks
            try:
                most_positive = sentiment_df.iloc[0]
                most_negative = sentiment_df.iloc[-1]
                
                pos_neg_cols = st.columns(2)
                
                with pos_neg_cols[0]:
                    st.success(f"Most Positive: **{most_positive['Symbol']}** ({most_positive['Overall Sentiment']})")
                    st.markdown(f"Compound Score: {most_positive['Compound Score']:.2f}")
                    st.markdown(f"Positive News: {most_positive['Positive (%)']:.1f}%")
                
                with pos_neg_cols[1]:
                    st.error(f"Most Negative: **{most_negative['Symbol']}** ({most_negative['Overall Sentiment']})")
                    st.markdown(f"Compound Score: {most_negative['Compound Score']:.2f}")
                    st.markdown(f"Negative News: {most_negative['Negative (%)']:.1f}%")
            except:
                st.info("Not enough data to determine sentiment extremes")
            
            # Show symbols with no news data
            no_news_symbols = sentiment_df[sentiment_df['News Count'] == 0]['Symbol'].tolist()
            if no_news_symbols:
                st.subheader("Symbols with No News Data")
                st.markdown(', '.join(no_news_symbols))
        else:
            st.warning("No sentiment data available for symbols in this sector")
    
    else:  # Individual Symbol Analysis
        # Symbol selection
        selected_symbol = st.selectbox(
            "Select Symbol for Sentiment Analysis",
            options=sector_symbols,
            index=sector_symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in sector_symbols else 0
        )
        st.session_state.selected_symbol = selected_symbol
        
        st.subheader(f"Sentiment Analysis for {selected_symbol}")
        
        # Get stock data for context
        stock_data = get_stock_data(selected_symbol, period="1mo", interval="1d")
        
        if stock_data is not None and not stock_data.empty:
            # Display price chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#2962FF', width=2)
                )
            )
            
            fig.update_layout(
                title=f"{selected_symbol} - Price Chart (Last Month)",
                xaxis_title="Date",
                yaxis_title="Price",
                height=300,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Analyze news sentiment for the selected symbol
        news_df = analyze_news_sentiment(selected_symbol, analysis_days)
        
        if news_df is not None and not news_df.empty:
            # Display sentiment summary
            positive_count = (news_df['Sentiment'] == 'Positive').sum()
            neutral_count = (news_df['Sentiment'] == 'Neutral').sum()
            negative_count = (news_df['Sentiment'] == 'Negative').sum()
            total_count = len(news_df)
            
            positive_pct = (positive_count / total_count * 100) if total_count > 0 else 0
            neutral_pct = (neutral_count / total_count * 100) if total_count > 0 else 0
            negative_pct = (negative_count / total_count * 100) if total_count > 0 else 0
            
            avg_compound = news_df['Compound'].mean() if total_count > 0 else 0
            
            # Determine overall sentiment
            if positive_pct > 60:
                overall_sentiment = 'Very Positive'
                sentiment_color = "#16A34A"
            elif positive_pct > 40:
                overall_sentiment = 'Positive'
                sentiment_color = "#4ADE80"
            elif negative_pct > 60:
                overall_sentiment = 'Very Negative'
                sentiment_color = "#DC2626"
            elif negative_pct > 40:
                overall_sentiment = 'Negative'
                sentiment_color = "#EF4444"
            else:
                overall_sentiment = 'Neutral'
                sentiment_color = "#64748B"
            
            # Display sentiment summary
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric(
                    "Overall Sentiment",
                    overall_sentiment,
                    delta=f"{avg_compound:.2f} Score"
                )
            with summary_cols[1]:
                st.metric(
                    "Positive News",
                    f"{positive_pct:.1f}%",
                    delta=f"{positive_count} articles"
                )
            with summary_cols[2]:
                st.metric(
                    "Neutral News",
                    f"{neutral_pct:.1f}%",
                    delta=f"{neutral_count} articles"
                )
            with summary_cols[3]:
                st.metric(
                    "Negative News",
                    f"{negative_pct:.1f}%",
                    delta=f"{negative_count} articles"
                )
            
            # Create sentiment distribution pie chart
            data_cols = st.columns([2, 3])
            
            with data_cols[0]:
                # Create pie chart
                fig = go.Figure(
                    go.Pie(
                        labels=['Positive', 'Neutral', 'Negative'],
                        values=[positive_pct, neutral_pct, negative_pct],
                        hole=0.5,
                        marker=dict(colors=['#16A34A', '#64748B', '#DC2626'])
                    )
                )
                
                fig.update_layout(
                    title=f"Sentiment Distribution",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with data_cols[1]:
                # Display sentiment over time
                # Convert date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(news_df['Date']):
                    news_df['Date'] = pd.to_datetime(news_df['Date'])
                
                # Sort by date
                news_df = news_df.sort_values('Date')
                
                # Create line chart for sentiment over time
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=news_df['Date'],
                        y=news_df['Compound'],
                        mode='lines+markers',
                        name='Sentiment Score',
                        line=dict(color='#2962FF', width=2)
                    )
                )
                
                # Add reference lines
                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
                fig.add_hline(y=0.05, line_width=1, line_dash="dot", line_color="#16A34A")
                fig.add_hline(y=-0.05, line_width=1, line_dash="dot", line_color="#DC2626")
                
                fig.update_layout(
                    title=f"Sentiment Score Over Time",
                    xaxis_title="Date",
                    yaxis_title="Compound Score",
                    height=300,
                    template="plotly_white",
                    yaxis=dict(range=[-1, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display recent news headlines with sentiment
            st.subheader("Recent News Headlines")
            
            # Sort by date (newest first)
            news_df = news_df.sort_values('Date', ascending=False)
            
            # Display each news item with sentiment
            for _, row in news_df.iterrows():
                # Determine sentiment color
                if row['Sentiment'] == 'Positive':
                    card_color = "#16A34A"
                elif row['Sentiment'] == 'Negative':
                    card_color = "#DC2626"
                else:
                    card_color = "#64748B"
                
                # Create a card-like display for each news item
                st.markdown(
                    f"""
                    <div style="padding: 10px; border-left: 5px solid {card_color}; margin-bottom: 10px;">
                        <h5>{row['Title']}</h5>
                        <p style="color: gray; margin-bottom: 5px;">{row['Date']} | {row['Publisher']}</p>
                        <p>Sentiment: <span style="color: {card_color}; font-weight: bold;">{row['Sentiment']}</span> (Score: {row['Compound']:.2f})</p>
                        <a href="{row['Link']}" target="_blank">Read article</a>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            # Option to download the sentiment data
            csv = news_df.to_csv(index=False)
            st.download_button(
                label="Download Sentiment Data",
                data=csv,
                file_name=f"{selected_symbol}_sentiment_data.csv",
                mime="text/csv"
            )
        else:
            st.warning(f"No news articles found for {selected_symbol} in the selected time period")

# Educational section
with st.expander("About Sentiment Analysis"):
    st.markdown("""
    ### What is Sentiment Analysis?
    
    Sentiment analysis is the process of determining the emotional tone behind a series of words, used to gain an understanding of attitudes, opinions, and emotions expressed in text data. In financial markets, sentiment analysis can help gauge market perception of stocks, sectors, or the overall market.
    
    ### How It Works
    
    This platform uses Natural Language Processing (NLP) techniques to analyze news headlines related to stocks and sectors. The analysis process involves:
    
    1. **Collection**: Gathering news articles and headlines from financial news sources.
    2. **Processing**: Cleaning and preparing text data for analysis.
    3. **Scoring**: Using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to assign sentiment scores.
    4. **Aggregation**: Combining scores to determine overall sentiment.
    
    ### Interpreting Sentiment Scores
    
    - **Compound Score**: A normalized score between -1 (very negative) and +1 (very positive).
    - **Positive/Neutral/Negative Percentages**: Distribution of sentiment across analyzed news.
    - **Overall Sentiment**: A categorical label based on the compound score and distribution.
    
    ### Using Sentiment in Trading
    
    Sentiment analysis can be a valuable tool for traders and investors:
    
    - **Market Timing**: Extreme sentiment (either positive or negative) often precedes market reversals.
    - **Stock Selection**: Stocks with improving sentiment may outperform their peers.
    - **Risk Management**: Deteriorating sentiment may signal upcoming volatility or price declines.
    
    ### Limitations
    
    Sentiment analysis is not perfect and should be used alongside other forms of analysis:
    
    - News headlines may not capture the full context.
    - Automated analysis may miss nuance, sarcasm, or industry-specific terminology.
    - The relationship between sentiment and price movement isn't always direct or immediate.
    
    Always combine sentiment analysis with fundamental and technical analysis for a more complete investment approach.
    """)

# Footer
st.markdown("---")
st.markdown("Powered by yfinance, NLTK, Streamlit, and Plotly | Data for educational purposes only")

# Disclaimer
with st.expander("Disclaimer"):
    st.warning(
        "This application is for educational and informational purposes only. "
        "The data and analysis provided should not be considered as financial advice. "
        "Past performance is not indicative of future results. "
        "Always conduct your own research or consult with a financial advisor before making investment decisions."
    )
