import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import requests
from datetime import datetime, timedelta
import time
import os
from newsapi import NewsApiClient

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize the NLTK Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

@st.cache_data(ttl=3600)
def get_news_for_symbol(symbol, num_days=7):
    """
    Get news articles for a specific symbol using NewsAPI
    
    Parameters:
    - symbol: Stock ticker symbol
    - num_days: Number of days to look back
    
    Returns:
    - List of news articles
    """
    try:
        # Get API key from environment or session state
        api_key = os.getenv("NEWS_API_KEY", "")
        if not api_key and "news_api_key" in st.session_state:
            api_key = st.session_state.news_api_key
        
        if not api_key:
            # If no API key is available, inform the user
            st.warning("News API key is not set. Please go to Settings page to set your API key.")
            sample_news = [{
                'title': f"Add a News API key to get real news for {symbol}",
                'publishedAt': datetime.now().isoformat(),
                'publisher': 'NewsAPI',
                'link': 'https://newsapi.org',
                'source': {'name': 'NewsAPI'}
            }]
            return sample_news
        
        # Initialize NewsAPI client
        newsapi = NewsApiClient(api_key=api_key)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        
        # Get news related to the symbol
        all_articles = newsapi.get_everything(
            q=f"{symbol} stock",
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            language='en',
            sort_by='publishedAt',
            page_size=10  # Limit results
        )
        
        # Format news data
        news_items = []
        for article in all_articles.get('articles', []):
            news_items.append({
                'title': article.get('title', ''),
                'link': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'publisher': article.get('source', {}).get('name', 'Unknown'),
                'description': article.get('description', ''),
                'source': article.get('source', {})
            })
        
        return news_items
    except Exception as e:
        st.error(f"Error fetching news for {symbol}: {e}")
        return []

def analyze_sentiment_for_text(text):
    """
    Analyze sentiment of a given text
    
    Parameters:
    - text: Text to analyze
    
    Returns:
    - Dictionary with sentiment scores
    """
    if not text or not isinstance(text, str):
        return {
            'compound': 0,
            'pos': 0,
            'neu': 0,
            'neg': 0,
            'sentiment': 'Neutral'
        }
        
    # Clean text
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Get sentiment scores
    scores = sia.polarity_scores(text)
    
    # Determine sentiment label
    if scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    scores['sentiment'] = sentiment
    
    return scores

@st.cache_data(ttl=3600)
def analyze_news_sentiment(symbol, num_days=7):
    """
    Analyze sentiment of news articles for a symbol
    
    Parameters:
    - symbol: Stock ticker symbol
    - num_days: Number of days to look back
    
    Returns:
    - DataFrame with news articles and sentiment scores
    """
    news = get_news_for_symbol(symbol, num_days)
    
    if not news:
        return None
        
    # Analyze sentiment for each news article
    news_data = []
    for article in news:
        title = article.get('title', '')
        
        # Handle different date formats from different API sources
        published_date = 'Unknown'
        if 'publishedAt' in article:
            # Format from NewsAPI
            try:
                published_date = article.get('publishedAt', '')
            except:
                published_date = 'Unknown'
        elif 'providerPublishTime' in article:
            # Format from yfinance
            published_time = article.get('providerPublishTime', 0)
            published_date = datetime.fromtimestamp(published_time).strftime('%Y-%m-%d %H:%M:%S') if published_time else 'Unknown'
        
        # Get link and publisher
        link = article.get('link', article.get('url', ''))
        publisher = article.get('publisher', 'Unknown')
        if publisher == 'Unknown' and 'source' in article:
            publisher = article.get('source', {}).get('name', 'Unknown')
        
        # Analyze sentiment of title
        sentiment_scores = analyze_sentiment_for_text(title)
        
        news_data.append({
            'Symbol': symbol,
            'Title': title,
            'Publisher': publisher,
            'Date': published_date,
            'Link': link,
            'Compound': sentiment_scores['compound'],
            'Positive': sentiment_scores['pos'],
            'Neutral': sentiment_scores['neu'],
            'Negative': sentiment_scores['neg'],
            'Sentiment': sentiment_scores['sentiment']
        })
    
    # Create DataFrame
    if news_data:
        df = pd.DataFrame(news_data)
        
        # Convert and sort by date (newest first)
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date', ascending=False)
        except:
            # If date conversion fails, keep original order
            pass
        
        return df
    else:
        return None

@st.cache_data(ttl=3600)
def get_overall_sentiment_summary(df):
    """
    Calculate overall sentiment summary from news DataFrame
    
    Parameters:
    - df: DataFrame with sentiment scores
    
    Returns:
    - Dictionary with sentiment summary
    """
    if df is None or df.empty:
        return {
            'overall_sentiment': 'Neutral',
            'positive_pct': 0,
            'neutral_pct': 0,
            'negative_pct': 0,
            'average_compound': 0,
            'sentiment_trend': 'Stable',
            'count': 0
        }
    
    # Count sentiment categories
    sentiment_counts = df['Sentiment'].value_counts()
    positive_count = sentiment_counts.get('Positive', 0)
    neutral_count = sentiment_counts.get('Neutral', 0)
    negative_count = sentiment_counts.get('Negative', 0)
    total_count = len(df)
    
    # Calculate percentages
    positive_pct = (positive_count / total_count * 100) if total_count > 0 else 0
    neutral_pct = (neutral_count / total_count * 100) if total_count > 0 else 0
    negative_pct = (negative_count / total_count * 100) if total_count > 0 else 0
    
    # Average compound score
    average_compound = df['Compound'].mean() if total_count > 0 else 0
    
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
    
    # Calculate sentiment trend if enough data points
    sentiment_trend = 'Stable'
    if len(df) >= 3:
        # Convert to datetime and sort
        df_sorted = df.copy()
        df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
        df_sorted = df_sorted.sort_values('Date')
        
        # Split data into two halves
        half_idx = len(df_sorted) // 2
        first_half = df_sorted.iloc[:half_idx]
        second_half = df_sorted.iloc[half_idx:]
        
        # Calculate average compound for each half
        first_half_avg = first_half['Compound'].mean()
        second_half_avg = second_half['Compound'].mean()
        
        # Determine trend
        diff = second_half_avg - first_half_avg
        if diff > 0.2:
            sentiment_trend = 'Strongly Improving'
        elif diff > 0.05:
            sentiment_trend = 'Improving'
        elif diff < -0.2:
            sentiment_trend = 'Strongly Declining'
        elif diff < -0.05:
            sentiment_trend = 'Declining'
        else:
            sentiment_trend = 'Stable'
    
    return {
        'overall_sentiment': overall_sentiment,
        'positive_pct': positive_pct,
        'neutral_pct': neutral_pct,
        'negative_pct': negative_pct,
        'average_compound': average_compound,
        'sentiment_trend': sentiment_trend,
        'count': total_count
    }

def get_sentiment_color(sentiment):
    """
    Get color code for sentiment labels
    
    Parameters:
    - sentiment: Sentiment label
    
    Returns:
    - Color code
    """
    if 'Very Positive' in sentiment or 'Strongly Improving' in sentiment:
        return '#16A34A'  # Bright green
    elif 'Positive' in sentiment or 'Improving' in sentiment:
        return '#4ADE80'  # Light green
    elif 'Very Negative' in sentiment or 'Strongly Declining' in sentiment:
        return '#DC2626'  # Bright red
    elif 'Negative' in sentiment or 'Declining' in sentiment:
        return '#EF4444'  # Light red
    else:
        return '#64748B'  # Neutral blue/grey

@st.cache_data(ttl=3600)
def analyze_sector_sentiment(sector_symbols, num_days=7):
    """
    Analyze sentiment for all symbols in a sector
    
    Parameters:
    - sector_symbols: List of stock ticker symbols
    - num_days: Number of days to look back
    
    Returns:
    - Dictionary with sentiment summary for each symbol
    """
    sector_sentiment = {}
    
    for symbol in sector_symbols:
        news_df = analyze_news_sentiment(symbol, num_days)
        sentiment_summary = get_overall_sentiment_summary(news_df)
        sector_sentiment[symbol] = {
            'summary': sentiment_summary,
            'news_df': news_df
        }
    
    return sector_sentiment
