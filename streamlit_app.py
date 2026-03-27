"""
Stock Valuation and Risk Dashboard

Single-file Streamlit app that fetches 3 years of historical data
for a user-provided ticker, computes basic return statistics,
and provides a simple 5-year revenue/net income forecast with a
final valuation (year 5 net income * P/E multiple).

Dependencies: streamlit, pandas, numpy, yfinance, matplotlib

Run: streamlit run streamlit_app.py
"""

import datetime
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def fetch_stock_data(ticker: str, years: int = 3) -> pd.DataFrame:
    """Fetch historical daily data for `ticker` for the past `years` years.

    Returns a DataFrame with Date index. If ticker is invalid or data
    is empty, returns an empty DataFrame.
    """
    end = date.today()
    start = end - timedelta(days=365 * years)
    try:
        # Use yf.download which returns an empty DataFrame for bad tickers
        df = yf.download(ticker, start=start, end=end)
        return df
    except Exception:
        return pd.DataFrame()


def compute_return_stats(df: pd.DataFrame) -> dict:
    """Compute daily returns, average daily return and annualized volatility."""
    returns = df['Close'].pct_change().dropna()
    avg_daily = returns.mean()
    annual_vol = returns.std() * np.sqrt(252)  # trading days
    return {
        'daily_returns': returns,
        'avg_daily_return': avg_daily,
        'annualized_volatility': annual_vol,
    }


def forecast_financials(current_revenue_m: float, growth_rate: float, profit_margin: float, years: int = 5) -> pd.DataFrame:
    """Forecast revenue and net income for `years` years.

    All monetary values are in millions for clarity.
    """
    revenue_list = []
    net_income_list = []
    rev = current_revenue_m
    for y in range(1, years + 1):
        rev = rev * (1 + growth_rate)
        net = rev * profit_margin
        revenue_list.append(rev)
        net_income_list.append(net)

    years_labels = [f'Year {i}' for i in range(1, years + 1)]
    df = pd.DataFrame({
        'Year': years_labels,
        'Revenue (M)': revenue_list,
        'Net Income (M)': net_income_list,
    })
    return df


def plot_close_price(df: pd.DataFrame, ticker: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], label='Close')
    ax.set_title(f'{ticker.upper()} Closing Price')
    ax.set_ylabel('Price')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    st.pyplot(fig)


def plot_revenue_net(df_proj: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df_proj))
    ax.plot(x, df_proj['Revenue (M)'], marker='o', label='Revenue (M)')
    ax.plot(x, df_proj['Net Income (M)'], marker='o', label='Net Income (M)')
    ax.set_xticks(x)
    ax.set_xticklabels(df_proj['Year'])
    ax.set_title('Projected Revenue and Net Income')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    st.pyplot(fig)


def main():
    st.set_page_config(page_title='Stock Valuation and Risk Dashboard', layout='wide')
    st.title('Stock Valuation and Risk Dashboard')

    # --- Sidebar: inputs ---
    st.sidebar.header('Inputs')
    ticker = st.sidebar.text_input('Stock Ticker', value='AAPL')
    years_of_history = st.sidebar.selectbox('History (years)', options=[1, 2, 3, 5], index=2)

    st.sidebar.markdown('---')
    st.sidebar.subheader('Financial Model Inputs (values in millions)')
    current_revenue_m = st.sidebar.number_input('Current Revenue (M)', value=100000.0, min_value=0.0, step=1000.0)
    growth_pct = st.sidebar.slider('Annual Growth Rate (%)', min_value=0.0, max_value=30.0, value=5.0, step=0.1)
    margin_pct = st.sidebar.slider('Profit Margin (%)', min_value=0.0, max_value=40.0, value=10.0, step=0.1)
    pe_multiple = st.sidebar.slider('P/E Multiple', min_value=5, max_value=40, value=15)

    # Convert percentage inputs to decimals
    growth_rate = growth_pct / 100.0
    profit_margin = margin_pct / 100.0

    # --- Fetch data ---
    with st.spinner(f'Fetching {years_of_history} years of data for {ticker.upper()}...'):
        data = fetch_stock_data(ticker, years=years_of_history)

    if data.empty or 'Close' not in data.columns:
        st.error('Could not fetch data for the ticker provided. Please check the ticker symbol and try again.')
        st.stop()

    # --- Price chart ---
    st.header('Price History')
    plot_close_price(data, ticker)

    # --- Return stats ---
    st.header('Return Statistics')
    stats = compute_return_stats(data)
    col1, col2, col3 = st.columns(3)
    col1.metric('Average Daily Return', f"{stats['avg_daily_return']:.4f}")
    col2.metric('Annualized Volatility', f"{stats['annualized_volatility']:.4f}")
    col3.metric('Most Recent Close', f"${data['Close'][-1]:.2f}")

    st.subheader('Daily Returns (last 30 days)')
    st.line_chart(stats['daily_returns'].tail(30))

    # --- Financial model ---
    st.header('Simple Financial Model')
    st.markdown('Forecast revenue and net income for 5 years using the inputs from the sidebar.')

    proj_df = forecast_financials(current_revenue_m, growth_rate, profit_margin, years=5)

    # Valuation: year 5 net income * P/E
    year5_net_income = proj_df['Net Income (M)'].iloc[-1]
    valuation_m = year5_net_income * pe_multiple  # in millions

    st.subheader('Projection Table')
    # Display with nicer formatting
    display_df = proj_df.copy()
    display_df['Revenue (M)'] = display_df['Revenue (M)'].map(lambda x: f"{x:,.1f}")
    display_df['Net Income (M)'] = display_df['Net Income (M)'].map(lambda x: f"{x:,.1f}")
    st.table(display_df.set_index('Year'))

    st.subheader('Projection Chart')
    plot_revenue_net(proj_df)

    st.subheader('Valuation')
    st.write(f"Year 5 Net Income: {year5_net_income:,.1f} M")
    st.markdown(f"**Estimated Company Valuation = Year 5 Net Income × P/E = {pe_multiple}**")
    st.markdown(f"### Estimated Valuation: **${valuation_m:,.0f} million**")

    st.info('This is a very simple model for illustrative/educational purposes only.')


if __name__ == '__main__':
    main()
