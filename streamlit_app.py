import datetime
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data
def fetch_stock_data(ticker: str, years: int = 3) -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=365 * years)

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False)

        if df.empty:
            return pd.DataFrame()

        # force Close to be a 1D Series
        if isinstance(df["Close"], pd.DataFrame):
            close_series = df["Close"].iloc[:, 0]
        else:
            close_series = df["Close"]

        clean_df = pd.DataFrame(index=df.index)
        clean_df["Close"] = pd.to_numeric(close_series, errors="coerce")
        clean_df = clean_df.dropna()

        return clean_df

    except Exception:
        return pd.DataFrame()


def compute_return_stats(df: pd.DataFrame) -> dict:
    close_prices = pd.to_numeric(df["Close"], errors="coerce").dropna()
    returns = close_prices.pct_change().dropna()

    avg_daily = float(returns.mean())
    annual_vol = float(returns.std() * np.sqrt(252))

    return {
        "daily_returns": returns,
        "avg_daily_return": avg_daily,
        "annualized_volatility": annual_vol,
    }


def forecast_financials(current_revenue_m, growth_rate, profit_margin, years=5):
    revenue_list = []
    net_income_list = []

    rev = current_revenue_m

    for _ in range(years):
        rev *= (1 + growth_rate)
        net = rev * profit_margin
        revenue_list.append(rev)
        net_income_list.append(net)

    return pd.DataFrame({
        "Year": [f"Year {i}" for i in range(1, years + 1)],
        "Revenue (M)": revenue_list,
        "Net Income (M)": net_income_list,
    })


def generate_signal(avg_return: float, volatility: float) -> str:
    avg_return = float(avg_return)
    volatility = float(volatility)

    if avg_return > 0.001 and volatility < 0.02:
        return "BUY"
    elif avg_return < 0 and volatility > 0.03:
        return "SELL"
    else:
        return "HOLD"


def plot_close_price(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"])
    ax.set_title(f"{ticker.upper()} Closing Price")
    ax.set_ylabel("Price")
    ax.grid(True)
    st.pyplot(fig)


def plot_revenue_net(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))

    ax.plot(x, df["Revenue (M)"], marker="o", label="Revenue")
    ax.plot(x, df["Net Income (M)"], marker="o", label="Net Income")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Year"])
    ax.set_title("Financial Projection")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Stock Dashboard", layout="wide")
    st.title("Stock Valuation & Risk Dashboard")

    st.sidebar.header("Inputs")
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    years = st.sidebar.selectbox("History (years)", [1, 2, 3, 5], index=2)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Financial Model")
    revenue = st.sidebar.number_input("Revenue (M)", value=100000.0, min_value=0.0)
    growth = st.sidebar.slider("Growth %", 0.0, 30.0, 5.0) / 100
    margin = st.sidebar.slider("Margin %", 0.0, 40.0, 10.0) / 100
    pe = st.sidebar.slider("P/E", 5, 40, 15)

    data = fetch_stock_data(ticker, years)

    if data.empty or "Close" not in data.columns:
        st.error("Could not fetch valid data for this ticker. Check the symbol and try again.")
        return

    st.header("Price History")
    plot_close_price(data, ticker)

    st.header("Return Statistics")
    stats = compute_return_stats(data)
    signal = generate_signal(stats["avg_daily_return"], stats["annualized_volatility"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Return", f"{stats['avg_daily_return']:.4f}")
    col2.metric("Volatility", f"{stats['annualized_volatility']:.4f}")
    col3.metric("Price", f"${float(data['Close'].iloc[-1]):.2f}")
    col4.metric("Signal", signal)

    st.subheader("Returns (30 days)")
    st.line_chart(stats["daily_returns"].tail(30))

    st.header("Valuation Model")
    proj = forecast_financials(revenue, growth, margin)

    year5_income = float(proj["Net Income (M)"].iloc[-1])
    valuation = year5_income * pe

    st.subheader("Projection")
    st.dataframe(proj.set_index("Year"))

    plot_revenue_net(proj)

    st.subheader("Valuation Output")
    st.write(f"Year 5 Net Income: {year5_income:,.1f} M")
    st.write(f"P/E Multiple: {pe}")
    st.success(f"Estimated Valuation: ${valuation:,.0f} M")


if __name__ == "__main__":
    main()
