from datetime import date, timedelta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Valuation Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .metric-card {
    background: #111620;
    border: 1px solid #1e2d42;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7fa3;
    margin-bottom: 6px;
  }
  .metric-sub { font-size: 11px; color: #6b7fa3; margin-top: 4px; }
  .signal-buy  { color: #00e5a0; font-size: 28px; font-weight: 800; }
  .signal-sell { color: #ff4d6a; font-size: 28px; font-weight: 800; }
  .signal-hold { color: #ffd166; font-size: 28px; font-weight: 800; }
  .thesis-box {
    background: #111620;
    border-left: 3px solid #3d9bff;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 13px;
    line-height: 1.6;
  }
  .warn-box {
    background: #1a1208;
    border-left: 3px solid #ffd166;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 13px;
  }
  .auto-badge {
    display: inline-block;
    background: #0f3020;
    color: #00e5a0;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
  }
  .manual-badge {
    display: inline-block;
    background: #1a1208;
    color: #ffd166;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
  }
  h2, h3 { color: #b0bdd4 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_price_history(ticker: str, years: int) -> pd.DataFrame:
    end   = date.today()
    start = end - timedelta(days=365 * years)
    try:
        return yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    result = {
        "revenue": None, "ebitda": None, "ebit": None, "da": None,
        "capex": None, "total_debt": None, "cash": None,
        "shares": None, "beta": None, "price": None,
        "ebitda_margin": None, "da_pct": None, "capex_pct": None,
        "source": "manual",
    }

    try:
        t    = yf.Ticker(ticker)
        info = t.info

        result["beta"]   = info.get("beta")
        result["price"]  = info.get("currentPrice") or info.get("regularMarketPrice")
        result["shares"] = (info.get("sharesOutstanding") or 0) / 1e6

        income = t.financials
        bs     = t.balance_sheet
        cf     = t.cashflow

        revenue = float(income.loc["Total Revenue"].iloc[0]) / 1e6
        da      = float(income.loc["Reconciled Depreciation"].iloc[0]) / 1e6
        ebit    = float(income.loc["EBIT"].iloc[0]) / 1e6
        ebitda  = ebit + da
        capex   = abs(float(cf.loc["Capital Expenditure"].iloc[0])) / 1e6
        debt    = float(bs.loc["Total Debt"].iloc[0]) / 1e6
        cash    = float(bs.loc["Cash And Cash Equivalents"].iloc[0]) / 1e6

        result.update({
            "revenue":      revenue,
            "ebitda":       ebitda,
            "ebit":         ebit,
            "da":           da,
            "capex":        capex,
            "total_debt":   debt,
            "cash":         cash,
            "ebitda_margin": ebitda / revenue if revenue else None,
            "da_pct":        da / revenue     if revenue else None,
            "capex_pct":     capex / revenue  if revenue else None,
            "source":       "auto",
        })
    except Exception:
        pass

    return result


def compute_technicals(df: pd.DataFrame) -> dict:
    close   = df["Close"].squeeze()
    returns = close.pct_change().dropna()

    avg_daily  = float(returns.mean())
    annual_vol = float(returns.std() * np.sqrt(252))
    sharpe     = (avg_daily * 252) / (annual_vol + 1e-9)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1])

    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    price = float(close.iloc[-1])

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = float((ema12 - ema26).iloc[-1])

    high52        = float(close.rolling(252).max().iloc[-1])
    low52         = float(close.rolling(252).min().iloc[-1])
    pct_from_high = (price - high52) / high52 * 100

    return {
        "returns": returns, "avg_daily": avg_daily,
        "annual_vol": annual_vol, "sharpe": sharpe,
        "rsi": rsi, "ma50": ma50, "ma200": ma200,
        "macd": macd, "price": price,
        "high52": high52, "low52": low52,
        "pct_from_high": pct_from_high,
    }


def technical_signal(t: dict) -> tuple:
    score = 50
    if t["rsi"] < 30:       score += 20
    elif t["rsi"] > 70:     score -= 20
    elif t["rsi"] < 45:     score += 8
    elif t["rsi"] > 55:     score -= 8
    score += 10 if t["price"] > t["ma50"] else -10
    if t["ma200"]:
        score += 10 if t["price"] > t["ma200"] else -10
    score += 8 if t["macd"] > 0 else -8
    if t["avg_daily"] > 0.001:   score += 5
    elif t["avg_daily"] < 0:     score -= 5
    score = max(0, min(100, score))
    if score >= 65:   return "BUY", score
    elif score <= 35: return "SELL", score
    else:             return "HOLD", score


def compute_wacc(beta, risk_free, mkt_premium, cost_of_debt, tax_rate, debt_m, mktcap_m) -> float:
    ke  = risk_free + beta * mkt_premium
    tot = debt_m + mktcap_m
    if tot == 0:
        return ke
    return (mktcap_m / tot) * ke + (debt_m / tot) * cost_of_debt * (1 - tax_rate)


def run_dcf(revenue_m, growth_rate, ebitda_margin, da_pct, capex_pct,
            nwc_pct, tax_rate, wacc, terminal_growth, ev_ebitda_mult,
            total_debt_m, cash_m, shares_m, years=5) -> dict:

    revenues, ebitdas, fcfs, pv_fcfs = [], [], [], []
    rev = revenue_m

    for i in range(1, years + 1):
        rev     *= (1 + growth_rate)
        ebitda   = rev * ebitda_margin
        ebit     = ebitda - rev * da_pct
        nopat    = ebit * (1 - tax_rate)
        fcf      = nopat + rev * da_pct - rev * capex_pct - rev * nwc_pct
        pv       = fcf / ((1 + wacc) ** i)
        revenues.append(rev)
        ebitdas.append(ebitda)
        fcfs.append(fcf)
        pv_fcfs.append(pv)

    sum_pv = sum(pv_fcfs)

    tv_perp  = (fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
                if wacc > terminal_growth else 0.0)
    pv_perp  = tv_perp / ((1 + wacc) ** years)

    tv_eveb  = ebitdas[-1] * ev_ebitda_mult
    pv_eveb  = tv_eveb / ((1 + wacc) ** years)

    net_debt = total_debt_m - cash_m

    def equity_price(pv_tv):
        eq = (sum_pv + pv_tv) - net_debt
        return eq / shares_m if shares_m > 0 else 0.0

    p_perp  = equity_price(pv_perp)
    p_eveb  = equity_price(pv_eveb)
    p_blend = 0.20 * p_perp + 0.80 * p_eveb

    return {
        "revenues": revenues, "ebitdas": ebitdas,
        "fcfs": fcfs, "pv_fcfs": pv_fcfs,
        "sum_pv_fcf": sum_pv,
        "pv_tv_perp": pv_perp, "pv_tv_eveb": pv_eveb,
        "ev_perp":  sum_pv + pv_perp,
        "ev_eveb":  sum_pv + pv_eveb,
        "price_perp":  p_perp,
        "price_eveb":  p_eveb,
        "price_blend": p_blend,
        "net_debt":    net_debt,
    }


def valuation_signal(market_price, dcf_price, wacc, net_debt_m, mktcap_m) -> tuple:
    if dcf_price <= 0 or market_price <= 0:
        return "HOLD", 30, ["Insufficient data for valuation signal"]

    upside   = (dcf_price - market_price) / market_price * 100
    leverage = net_debt_m / (mktcap_m + 1e-9)
    score    = 50
    reasons  = []

    if upside > 30:
        score += 25
        reasons.append(f"✅ DCF implies {upside:.0f}% upside — significantly undervalued")
    elif upside > 10:
        score += 12
        reasons.append(f"✅ DCF implies {upside:.0f}% upside — moderately undervalued")
    elif upside < -30:
        score -= 25
        reasons.append(f"🔴 DCF implies {abs(upside):.0f}% downside — significantly overvalued")
    elif upside < -10:
        score -= 12
        reasons.append(f"🔴 DCF implies {abs(upside):.0f}% downside — moderately overvalued")
    else:
        reasons.append(f"🟡 DCF implies {upside:.0f}% — roughly fairly valued")

    if wacc > 0.12:
        score -= 15
        reasons.append(f"🔴 High WACC ({wacc:.1%}) — future cash flows heavily discounted")
    elif wacc > 0.09:
        score -= 7
        reasons.append(f"🟡 Elevated WACC ({wacc:.1%}) — moderate risk discount")
    else:
        score += 5
        reasons.append(f"✅ Low WACC ({wacc:.1%}) — favorable risk profile")

    if leverage > 1.5:
        score -= 15
        reasons.append(f"🔴 High leverage ({leverage:.1f}x) — debt risk is significant")
    elif leverage > 0.8:
        score -= 7
        reasons.append(f"🟡 Moderate leverage ({leverage:.1f}x)")
    else:
        score += 5
        reasons.append(f"✅ Low leverage ({leverage:.1f}x) — clean balance sheet")

    score = max(0, min(100, score))
    if score >= 65:   return "BUY", score, reasons
    elif score <= 35: return "SELL", score, reasons
    else:             return "HOLD", score, reasons


def build_sensitivity(base_params: dict) -> pd.DataFrame:
    growth_rates = [0.15, 0.12, 0.09, 0.06, 0.03, 0.00, -0.03, -0.06]
    waccs        = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    rows = []
    for g in growth_rates:
        row = {}
        for w in waccs:
            p = {**base_params, "growth_rate": g, "wacc": w}
            row[f"{w:.0%}"] = round(run_dcf(**p)["price_blend"], 2)
        rows.append(row)
    df = pd.DataFrame(rows, index=[f"{g:.0%}" for g in growth_rates])
    df.index.name = "Growth \\ WACC"
    return df


def plot_price_chart(df, ticker):
    close = df["Close"].squeeze()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor("#0a0e14")
    ax.set_facecolor("#0a0e14")
    ax.plot(close.index, close.values, color="#3d9bff", linewidth=1.5, label="Price")
    ax.plot(close.rolling(50).mean().index, close.rolling(50).mean().values,
            color="#ffd166", linewidth=1, linestyle="--", alpha=0.8, label="MA50")
    if len(close) >= 200:
        ax.plot(close.rolling(200).mean().index, close.rolling(200).mean().values,
                color="#ff8c42", linewidth=1, linestyle="--", alpha=0.8, label="MA200")
    ax.set_title(f"{ticker.upper()} — Price & Moving Averages", color="#b0bdd4", fontsize=12)
    ax.tick_params(colors="#6b7fa3", labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor("#1e2d42")
    ax.grid(color="#1e2d42", linewidth=0.5)
    ax.legend(fontsize=9, facecolor="#111620", labelcolor="#b0bdd4", framealpha=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_fcf(dcf):
    years  = [f"Yr {i}" for i in range(1, len(dcf["fcfs"]) + 1)]
    fcfs   = dcf["fcfs"]
    colors = ["#00e5a0" if f >= 0 else "#ff4d6a" for f in fcfs]
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0a0e14")
    ax.set_facecolor("#0a0e14")
    bars = ax.bar(years, fcfs, color=colors, width=0.5)
    ax.axhline(0, color="#1e2d42", linewidth=1)
    for bar, val in zip(bars, fcfs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(abs(v) for v in fcfs) * 0.02,
                f"${val:,.0f}M", ha="center", va="bottom", color="#b0bdd4", fontsize=8)
    ax.set_title("Projected Free Cash Flow ($M)", color="#b0bdd4", fontsize=11)
    ax.tick_params(colors="#6b7fa3", labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor("#1e2d42")
    ax.grid(color="#1e2d42", linewidth=0.5, axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_valuation(dcf, market_price):
    implied = dcf["price_blend"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor("#0a0e14")

    ax1 = axes[0]
    ax1.set_facecolor("#0a0e14")
    sizes = [max(0, dcf["sum_pv_fcf"]), max(0, dcf["pv_tv_eveb"])]
    if sum(sizes) > 0:
        _, _, autotexts = ax1.pie(
            sizes, labels=["PV FCFs", "Terminal Value"],
            colors=["#3d9bff", "#a78bfa"], autopct="%1.0f%%",
            textprops={"color": "#b0bdd4", "fontsize": 9}, startangle=90
        )
        for at in autotexts:
            at.set_color("#0a0e14")
            at.set_fontweight("bold")
    ax1.set_title("Enterprise Value Composition", color="#b0bdd4", fontsize=10)

    ax2 = axes[1]
    ax2.set_facecolor("#0a0e14")
    bars = ax2.bar(
        ["DCF Implied", "Market Price"], [implied, market_price],
        color=["#00e5a0" if implied >= market_price else "#ff4d6a", "#6b7fa3"],
        width=0.4
    )
    for bar, val in zip(bars, [implied, market_price]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.02, f"${val:.2f}",
                 ha="center", va="bottom", color="#e8edf5", fontsize=11, fontweight="bold")
    ax2.set_title("Implied vs Market Price", color="#b0bdd4", fontsize=10)
    ax2.tick_params(colors="#6b7fa3")
    for s in ax2.spines.values(): s.set_edgecolor("#1e2d42")
    ax2.grid(color="#1e2d42", linewidth=0.5, axis="y")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_heatmap(sens_df, market_price):
    vals = sens_df.values.astype(float)
    norm = mcolors.TwoSlopeNorm(
        vmin=vals.min(), vcenter=market_price,
        vmax=max(vals.max(), market_price + 1)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0a0e14")
    ax.set_facecolor("#0a0e14")
    im = ax.imshow(vals, cmap=plt.cm.RdYlGn, norm=norm, aspect="auto")
    ax.set_xticks(range(len(sens_df.columns)))
    ax.set_xticklabels(sens_df.columns, color="#b0bdd4", fontsize=9)
    ax.set_yticks(range(len(sens_df.index)))
    ax.set_yticklabels(sens_df.index, color="#b0bdd4", fontsize=9)
    ax.set_xlabel("WACC", color="#6b7fa3", fontsize=10)
    ax.set_ylabel("Revenue Growth", color="#6b7fa3", fontsize=10)
    ax.set_title(f"Sensitivity Analysis — Market Price ${market_price:.2f}", color="#b0bdd4", fontsize=11)
    for i in range(len(sens_df.index)):
        for j in range(len(sens_df.columns)):
            val = vals[i, j]
            ax.text(j, i, f"${val:.0f}", ha="center", va="center",
                    color="black" if 0.3 < norm(val) < 0.7 else "white",
                    fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Implied Share Price")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.markdown("## 📈 Stock Valuation & Risk Dashboard")
    st.markdown("*DCF · WACC · Technical Analysis · Sensitivity · Confidence Scoring*")
    st.divider()

    with st.sidebar:
        st.header("🔍 Stock")
        ticker     = st.text_input("Ticker", "AAPL").upper().strip()
        hist_years = st.selectbox("Price history", [1, 2, 3, 5], index=2)

        st.divider()
        st.markdown("**Fetching live data...**")
        fund = fetch_fundamentals(ticker)
        is_auto = fund["source"] == "auto"
        badge = '<span class="auto-badge">LIVE</span>' if is_auto else '<span class="manual-badge">MANUAL</span>'
        st.markdown(f"Data source: {badge}", unsafe_allow_html=True)
        if not is_auto:
            st.warning("Could not pull financials automatically. Enter values manually below.")

        def num(label, key, default):
            val = fund.get(key)
            return st.number_input(label, value=float(val if val else default), step=float(default) * 0.05)

        st.divider()
        st.header("📊 DCF Assumptions")
        revenue_m      = num("Revenue ($M)",       "revenue",   10000.0)
        growth_rate    = st.slider("Revenue Growth %", -15.0, 20.0, 6.0, 0.5) / 100
        ebitda_margin  = st.slider("EBITDA Margin %",   5.0, 60.0,
                                   round((fund["ebitda_margin"] or 0.20) * 100, 1), 0.5) / 100
        da_pct         = st.slider("D&A % of Revenue",  1.0, 20.0,
                                   round((fund["da_pct"] or 0.05) * 100, 1), 0.5) / 100
        capex_pct      = st.slider("CapEx % of Revenue", 1.0, 50.0,
                                   round((fund["capex_pct"] or 0.10) * 100, 1), 0.5) / 100
        nwc_pct        = st.slider("NWC Change % of Revenue", -5.0, 10.0, 2.0, 0.5) / 100
        terminal_growth = st.slider("Terminal Growth %", 0.5, 5.0, 2.5, 0.5) / 100
        ev_multiple    = st.slider("EV/EBITDA Exit Multiple", 5.0, 30.0, 12.0, 0.5)

        st.divider()
        st.header("⚖️ Capital Structure")
        total_debt_m = num("Total Debt ($M)",         "total_debt", 1000.0)
        cash_m       = num("Cash ($M)",               "cash",        500.0)
        shares_m     = num("Shares Outstanding (M)",  "shares",      500.0)

        st.divider()
        st.header("🏦 WACC Inputs")
        beta         = st.slider("Beta", 0.3, 3.0, round(float(fund["beta"] or 1.0), 1), 0.1)
        risk_free    = st.slider("Risk Free Rate %",      1.0, 7.0, 4.3, 0.1) / 100
        mkt_premium  = st.slider("Market Risk Premium %", 4.0, 8.0, 5.5, 0.1) / 100
        cost_of_debt = st.slider("Cost of Debt %",        2.0, 12.0, 5.0, 0.1) / 100
        tax_rate     = st.slider("Tax Rate %",           10.0, 35.0, 21.0, 1.0) / 100

    with st.spinner(f"Loading {ticker}..."):
        df = fetch_price_history(ticker, hist_years)

    if df.empty:
        st.error(f"Could not fetch price data for '{ticker}'. Check the ticker symbol.")
        return

    tech    = compute_technicals(df)
    mktcap  = tech["price"] * shares_m
    wacc    = compute_wacc(beta, risk_free, mkt_premium, cost_of_debt, tax_rate, total_debt_m, mktcap)

    dcf_params = dict(
        revenue_m=revenue_m, growth_rate=growth_rate,
        ebitda_margin=ebitda_margin, da_pct=da_pct,
        capex_pct=capex_pct, nwc_pct=nwc_pct,
        tax_rate=tax_rate, wacc=wacc,
        terminal_growth=terminal_growth, ev_ebitda_mult=ev_multiple,
        total_debt_m=total_debt_m, cash_m=cash_m, shares_m=shares_m,
    )
    dcf = run_dcf(**dcf_params)

    t_sig, t_score            = technical_signal(tech)
    v_sig, v_score, v_reasons = valuation_signal(
        tech["price"], dcf["price_blend"], wacc, dcf["net_debt"], mktcap
    )
    combined = round(0.4 * t_score + 0.6 * v_score)
    if combined >= 65:   c_sig, c_cls, c_emoji = "BUY",  "signal-buy",  "🟢"
    elif combined <= 35: c_sig, c_cls, c_emoji = "SELL", "signal-sell", "🔴"
    else:                c_sig, c_cls, c_emoji = "HOLD", "signal-hold", "🟡"

    upside = (dcf["price_blend"] - tech["price"]) / tech["price"] * 100

    st.subheader(f"📋 {ticker} — Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",          f"${tech['price']:.2f}")
    c2.metric("DCF Fair Value", f"${dcf['price_blend']:.2f}")
    c3.metric("Upside / Down",  f"{upside:+.1f}%")
    c4.metric("WACC",           f"{wacc:.2%}")
    c5.metric("RSI (14d)",      f"{tech['rsi']:.1f}")
    c6.metric("Sharpe",         f"{tech['sharpe']:.2f}")
    st.divider()

    st.subheader("🎯 Combined Signal")
    col_sig, col_bars = st.columns([1, 2])
    with col_sig:
        st.markdown(f"""
        <div class="metric-card" style="padding:24px">
          <div class="metric-label">Recommendation</div>
          <div class="{c_cls}">{c_emoji} {c_sig}</div>
          <div class="metric-sub">Confidence: {combined}/100</div>
        </div>""", unsafe_allow_html=True)
    with col_bars:
        st.markdown(f"""
        <div class="metric-card">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span class="metric-label">Technical Score</span>
            <span style="color:#3d9bff;font-weight:700">{t_score}/100 → {t_sig}</span>
          </div>
          <div style="background:#1e2d42;border-radius:4px;height:8px;margin-bottom:12px">
            <div style="background:#3d9bff;width:{t_score}%;height:100%;border-radius:4px"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <span class="metric-label">Valuation Score</span>
            <span style="color:#a78bfa;font-weight:700">{v_score}/100 → {v_sig}</span>
          </div>
          <div style="background:#1e2d42;border-radius:4px;height:8px">
            <div style="background:#a78bfa;width:{v_score}%;height:100%;border-radius:4px"></div>
          </div>
          <div class="metric-sub" style="margin-top:10px">Signal = 40% Technical + 60% Valuation</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="thesis-box"><b>Valuation Thesis</b><br>' +
                "<br>".join(v_reasons) + "</div>", unsafe_allow_html=True)
    st.divider()

    st.subheader("📉 Price History")
    plot_price_chart(df, ticker)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MA50",     f"${tech['ma50']:.2f}",
                "Above" if tech["price"] > tech["ma50"] else "Below")
    col2.metric("MA200",    f"${tech['ma200']:.2f}" if tech["ma200"] else "N/A")
    col3.metric("52W High", f"${tech['high52']:.2f}", f"{tech['pct_from_high']:.1f}%")
    col4.metric("52W Low",  f"${tech['low52']:.2f}")
    st.divider()

    st.subheader("💰 DCF Valuation")
    d1, d2, d3 = st.columns(3)
    d1.metric("Perpetual Growth", f"${dcf['price_perp']:.2f}")
    d2.metric("EV/EBITDA Method", f"${dcf['price_eveb']:.2f}")
    d3.metric("Blended (80/20)",  f"${dcf['price_blend']:.2f}", f"{upside:+.1f}% vs market")
    plot_valuation(dcf, tech["price"])

    proj_df = pd.DataFrame({
        "Revenue ($M)":  [f"${r:,.0f}" for r in dcf["revenues"]],
        "EBITDA ($M)":   [f"${e:,.0f}" for e in dcf["ebitdas"]],
        "FCF ($M)":      [f"${f:,.0f}" for f in dcf["fcfs"]],
        "PV of FCF ($M)":[f"${p:,.0f}" for p in dcf["pv_fcfs"]],
    }, index=[f"Year {i}" for i in range(1, 6)])
    st.dataframe(proj_df, use_container_width=True)

    w1, w2, w3 = st.columns(3)
    w1.metric("WACC",        f"{wacc:.2%}")
    w2.metric("Net Debt",    f"${dcf['net_debt']/1000:.1f}B")
    w3.metric("Sum PV FCFs", f"${dcf['sum_pv_fcf']:,.0f}M")
    st.divider()

    st.subheader("📊 Free Cash Flow Projection")
    plot_fcf(dcf)
    st.divider()

    st.subheader("🔬 Sensitivity Analysis")
    st.caption("Green = above market price. Red = below. Each cell shows blended DCF implied price.")
    with st.spinner("Building sensitivity table..."):
        sens_df = build_sensitivity(dcf_params)
    plot_heatmap(sens_df, tech["price"])
    with st.expander("View raw table"):
        st.dataframe(sens_df.style.format("${:.2f}"), use_container_width=True)
    st.divider()

    st.subheader("⚠️ Risk Flags")
    flags = []
    if wacc > 0.11:
        flags.append(f"🔴 **High WACC ({wacc:.1%})** — Beta {beta} signals elevated market risk.")
    if dcf["net_debt"] / (mktcap + 1e-9) > 1.0:
        flags.append(f"🔴 **Debt exceeds market cap** — Net debt ${dcf['net_debt']/1000:.1f}B vs market cap ${mktcap/1000:.1f}B.")
    if tech["rsi"] > 70:
        flags.append(f"🟡 **RSI overbought ({tech['rsi']:.0f})** — Stock may be due for a pullback.")
    if tech["rsi"] < 30:
        flags.append(f"🟢 **RSI oversold ({tech['rsi']:.0f})** — Potential technical entry opportunity.")
    if tech["pct_from_high"] < -30:
        flags.append(f"🟡 **{abs(tech['pct_from_high']):.0f}% below 52-week high** — Large drawdown. Verify if value or value trap.")
    if terminal_growth >= wacc:
        flags.append("🔴 **Terminal growth ≥ WACC** — Invalid DCF assumption. Reduce terminal growth or increase WACC.")
    if dcf["price_blend"] <= 0:
        flags.append("🔴 **Negative equity value** — Debt overwhelms enterprise value. Extreme caution.")

    if flags:
        for f in flags:
            st.markdown(f'<div class="warn-box">{f}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No major risk flags detected.")

    st.divider()
    st.caption("For educational purposes only. Not financial advice. Always verify data and stress-test assumptions.")


if __name__ == "__main__":
    main()
