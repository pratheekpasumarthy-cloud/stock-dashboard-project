from datetime import date, timedelta

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Valuation Dashboard", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  .stApp { background: #09090b; }
  .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

  section[data-testid="stSidebar"] {
    background: #111113;
    border-right: 1px solid #1f1f23;
  }

  section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
  }

  .dash-title {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #71717a;
    margin-bottom: 2px;
  }

  .dash-ticker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 42px;
    font-weight: 700;
    color: #fafafa;
    line-height: 1;
    margin-bottom: 4px;
  }

  .dash-sub {
    font-size: 12px;
    color: #52525b;
    letter-spacing: 1px;
    margin-bottom: 32px;
  }

  .kpi-row {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1px;
    background: #1f1f23;
    border: 1px solid #1f1f23;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 32px;
  }

  .kpi {
    background: #111113;
    padding: 16px 18px;
  }

  .kpi-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 6px;
  }

  .kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    font-weight: 500;
    color: #fafafa;
    line-height: 1;
  }

  .kpi-value.green  { color: #22c55e; }
  .kpi-value.red    { color: #ef4444; }
  .kpi-value.amber  { color: #f59e0b; }

  .kpi-delta {
    font-size: 11px;
    color: #52525b;
    margin-top: 4px;
  }

  .signal-panel {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 1px;
    background: #1f1f23;
    border: 1px solid #1f1f23;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 32px;
  }

  .signal-main {
    background: #111113;
    padding: 28px 24px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
  }

  .signal-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 10px;
  }

  .signal-word {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 10px;
  }

  .signal-word.buy  { color: #22c55e; }
  .signal-word.sell { color: #ef4444; }
  .signal-word.hold { color: #f59e0b; }

  .signal-conf {
    font-size: 12px;
    color: #52525b;
  }

  .signal-scores {
    background: #111113;
    padding: 24px 28px;
  }

  .score-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 6px;
  }

  .score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
  }

  .score-name {
    font-size: 12px;
    color: #a1a1aa;
    width: 90px;
    flex-shrink: 0;
  }

  .score-bar-bg {
    flex: 1;
    height: 4px;
    background: #27272a;
    border-radius: 2px;
    overflow: hidden;
  }

  .score-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .score-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #71717a;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
  }

  .thesis {
    border-top: 1px solid #1f1f23;
    padding-top: 14px;
    margin-top: 4px;
  }

  .thesis-line {
    font-size: 12px;
    color: #71717a;
    line-height: 1.7;
    padding: 2px 0;
  }

  .thesis-line.neg { color: #ef4444; }
  .thesis-line.pos { color: #22c55e; }
  .thesis-line.neu { color: #f59e0b; }

  .section-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1f1f23;
  }

  .data-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #1f1f23;
    border: 1px solid #1f1f23;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 24px;
  }

  .data-cell {
    background: #111113;
    padding: 14px 16px;
  }

  .data-cell-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #52525b;
    margin-bottom: 5px;
  }

  .data-cell-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px;
    font-weight: 500;
    color: #fafafa;
  }

  .data-cell-sub {
    font-size: 11px;
    margin-top: 3px;
  }

  .dcf-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #1f1f23;
    border: 1px solid #1f1f23;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 24px;
  }

  .flag-row {
    padding: 12px 16px;
    border: 1px solid #1f1f23;
    border-radius: 6px;
    margin-bottom: 8px;
    font-size: 12px;
    line-height: 1.6;
  }

  .flag-row.danger {
    background: #0f0808;
    border-color: #2d1515;
    color: #ef4444;
  }

  .flag-row.warn {
    background: #0f0e08;
    border-color: #2d2915;
    color: #f59e0b;
  }

  .flag-row.ok {
    background: #080f0a;
    border-color: #152d1d;
    color: #22c55e;
  }

  .flag-title {
    font-weight: 600;
    font-size: 12px;
    margin-bottom: 2px;
  }

  .flag-desc {
    font-size: 11px;
    opacity: 0.75;
  }

  .source-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1px;
    padding: 3px 9px;
    border-radius: 3px;
    margin-bottom: 16px;
  }

  .source-live   { background: #14291e; color: #22c55e; border: 1px solid #1a3d2a; }
  .source-manual { background: #29200a; color: #f59e0b; border: 1px solid #3d2f0a; }

  .footer-note {
    font-size: 11px;
    color: #3f3f46;
    margin-top: 32px;
    padding-top: 20px;
    border-top: 1px solid #1f1f23;
  }

  div[data-testid="stMetric"] {
    background: transparent !important;
  }

  div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 18px !important;
    color: #fafafa !important;
  }

  div[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #52525b !important;
  }

  .stDataFrame { border: 1px solid #1f1f23; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_price_history(ticker, years):
    end   = date.today()
    start = end - timedelta(days=365 * years)
    try:
        return yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker):
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
            "revenue": revenue, "ebitda": ebitda, "ebit": ebit, "da": da,
            "capex": capex, "total_debt": debt, "cash": cash,
            "ebitda_margin": ebitda / revenue if revenue else None,
            "da_pct":        da / revenue     if revenue else None,
            "capex_pct":     capex / revenue  if revenue else None,
            "source": "auto",
        })
    except Exception:
        pass
    return result


def compute_technicals(df):
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
        "returns": returns, "avg_daily": avg_daily, "annual_vol": annual_vol,
        "sharpe": sharpe, "rsi": rsi, "ma50": ma50, "ma200": ma200,
        "macd": macd, "price": price, "high52": high52, "low52": low52,
        "pct_from_high": pct_from_high,
    }


def technical_signal(t):
    score = 50
    if t["rsi"] < 30:       score += 20
    elif t["rsi"] > 70:     score -= 20
    elif t["rsi"] < 45:     score += 8
    elif t["rsi"] > 55:     score -= 8
    score += 10 if t["price"] > t["ma50"] else -10
    if t["ma200"]:
        score += 10 if t["price"] > t["ma200"] else -10
    score += 8 if t["macd"] > 0 else -8
    if t["avg_daily"] > 0.001:  score += 5
    elif t["avg_daily"] < 0:    score -= 5
    score = max(0, min(100, score))
    if score >= 65:   return "BUY", score
    elif score <= 35: return "SELL", score
    return "HOLD", score


def compute_wacc(beta, risk_free, mkt_premium, cost_of_debt, tax_rate, debt_m, mktcap_m):
    ke  = risk_free + beta * mkt_premium
    tot = debt_m + mktcap_m
    if tot == 0:
        return ke
    return (mktcap_m / tot) * ke + (debt_m / tot) * cost_of_debt * (1 - tax_rate)


def run_dcf(revenue_m, growth_rate, ebitda_margin, da_pct, capex_pct,
            nwc_pct, tax_rate, wacc, terminal_growth, ev_ebitda_mult,
            total_debt_m, cash_m, shares_m, years=5):
    revenues, ebitdas, fcfs, pv_fcfs = [], [], [], []
    rev = revenue_m
    for i in range(1, years + 1):
        rev    *= (1 + growth_rate)
        ebitda  = rev * ebitda_margin
        ebit    = ebitda - rev * da_pct
        nopat   = ebit * (1 - tax_rate)
        fcf     = nopat + rev * da_pct - rev * capex_pct - rev * nwc_pct
        pv      = fcf / ((1 + wacc) ** i)
        revenues.append(rev); ebitdas.append(ebitda)
        fcfs.append(fcf);     pv_fcfs.append(pv)

    sum_pv  = sum(pv_fcfs)
    tv_perp = (fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
               if wacc > terminal_growth else 0.0)
    pv_perp = tv_perp / ((1 + wacc) ** years)
    tv_eveb = ebitdas[-1] * ev_ebitda_mult
    pv_eveb = tv_eveb / ((1 + wacc) ** years)
    net_debt = total_debt_m - cash_m

    def ep(pv_tv):
        eq = (sum_pv + pv_tv) - net_debt
        return eq / shares_m if shares_m > 0 else 0.0

    p_perp = ep(pv_perp); p_eveb = ep(pv_eveb)
    p_blend = 0.20 * p_perp + 0.80 * p_eveb

    return {
        "revenues": revenues, "ebitdas": ebitdas, "fcfs": fcfs, "pv_fcfs": pv_fcfs,
        "sum_pv_fcf": sum_pv, "pv_tv_perp": pv_perp, "pv_tv_eveb": pv_eveb,
        "ev_perp": sum_pv + pv_perp, "ev_eveb": sum_pv + pv_eveb,
        "price_perp": p_perp, "price_eveb": p_eveb, "price_blend": p_blend,
        "net_debt": net_debt,
    }


def valuation_signal(market_price, dcf_price, wacc, net_debt_m, mktcap_m):
    if dcf_price <= 0 or market_price <= 0:
        return "HOLD", 30, [("Insufficient data for valuation signal", "neu")]
    upside   = (dcf_price - market_price) / market_price * 100
    leverage = net_debt_m / (mktcap_m + 1e-9)
    score    = 50
    reasons  = []

    if upside > 30:
        score += 25; reasons.append((f"DCF implies +{upside:.0f}% upside — significantly undervalued", "pos"))
    elif upside > 10:
        score += 12; reasons.append((f"DCF implies +{upside:.0f}% upside — moderately undervalued", "pos"))
    elif upside < -30:
        score -= 25; reasons.append((f"DCF implies {upside:.0f}% downside — significantly overvalued", "neg"))
    elif upside < -10:
        score -= 12; reasons.append((f"DCF implies {upside:.0f}% downside — moderately overvalued", "neg"))
    else:
        reasons.append((f"DCF implies {upside:.0f}% — roughly fairly valued", "neu"))

    if wacc > 0.12:
        score -= 15; reasons.append((f"High WACC ({wacc:.1%}) — cash flows heavily discounted", "neg"))
    elif wacc > 0.09:
        score -= 7;  reasons.append((f"Elevated WACC ({wacc:.1%}) — moderate risk discount", "neu"))
    else:
        score += 5;  reasons.append((f"Low WACC ({wacc:.1%}) — favorable risk profile", "pos"))

    if leverage > 1.5:
        score -= 15; reasons.append((f"High leverage ({leverage:.1f}x) — debt risk is significant", "neg"))
    elif leverage > 0.8:
        score -= 7;  reasons.append((f"Moderate leverage ({leverage:.1f}x)", "neu"))
    else:
        score += 5;  reasons.append((f"Low leverage ({leverage:.1f}x) — clean balance sheet", "pos"))

    score = max(0, min(100, score))
    if score >= 65:   return "BUY", score, reasons
    elif score <= 35: return "SELL", score, reasons
    return "HOLD", score, reasons


def build_sensitivity(base_params):
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
    df.index.name = "Growth / WACC"
    return df


def _fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#111113")
    ax.set_facecolor("#111113")
    for s in ax.spines.values():
        s.set_edgecolor("#27272a")
    ax.tick_params(colors="#52525b", labelsize=9)
    ax.xaxis.label.set_color("#52525b")
    ax.yaxis.label.set_color("#52525b")
    ax.title.set_color("#a1a1aa")
    ax.grid(color="#1f1f23", linewidth=0.5)
    return fig, ax


def plot_price_chart(df, ticker):
    close = df["Close"].squeeze()
    fig, ax = _fig()
    fig.set_size_inches(11, 3)
    ax.plot(close.index, close.values, color="#3b82f6", linewidth=1.4, label="Price")
    ax.plot(close.rolling(50).mean().index, close.rolling(50).mean().values,
            color="#f59e0b", linewidth=0.9, linestyle="--", alpha=0.7, label="MA 50")
    if len(close) >= 200:
        ax.plot(close.rolling(200).mean().index, close.rolling(200).mean().values,
                color="#8b5cf6", linewidth=0.9, linestyle="--", alpha=0.7, label="MA 200")
    ax.set_title(f"{ticker} — Daily Close", fontsize=11, pad=10)
    ax.legend(fontsize=9, facecolor="#111113", labelcolor="#a1a1aa", framealpha=1,
              edgecolor="#27272a")
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)


def plot_fcf(dcf):
    labels = [f"Yr {i}" for i in range(1, len(dcf["fcfs"]) + 1)]
    fcfs   = dcf["fcfs"]
    colors = ["#22c55e" if f >= 0 else "#ef4444" for f in fcfs]
    fig, ax = _fig()
    fig.set_size_inches(7, 3)
    bars = ax.bar(labels, fcfs, color=colors, width=0.45, zorder=3)
    ax.axhline(0, color="#27272a", linewidth=1)
    span = max(abs(v) for v in fcfs) if fcfs else 1
    for bar, val in zip(bars, fcfs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + span * 0.03,
                f"${val:,.0f}M", ha="center", va="bottom",
                color="#71717a", fontsize=8, fontfamily="monospace")
    ax.set_title("Free Cash Flow Projection", fontsize=11, pad=10)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_valuation(dcf, market_price):
    implied = dcf["price_blend"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    fig.patch.set_facecolor("#111113")

    ax1 = axes[0]
    ax1.set_facecolor("#111113")
    sizes = [max(0, dcf["sum_pv_fcf"]), max(0, dcf["pv_tv_eveb"])]
    if sum(sizes) > 0:
        _, texts, autotexts = ax1.pie(
            sizes, labels=["PV of FCFs", "Terminal Value"],
            colors=["#3b82f6", "#8b5cf6"], autopct="%1.0f%%",
            textprops={"color": "#71717a", "fontsize": 9}, startangle=90,
            wedgeprops={"linewidth": 0}
        )
        for at in autotexts:
            at.set_color("#111113"); at.set_fontweight("700")
    ax1.set_title("EV Composition", color="#a1a1aa", fontsize=10, pad=10)

    ax2 = axes[1]
    ax2.set_facecolor("#111113")
    for s in ax2.spines.values(): s.set_edgecolor("#27272a")
    ax2.tick_params(colors="#52525b")
    bar_color = "#22c55e" if implied >= market_price else "#ef4444"
    bars = ax2.bar(["DCF Implied", "Market Price"],
                   [implied, market_price],
                   color=[bar_color, "#3f3f46"], width=0.4, zorder=3)
    for bar, val in zip(bars, [implied, market_price]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.03, f"${val:.2f}",
                 ha="center", va="bottom", color="#fafafa",
                 fontsize=12, fontweight="700", fontfamily="monospace")
    ax2.set_title("Implied vs Market", color="#a1a1aa", fontsize=10, pad=10)
    ax2.grid(color="#1f1f23", linewidth=0.5, axis="y", zorder=0)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


def plot_heatmap(sens_df, market_price):
    vals = sens_df.values.astype(float)
    norm = mcolors.TwoSlopeNorm(
        vmin=vals.min(), vcenter=market_price,
        vmax=max(vals.max(), market_price + 1)
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#111113")
    ax.set_facecolor("#111113")
    cmap = plt.cm.RdYlGn
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(sens_df.columns)))
    ax.set_xticklabels(sens_df.columns, color="#71717a", fontsize=9)
    ax.set_yticks(range(len(sens_df.index)))
    ax.set_yticklabels(sens_df.index, color="#71717a", fontsize=9)
    ax.set_xlabel("WACC", color="#52525b", fontsize=10)
    ax.set_ylabel("Revenue Growth", color="#52525b", fontsize=10)
    ax.set_title(f"Sensitivity — market at ${market_price:.2f}",
                 color="#a1a1aa", fontsize=11, pad=10)
    for i in range(len(sens_df.index)):
        for j in range(len(sens_df.columns)):
            val = vals[i, j]
            ax.text(j, i, f"${val:.0f}", ha="center", va="center",
                    color="black" if 0.3 < norm(val) < 0.7 else "white",
                    fontsize=8, fontweight="600", fontfamily="monospace")
    cb = plt.colorbar(im, ax=ax)
    cb.ax.yaxis.set_tick_params(color="#52525b")
    cb.ax.tick_params(labelsize=8, labelcolor="#52525b")
    cb.outline.set_edgecolor("#27272a")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


def main():
    with st.sidebar:
        st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#52525b;margin-bottom:16px">Configuration</div>', unsafe_allow_html=True)
        ticker     = st.text_input("Ticker", "AAPL").upper().strip()
        hist_years = st.selectbox("History", [1, 2, 3, 5], index=2)

        fund    = fetch_fundamentals(ticker)
        is_auto = fund["source"] == "auto"
        pill_cls = "source-live" if is_auto else "source-manual"
        pill_txt = "Live data" if is_auto else "Manual mode"
        st.markdown(f'<span class="{pill_cls}">{pill_txt}</span>', unsafe_allow_html=True)
        if not is_auto:
            st.warning("Could not pull financials. Enter values below.")

        def num(label, key, default):
            val = fund.get(key)
            return st.number_input(label, value=float(val if val else default), step=float(default) * 0.05)

        st.markdown("---")
        st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#52525b;margin-bottom:8px">DCF</div>', unsafe_allow_html=True)
        revenue_m      = num("Revenue ($M)",        "revenue",    10000.0)
        growth_rate    = st.slider("Revenue growth %",  -15.0, 20.0, 6.0,  0.5) / 100
        ebitda_margin  = st.slider("EBITDA margin %",     5.0, 60.0,
                                   round((fund["ebitda_margin"] or 0.20) * 100, 1), 0.5) / 100
        da_pct         = st.slider("D&A % of revenue",    1.0, 20.0,
                                   round((fund["da_pct"]     or 0.05) * 100, 1), 0.5) / 100
        capex_pct      = st.slider("CapEx % of revenue",  1.0, 50.0,
                                   round((fund["capex_pct"] or 0.10) * 100, 1), 0.5) / 100
        nwc_pct        = st.slider("NWC change % of rev", -5.0, 10.0, 2.0, 0.5) / 100
        terminal_growth = st.slider("Terminal growth %",   0.5,  5.0, 2.5, 0.5) / 100
        ev_multiple    = st.slider("EV/EBITDA multiple",   5.0, 30.0, 12.0, 0.5)

        st.markdown("---")
        st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#52525b;margin-bottom:8px">Capital structure</div>', unsafe_allow_html=True)
        total_debt_m = num("Total debt ($M)",        "total_debt", 1000.0)
        cash_m       = num("Cash ($M)",              "cash",        500.0)
        shares_m     = num("Shares outstanding (M)", "shares",      500.0)

        st.markdown("---")
        st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#52525b;margin-bottom:8px">WACC</div>', unsafe_allow_html=True)
        beta         = st.slider("Beta",               0.3, 3.0, round(float(fund["beta"] or 1.0), 1), 0.1)
        risk_free    = st.slider("Risk free rate %",   1.0, 7.0, 4.3, 0.1) / 100
        mkt_premium  = st.slider("Market premium %",  4.0, 8.0, 5.5, 0.1) / 100
        cost_of_debt = st.slider("Cost of debt %",    2.0, 12.0, 5.0, 0.1) / 100
        tax_rate     = st.slider("Tax rate %",        10.0, 35.0, 21.0, 1.0) / 100

    with st.spinner(f"Loading {ticker}..."):
        df = fetch_price_history(ticker, hist_years)

    if df.empty:
        st.error(f"No data for '{ticker}'. Check the ticker and try again.")
        return

    tech    = compute_technicals(df)
    mktcap  = tech["price"] * shares_m
    wacc    = compute_wacc(beta, risk_free, mkt_premium, cost_of_debt, tax_rate, total_debt_m, mktcap)

    dcf_params = dict(
        revenue_m=revenue_m, growth_rate=growth_rate, ebitda_margin=ebitda_margin,
        da_pct=da_pct, capex_pct=capex_pct, nwc_pct=nwc_pct, tax_rate=tax_rate,
        wacc=wacc, terminal_growth=terminal_growth, ev_ebitda_mult=ev_multiple,
        total_debt_m=total_debt_m, cash_m=cash_m, shares_m=shares_m,
    )
    dcf = run_dcf(**dcf_params)

    t_sig, t_score            = technical_signal(tech)
    v_sig, v_score, v_reasons = valuation_signal(
        tech["price"], dcf["price_blend"], wacc, dcf["net_debt"], mktcap
    )
    combined = round(0.4 * t_score + 0.6 * v_score)
    if combined >= 65:   c_sig, c_cls = "BUY",  "buy"
    elif combined <= 35: c_sig, c_cls = "SELL", "sell"
    else:                c_sig, c_cls = "HOLD", "hold"
    upside = (dcf["price_blend"] - tech["price"]) / tech["price"] * 100

    st.markdown(f'<div class="dash-title">Valuation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="dash-ticker">{ticker}</div>', unsafe_allow_html=True)
    price_chg_cls = "green" if upside >= 0 else "red"
    st.markdown(f'<div class="dash-sub">DCF implied {upside:+.1f}% vs market &nbsp;·&nbsp; WACC {wacc:.2%} &nbsp;·&nbsp; {hist_years}Y history</div>', unsafe_allow_html=True)

    up_cls = "green" if upside >= 0 else "red"
    rsi_cls = "red" if tech["rsi"] > 70 else ("green" if tech["rsi"] < 30 else "")
    sharpe_cls = "green" if tech["sharpe"] > 1 else ("red" if tech["sharpe"] < 0 else "")

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi">
        <div class="kpi-label">Market price</div>
        <div class="kpi-value">${tech['price']:.2f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">DCF fair value</div>
        <div class="kpi-value {up_cls}">${dcf['price_blend']:.2f}</div>
        <div class="kpi-delta">blended 80/20</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Implied move</div>
        <div class="kpi-value {up_cls}">{upside:+.1f}%</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">WACC</div>
        <div class="kpi-value {'red' if wacc > 0.10 else ''}">{wacc:.2%}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">RSI 14d</div>
        <div class="kpi-value {rsi_cls}">{tech['rsi']:.1f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Sharpe ratio</div>
        <div class="kpi-value {sharpe_cls}">{tech['sharpe']:.2f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    thesis_html = "".join(
        f'<div class="thesis-line {cls}">{text}</div>'
        for text, cls in v_reasons
    )

    st.markdown(f"""
    <div class="signal-panel">
      <div class="signal-main">
        <div class="signal-eyebrow">Signal</div>
        <div class="signal-word {c_cls}">{c_sig}</div>
        <div class="signal-conf">{combined}/100 confidence</div>
      </div>
      <div class="signal-scores">
        <div class="score-label">Score breakdown</div>
        <div class="score-row">
          <div class="score-name">Technical</div>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{t_score}%;background:#3b82f6"></div>
          </div>
          <div class="score-num">{t_score} — {t_sig}</div>
        </div>
        <div class="score-row">
          <div class="score-name">Valuation</div>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{v_score}%;background:#8b5cf6"></div>
          </div>
          <div class="score-num">{v_score} — {v_sig}</div>
        </div>
        <div class="thesis">{thesis_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">Price history</div>', unsafe_allow_html=True)
    plot_price_chart(df, ticker)

    ma200_val = f"${tech['ma200']:.2f}" if tech["ma200"] else "N/A"
    ma200_sub = ("above" if tech["price"] > tech["ma200"] else "below") if tech["ma200"] else ""
    st.markdown(f"""
    <div class="data-strip">
      <div class="data-cell">
        <div class="data-cell-label">MA 50</div>
        <div class="data-cell-val">${tech['ma50']:.2f}</div>
        <div class="data-cell-sub {'green' if tech['price'] > tech['ma50'] else 'red'}" style="color:{'#22c55e' if tech['price'] > tech['ma50'] else '#ef4444'}">price {'above' if tech['price'] > tech['ma50'] else 'below'}</div>
      </div>
      <div class="data-cell">
        <div class="data-cell-label">MA 200</div>
        <div class="data-cell-val">{ma200_val}</div>
        <div class="data-cell-sub" style="color:#52525b">{ma200_sub}</div>
      </div>
      <div class="data-cell">
        <div class="data-cell-label">52W high</div>
        <div class="data-cell-val">${tech['high52']:.2f}</div>
        <div class="data-cell-sub" style="color:#ef4444">{tech['pct_from_high']:.1f}% off high</div>
      </div>
      <div class="data-cell">
        <div class="data-cell-label">52W low</div>
        <div class="data-cell-val">${tech['low52']:.2f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-eyebrow">DCF valuation</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="dcf-strip">
      <div class="kpi">
        <div class="kpi-label">Perpetual growth</div>
        <div class="kpi-value {'green' if dcf['price_perp'] >= tech['price'] else 'red'}">${dcf['price_perp']:.2f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">EV / EBITDA</div>
        <div class="kpi-value {'green' if dcf['price_eveb'] >= tech['price'] else 'red'}">${dcf['price_eveb']:.2f}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Blended 80/20</div>
        <div class="kpi-value {'green' if dcf['price_blend'] >= tech['price'] else 'red'}">${dcf['price_blend']:.2f}</div>
        <div class="kpi-delta">{upside:+.1f}% vs market</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    plot_valuation(dcf, tech["price"])

    proj_df = pd.DataFrame({
        "Revenue ($M)":  [f"${r:,.0f}" for r in dcf["revenues"]],
        "EBITDA ($M)":   [f"${e:,.0f}" for e in dcf["ebitdas"]],
        "FCF ($M)":      [f"${f:,.0f}" for f in dcf["fcfs"]],
        "PV of FCF ($M)":[f"${p:,.0f}" for p in dcf["pv_fcfs"]],
    }, index=[f"Year {i}" for i in range(1, 6)])
    st.dataframe(proj_df, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("WACC",        f"{wacc:.2%}")
    col_b.metric("Net debt",    f"${dcf['net_debt']/1000:.1f}B")
    col_c.metric("Sum PV FCFs", f"${dcf['sum_pv_fcf']:,.0f}M")

    st.markdown("---")
    st.markdown('<div class="section-eyebrow">Free cash flow</div>', unsafe_allow_html=True)
    plot_fcf(dcf)

    st.markdown("---")
    st.markdown('<div class="section-eyebrow">Sensitivity — blended DCF price</div>', unsafe_allow_html=True)
    st.caption("Each cell shows implied share price. Green cells are above the current market price. Red cells are below.")
    with st.spinner("Building sensitivity table..."):
        sens_df = build_sensitivity(dcf_params)
    plot_heatmap(sens_df, tech["price"])
    with st.expander("Raw table"):
        st.dataframe(sens_df.style.format("${:.2f}"), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-eyebrow">Risk flags</div>', unsafe_allow_html=True)

    flags = []
    if wacc > 0.11:
        flags.append(("danger", "High WACC", f"Beta {beta:.1f} signals elevated market risk. At {wacc:.1%}, future cash flows are heavily discounted."))
    if dcf["net_debt"] / (mktcap + 1e-9) > 1.0:
        flags.append(("danger", "Debt exceeds market cap", f"Net debt ${dcf['net_debt']/1000:.1f}B vs market cap ${mktcap/1000:.1f}B. Equity value is at risk."))
    if tech["rsi"] > 70:
        flags.append(("warn", "RSI overbought", f"RSI at {tech['rsi']:.0f}. Stock may be extended — watch for a pullback."))
    if tech["rsi"] < 30:
        flags.append(("ok", "RSI oversold", f"RSI at {tech['rsi']:.0f}. Potential technical entry on a bounce."))
    if tech["pct_from_high"] < -30:
        flags.append(("warn", "Large drawdown", f"{abs(tech['pct_from_high']):.0f}% below 52-week high. Verify whether this is value or a value trap."))
    if terminal_growth >= wacc:
        flags.append(("danger", "Invalid terminal growth assumption", "Terminal growth rate exceeds WACC. Reduce terminal growth or increase WACC."))
    if dcf["price_blend"] <= 0:
        flags.append(("danger", "Negative equity value", "Debt overwhelms enterprise value at these inputs. Treat result with extreme caution."))

    if not flags:
        st.markdown('<div class="flag-row ok"><div class="flag-title">No major risk flags</div><div class="flag-desc">All checks passed at current inputs.</div></div>', unsafe_allow_html=True)
    else:
        for cls, title, desc in flags:
            st.markdown(f'<div class="flag-row {cls}"><div class="flag-title">{title}</div><div class="flag-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="footer-note">For educational and research purposes only. Not financial advice. DCF outputs are highly sensitive to assumptions — always cross-check with other methods and primary sources.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
