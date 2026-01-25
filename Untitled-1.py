import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# 1. PAGE SETUP
# =====================================================
st.set_page_config(page_title="Quant Performance Analyzer", layout="wide")
st.title("📈 Quant Performance Analyzer")

# =====================================================
# 2. FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload your data (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

# =====================================================
# 3. LOAD & PREPROCESS DATA
# =====================================================
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

if "Data" in df.columns:
    df.rename(columns={"Data": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"], format='%b-%y', errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# =====================================================
# 4. CONFIGURATION (SIDEBAR & MAIN)
# =====================================================
st.sidebar.header("🔧 Settings")

rfr_col = st.sidebar.selectbox(
    "Select Risk-Free Rate Column",
    options=df.columns,
    index=df.columns.get_loc("13 Wk US Treasury Bills") if "13 Wk US Treasury Bills" in df.columns else 0
)

all_manager_cols = [c for c in df.columns if c not in ["Date", "Year", rfr_col]]
manager_cols = st.sidebar.multiselect(
    "Select Managers for Analysis",
    options=all_manager_cols,
    default=all_manager_cols[:5] if len(all_manager_cols) > 5 else all_manager_cols
)

if not manager_cols:
    st.warning("Please select at least one manager in the sidebar.")
    st.stop()

# =====================================================
# 5. DATA CLEANING
# =====================================================
cleaned_df = df.copy()

for col in manager_cols + [rfr_col]:
    s = cleaned_df[col].astype(str).str.strip()
    is_pct = s.str.contains("%", regex=False)

    s = (
        s.replace(["", "nan", "-", "–", " - "], np.nan)
         .str.replace("%", "", regex=False)
         .str.replace(r"\((.*?)\)", r"-\1", regex=True)
         .astype(float)
    )

    s.loc[is_pct] = s.loc[is_pct] / 100
    cleaned_df[col] = s

# =====================================================
# 6. CALCULATION HELPERS
# =====================================================
def slice_series(series, years=None):
    s = series.dropna()
    if years is None:
        return s
    months = int(years * 12)
    return s.tail(months) if len(s) >= months else None

def get_ann_ret(s):
    """Geometric Annualized Return"""
    if s is None or len(s) < 12:
        return np.nan
    return (np.prod(1 + s)) ** (1 / (len(s) / 12)) - 1

def get_ann_vol(s):
    """Annualized Standard Deviation"""
    if s is None or len(s) < 2:
        return np.nan
    return s.std(ddof=1) * np.sqrt(12)

def get_downside_dev(s):
    """Annualized Downside Deviation (Target 0%)"""
    if s is None or len(s) < 12:
        return np.nan
    downside = np.minimum(s, 0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(12)

def get_upside_dev(s):
    """Annualized Upside Deviation"""
    if s is None or len(s) < 12:
        return np.nan
    upside = np.maximum(s, 0)
    return np.sqrt(np.mean(upside ** 2)) * np.sqrt(12)

def get_max_drawdown(s):
    if s is None or len(s) == 0:
        return np.nan
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

# =====================================================
# 7. CALCULATIONS
# =====================================================
horizons = list(range(1, 11))
horizon_keys = horizons + [None]
labels = [f"{y} Year" if y is not None else "Since Inception" for y in horizon_keys]

metrics = [
    "Annualized Return (%)",
    "Annualized Volatility (%)",
    "Upside Deviation (%)",
    "Downside Deviation (%)",
    "Sharpe Ratio",
    "Sortino Ratio",
    "Max Drawdown (%)"
]

results = {m: pd.DataFrame(index=labels, columns=manager_cols) for m in metrics}

for y, label in zip(horizon_keys, labels):
    for mgr in manager_cols:
        s_mgr = slice_series(cleaned_df[mgr], y)
        s_rfr = slice_series(cleaned_df[rfr_col], y)

        if s_mgr is not None and s_rfr is not None:
            ann_mgr_ret = get_ann_ret(s_mgr)
            ann_rfr_ret = get_ann_ret(s_rfr)
            ann_vol = get_ann_vol(s_mgr)
            ann_ddev = get_downside_dev(s_mgr)
            ann_udev = get_upside_dev(s_mgr)
            
            excess_ret = ann_mgr_ret - ann_rfr_ret

            results["Annualized Return (%)"].at[label, mgr] = ann_mgr_ret * 100
            results["Annualized Volatility (%)"].at[label, mgr] = ann_vol * 100
            results["Upside Deviation (%)"].at[label, mgr] = ann_udev * 100
            results["Downside Deviation (%)"].at[label, mgr] = ann_ddev * 100
            results["Sharpe Ratio"].at[label, mgr] = excess_ret / ann_vol if ann_vol > 0 else np.nan
            results["Sortino Ratio"].at[label, mgr] = excess_ret / ann_ddev if ann_ddev > 0 else np.nan
            results["Max Drawdown (%)"].at[label, mgr] = get_max_drawdown(s_mgr) * 100

# =====================================================
# 8. DASHBOARD SECTIONS
# =====================================================

# --- 8.1 PERFORMANCE SNAPSHOT ---
st.divider()
st.subheader("📌 Performance Snapshot")

snapshot_horizon = st.selectbox("Select Evaluation Horizon for Snapshot", options=labels, index=0)

sharpe_row = pd.to_numeric(results["Sharpe Ratio"].loc[snapshot_horizon], errors='coerce')
return_row = pd.to_numeric(results["Annualized Return (%)"].loc[snapshot_horizon], errors='coerce')
drawdown_row = pd.to_numeric(results["Max Drawdown (%)"].loc[snapshot_horizon], errors='coerce')

c1, c2, c3 = st.columns(3)
with c1:
    if not sharpe_row.dropna().empty:
        st.metric("🏆 Best Sharpe", sharpe_row.idxmax(), f"{sharpe_row.max():.2f}")
with c2:
    if not return_row.dropna().empty:
        st.metric("📈 Highest Return", return_row.idxmax(), f"{return_row.max():.2f}%")
with c3:
    if not drawdown_row.dropna().empty:
        st.metric("📉 Lowest Drawdown", drawdown_row.idxmax(), f"{drawdown_row.max():.2f}%")

# --- 8.2 RISK–RETURN ANALYSIS ---
st.divider()
with st.expander("📉 Risk–Return Analysis (Volatility vs Return)", expanded=True):
    rr_label = st.selectbox("Select Horizon for Risk-Return Plot", options=labels, index=0, key="rr_sel")
    
    rr_df = pd.DataFrame({
        "Annualized Return (%)": results["Annualized Return (%)"].loc[rr_label],
        "Annualized Volatility (%)": results["Annualized Volatility (%)"].loc[rr_label]
    }).apply(pd.to_numeric).round(2).dropna()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(rr_df, use_container_width=True)
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(rr_df["Annualized Volatility (%)"], rr_df["Annualized Return (%)"], color='blue')
        for mgr in rr_df.index:
            ax.annotate(mgr, (rr_df.loc[mgr, "Annualized Volatility (%)"], rr_df.loc[mgr, "Annualized Return (%)"]), fontsize=8, alpha=0.7)
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title(f"Risk-Return Plot ({rr_label})")
        ax.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig)

# --- 8.3 UPSIDE–DOWNSIDE ANALYSIS ---
st.divider()
with st.expander("📈 Upside–Downside Analysis (Downside Risk vs Return)", expanded=True):
    ud_label = st.selectbox("Select Horizon for Upside-Downside Plot", options=labels, index=0, key="ud_sel")
    
    ud_df = pd.DataFrame({
        "Annualized Return (%)": results["Annualized Return (%)"].loc[ud_label],
        "Downside Deviation (%)": results["Downside Deviation (%)"].loc[ud_label]
    }).apply(pd.to_numeric).round(2).dropna()

    col_c, col_d = st.columns([1, 2])
    with col_c:
        st.dataframe(ud_df, use_container_width=True)
    with col_d:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(ud_df["Downside Deviation (%)"], ud_df["Annualized Return (%)"], color='red')
        for mgr in ud_df.index:
            ax2.annotate(mgr, (ud_df.loc[mgr, "Downside Deviation (%)"], ud_df.loc[mgr, "Annualized Return (%)"]), fontsize=8, alpha=0.7)
        ax2.set_xlabel("Downside Deviation (%) (Bad Volatility)")
        ax2.set_ylabel("Annualized Return (%) (Reward)")
        ax2.set_title(f"Upside vs. Downside Plot ({ud_label})")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.axvline(0, color='black', lw=0.8)
        ax2.axhline(0, color='black', lw=0.8)
        st.pyplot(fig2)

# --- 8.4 DETAILED METRIC TABLES ---
st.divider()
st.subheader("📊 Detailed Metric Tables")
for metric in metrics:
    with st.expander(f"View: {metric}", expanded=False):
        st.dataframe(results[metric].apply(pd.to_numeric, errors='coerce').round(2), use_container_width=True)

# =====================================================
# 9. DOWNLOAD DATA
# =====================================================
st.divider()
csv_data = pd.concat(results, axis=0).to_csv().encode("utf-8")
st.download_button(
    "⬇️ Download Full Results (CSV)",
    data=csv_data,
    file_name="performance_analytics_report.csv",
    mime="text/csv"
)