import streamlit as st
import numpy as np
import pandas as pd
 
# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(page_title="Quant Performance Analyzer", layout="wide")
st.title("📈 Quant Performance Analyzer")
 
# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload your data (CSV or Excel)",
    type=["csv", "xlsx"]
)
 
if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()
 
# =====================================================
# LOAD DATA
# =====================================================
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)
 
if "Data" in df.columns:
    df.rename(columns={"Data": "Date"}, inplace=True)
 
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)
 
# =====================================================
# MAIN PAGE CONFIGURATION
# =====================================================
st.subheader("🔧 Configuration")
 
col1, col2 = st.columns(2)
 
with col1:
    rfr_col = st.selectbox(
        "Select Risk-Free Rate Column (13W T-Bill)",
        options=df.columns
    )
 
with col2:
    all_manager_cols = [c for c in df.columns if c not in ["Date", "Year", rfr_col]]
 
    selected_managers = st.multiselect(
        "Select Managers",
        options=all_manager_cols,
        default=all_manager_cols
    )
 
if not selected_managers:
    st.warning("Please select at least one manager.")
    st.stop()
 
manager_cols = selected_managers
 
# =====================================================
# CLEAN RETURNS
# =====================================================
cleaned_df = df.copy()
 
for col in manager_cols + [rfr_col]:
    s = cleaned_df[col].astype(str).str.strip()
    is_pct = s.str.contains("%", regex=False)
 
    s = (
        s.replace(["", "nan", "-", "–"], np.nan)
         .str.replace("%", "", regex=False)
         .str.replace(r"\((.*?)\)", r"-\1", regex=True)
         .astype(float)
    )
 
    s.loc[is_pct] = s.loc[is_pct] / 100
    cleaned_df[col] = s
 
# =====================================================
# HELPERS
# =====================================================
def slice_series(series, years=None):
    s = series.dropna()
    if years is None:
        return s
    months = years * 12
    return s.tail(months) if len(s) >= months else None
 
def annualized_return(s):
    return (np.prod(1 + s)) ** (1 / (len(s) / 12)) - 1 if s is not None and len(s) >= 12 else np.nan
 
def annualized_volatility(s):
    return s.std(ddof=1) * np.sqrt(12) if s is not None and len(s) > 1 else np.nan
 
def annualized_downside_deviation(s):
    if s is None or len(s) < 12:
        return np.nan
    downside = np.minimum(s, 0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(12)
 
def max_drawdown(s):
    if s is None or len(s) == 0:
        return np.nan
    cum = (1 + s).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()
 
# =====================================================
# PROCESSING
# =====================================================
horizons = list(range(1, 11))
horizon_keys = horizons + [None]
labels = [f"{y} Year" if y is not None else "Since Inception" for y in horizon_keys]
 
metrics = [
    "Annualized Return (%)",
    "Annualized Volatility (%)",
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
 
        if s_mgr is None or s_rfr is None:
            continue
 
        ann_mgr_ret = annualized_return(s_mgr)
        ann_rfr_ret = annualized_return(s_rfr)
        ann_vol = annualized_volatility(s_mgr)
        ann_dd = annualized_downside_deviation(s_mgr)
 
        excess_ret = ann_mgr_ret - ann_rfr_ret
 
        results["Annualized Return (%)"].at[label, mgr] = ann_mgr_ret * 100
        results["Annualized Volatility (%)"].at[label, mgr] = ann_vol * 100
        results["Downside Deviation (%)"].at[label, mgr] = ann_dd * 100
        results["Sharpe Ratio"].at[label, mgr] = excess_ret / ann_vol if ann_vol > 0 else np.nan
        results["Sortino Ratio"].at[label, mgr] = excess_ret / ann_dd if ann_dd > 0 else np.nan
        results["Max Drawdown (%)"].at[label, mgr] = max_drawdown(s_mgr) * 100
 
# =====================================================
# DISPLAY
# =====================================================
st.divider()
st.subheader("📊 Results")
 
for metric in metrics:
    with st.expander(metric, expanded=True):
        display_df = results[metric].apply(pd.to_numeric).round(2)
        st.dataframe(display_df, use_container_width=True)
 
# =====================================================
# DOWNLOAD
# =====================================================
st.divider()
st.download_button(
    "⬇️ Download Results (CSV)",
    data=pd.concat(results).to_csv().encode("utf-8"),
    file_name="quant_performance_results.csv",
    mime="text/csv"
)