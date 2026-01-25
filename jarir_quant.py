import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import tempfile
import os

# =====================================================
# 1. PAGE SETUP
# =====================================================
st.set_page_config(page_title="Quant Performance Analyzer", layout="wide")
st.title("📈 Quant Performance Analyzer")

# =====================================================
# 2. FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload your data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

# =====================================================
# 3. DATA LOADING
# =====================================================
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    df.columns = df.columns.str.strip()
    if "Data" in df.columns:
        df.rename(columns={"Data": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format='%b-%y', errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# =====================================================
# 4. SETTINGS (SIDEBAR)
# =====================================================
st.sidebar.header("🔧 Settings")

rfr_options = list(df.columns)
rfr_index = df.columns.get_loc("13 Wk US Treasury Bills") if "13 Wk US Treasury Bills" in df.columns else 0

rfr_col = st.sidebar.selectbox("Select Risk-Free Rate Column", options=rfr_options, index=rfr_index)

all_manager_cols = [c for c in df.columns if c not in ["Date", "Year", rfr_col]]
manager_cols = st.sidebar.multiselect("Select Managers", options=all_manager_cols, default=all_manager_cols[:10])

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
    s = s.replace(["", "nan", "-", "–", " - "], np.nan).str.replace("%", "", regex=False).str.replace(r"\((.*?)\)", r"-\1", regex=True).astype(float)
    s.loc[is_pct] = s.loc[is_pct] / 100
    cleaned_df[col] = s

# =====================================================
# 6. CALCULATIONS
# =====================================================
def get_metrics(s, rfr_s):
    if s is None or len(s) < 12: 
        return [np.nan] * 7 
    
    ann_ret = ((1 + s).prod()) ** (1 / (len(s) / 12)) - 1
    ann_rfr = ((1 + rfr_s).prod()) ** (1 / (len(rfr_s) / 12)) - 1
    ann_vol = s.std(ddof=1) * np.sqrt(12)
    upside = np.sqrt(np.mean(np.maximum(s, 0)**2)) * np.sqrt(12)
    downside = np.sqrt(np.mean(np.minimum(s, 0)**2)) * np.sqrt(12)
    
    excess = ann_ret - ann_rfr
    sharpe = excess / ann_vol if ann_vol > 0 else np.nan
    sortino = excess / downside if downside > 0 else np.nan
    
    cum = (1 + s).cumprod()
    mdd = ((cum / cum.cummax()) - 1).min()
    
    return ann_ret*100, ann_vol*100, upside*100, downside*100, sharpe, sortino, mdd*100

horizons = [1, 3, 4, 5, 8, 10, None]
labels = [f"{y} Year" if y else "Inception" for y in horizons]
res_dict = {mgr: {} for mgr in manager_cols}

for y, label in zip(horizons, labels):
    for mgr in manager_cols:
        count = int(y * 12) if y else len(cleaned_df[mgr].dropna())
        s_mgr = cleaned_df[mgr].dropna().tail(count)
        s_rfr = cleaned_df[rfr_col].dropna().tail(count)
        res_dict[mgr][label] = get_metrics(s_mgr, s_rfr)

# =====================================================
# 7. DASHBOARD DISPLAY
# =====================================================
st.divider()
view_label = st.selectbox("Select View Horizon", options=labels, index=1)

p_list = []
for mgr in manager_cols:
    m = res_dict[mgr][view_label]
    p_list.append({"Manager": mgr, "Return (%)": m[0], "Volatility (%)": m[1], "Sharpe": m[4], "Sortino": m[5], "MDD (%)": m[6]})
p_df = pd.DataFrame(p_list).dropna()

st.subheader(f"Analysis for {view_label}")
fig = px.scatter(p_df, x="Volatility (%)", y="Return (%)", text="Manager", color="Sharpe", color_continuous_scale="RdYlGn")
fig.update_traces(textposition='top center')
st.plotly_chart(fig, use_container_width=True)

st.dataframe(p_df.set_index("Manager").style.format("{:.2f}"), use_container_width=True)

# =====================================================
# 8. PDF ENGINE (3, 5, 8 Year Report)
# =====================================================
def generate_pdf(res_dict, manager_cols):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 24)
    pdf.cell(0, 40, "Quant Analysis Final Report", ln=True, align='C')
    pdf.set_font("helvetica", '', 12)
    pdf.cell(0, 10, f"Period Covered: 3Y, 5Y, 8Y horizons", ln=True, align='C')
    pdf.cell(0, 10, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", ln=True, align='C')

    report_years = ["3 Year", "5 Year", "8 Year"]
    
    for yr in report_years:
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 16)
        pdf.cell(0, 15, f"Horizon: {yr}", ln=True)
        
        # Performance Table
        pdf.set_font("helvetica", 'B', 8)
        cols = ["Manager", "Return%", "Vol%", "Sharpe", "Sortino", "MDD%"]
        w = pdf.epw / len(cols)
        for c in cols: pdf.cell(w, 8, c, border=1, align='C')
        pdf.ln()
        
        pdf.set_font("helvetica", '', 8)
        chart_data = []
        for mgr in manager_cols:
            m = res_dict[mgr].get(yr, [np.nan]*7)
            if not np.isnan(m[0]):
                chart_data.append({"Mgr": mgr, "Ret": m[0], "Vol": m[1]})
                row = [mgr[:15], f"{m[0]:.2f}", f"{m[1]:.2f}", f"{m[4]:.2f}", f"{m[5]:.2f}", f"{m[6]:.2f}"]
                for val in row: pdf.cell(w, 7, str(val), border=1, align='C')
                pdf.ln()
        
        # Graphs
        if chart_data:
            pdf.ln(10)
            c_df = pd.DataFrame(chart_data)
            fig_pdf = px.scatter(c_df, x="Vol", y="Ret", text="Mgr", title=f"Risk-Return {yr}")
            fig_pdf.update_traces(textposition='top center')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                # This requires kaleido
                fig_pdf.write_image(tmp.name, engine="kaleido")
                pdf.image(tmp.name, x=10, y=pdf.get_y(), w=180)
                os.unlink(tmp.name) # Clean up temp file

    return pdf.output()

# =====================================================
# 9. DOWNLOAD BUTTON
# =====================================================
st.divider()
st.subheader("📥 Download Final Report")
if st.button("Prepare PDF"):
    try:
        pdf_content = generate_pdf(res_dict, manager_cols)
        st.download_button(
            label="Download PDF Now",
            data=bytes(pdf_content),
            file_name="Quant_Full_Report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Failed to create PDF. Error: {e}")
        st.info("Check if 'kaleido' and 'fpdf2' are in your requirements.txt")
