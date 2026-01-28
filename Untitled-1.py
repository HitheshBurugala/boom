import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, datetime as dt
import openai

# ====== PDF imports (reportlab) ======
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (
    SimpleDocTemplate, LongTable, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# =====================================================
# 1. PAGE SETUP & DATA LOADING
# =====================================================
st.set_page_config(page_title="Jarir Quant Analyzer", layout="wide")
st.title("Jarir Quant Analysis with AI assistance")

uploaded_file = st.file_uploader("Please upload the raw quant file to begin.", type=["csv", "xlsx"])

if uploaded_file is None:
    st.info("Please upload the raw quant file to begin.")
    st.stop()

@st.cache_data
def load_and_clean_base(file):
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    df.columns = df.columns.str.strip()
    if "Data" in df.columns: df.rename(columns={"Data": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format='%b-%y', errors="coerce")
    if "Year" in df.columns:
        df['Date'] = df.apply(lambda x: x['Date'].replace(year=int(x['Year'])) if pd.notnull(x['Date']) else x['Date'], axis=1)
    mgr_cols = [c for c in df.columns if c not in ["Date", "Year"]]
    last_idx = df[mgr_cols].dropna(how='all').index.max()
    return df.iloc[:last_idx + 1].sort_values("Date").reset_index(drop=True)

df_raw = load_and_clean_base(uploaded_file)

# =====================================================
# 2. SIDEBAR SETTINGS (FILTERS)
# =====================================================
st.sidebar.header("Analysis Settings")
available_dates = df_raw["Date"].dropna().sort_values().unique()
date_labels = [d.strftime('%b-%Y') for d in available_dates]

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    start_label = st.selectbox("Start Month", options=date_labels, index=0)
with col_s2:
    end_label = st.selectbox("End Month", options=date_labels, index=len(date_labels)-1)

start_dt = pd.to_datetime(start_label, format='%b-%Y')
end_dt = pd.to_datetime(end_label, format='%b-%Y')
df_filtered = df_raw[(df_raw["Date"] >= start_dt) & (df_raw["Date"] <= end_dt)].reset_index(drop=True)

rfr_target = "13 Wk US Treasury Bills"
default_rfr_idx = df_filtered.columns.get_loc(rfr_target) if rfr_target in df_filtered.columns else 0
rfr_col = st.sidebar.selectbox("Risk-Free Rate Column", options=df_filtered.columns, index=default_rfr_idx)

all_mgrs = [c for c in df_filtered.columns if c not in ["Date", "Year", rfr_col]]
manager_cols = st.sidebar.multiselect("Select Managers", options=all_mgrs, default=all_mgrs[:5])

if not manager_cols:
    st.warning("Please select managers.")
    st.stop()

# =====================================================
# 3. ANALYTICS ENGINE
# =====================================================
cleaned_df = df_filtered.copy()
for col in manager_cols + [rfr_col]:
    s = cleaned_df[col].astype(str).str.strip()
    is_pct = s.str.contains("%", regex=False)
    s = s.replace(["", "nan", "-", "–"], np.nan).str.replace("%", "", regex=False).str.replace(r"\((.*?)\)", r"-\1", regex=True).astype(float)
    if is_pct.any(): s /= 100
    cleaned_df[col] = s

def get_cap(s_m, s_b):
    combined = pd.DataFrame({'m': s_m, 'b': s_b}).dropna()
    up = combined[combined['b'] > 0]; dn = combined[combined['b'] < 0]
    u_cap = (np.prod(1+up['m'])/np.prod(1+up['b']))*100 if not up.empty else np.nan
    d_cap = (np.prod(1+dn['m'])/np.prod(1+dn['b']))*100 if not dn.empty else np.nan
    return u_cap, d_cap

horizons = [1, 2, 3, 4, 5, 8, 10]
h_labels = [f"{y} Year" for y in horizons] + ["Since Inception"]
metrics = ["Annualized Return (%)", "Annualized Volatility (%)", "Upward Deviation (%)", "Downward Deviation (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)"]
results = {m: pd.DataFrame(index=h_labels, columns=manager_cols) for m in metrics}

for idx, yrs in enumerate(horizons + [None]):
    lbl = h_labels[idx]
    for mgr in manager_cols:
        s_m = cleaned_df[mgr].dropna(); s_rf = cleaned_df[rfr_col].dropna()
        if yrs and len(s_m) >= yrs*12: s_m, s_rf = s_m.tail(yrs*12), s_rf.tail(yrs*12)
        elif yrs: continue
        ann_ret = (np.prod(s_m + 1, axis=0) ** (12 / len(s_m))) - 1
        vol = s_m.std(ddof=1) * np.sqrt(12)
        udev = (np.sqrt(np.mean(np.maximum(s_m, 0)**2)) * np.sqrt(12)) * 100
        ddev = (np.sqrt(np.mean(np.minimum(s_m, 0)**2)) * np.sqrt(12)) * 100
        rfr_ann = (np.prod(s_rf + 1, axis=0) ** (12 / len(s_rf))) - 1
        res_v = [ann_ret*100, vol*100, udev, ddev, (ann_ret-rfr_ann)/vol if vol > 0 else np.nan, (ann_ret-rfr_ann)/(ddev/100) if ddev > 0 else np.nan, (((1+s_m).cumprod()/(1+s_m).cumprod().cummax())-1).min()*100]
        for i, m in enumerate(metrics): results[m].at[lbl, mgr] = res_v[i]

# =====================================================
# 4. DASHBOARD UI
# =====================================================
# NAN FIX: Added placeholder="" to hide NaN values
def style_df(df_in, pct=True):
    fmt = "{:.2f}%" if pct else "{:.2f}"
    return df_in.apply(pd.to_numeric).style.map(lambda x: 'color: red' if x < 0 else 'color: black').background_gradient(cmap='BuGn', axis=None).format(fmt, na_rep="")

for m in metrics:
    is_p = "Ratio" not in m
    with st.expander(f"View: {m}", expanded=True):
        st.dataframe(style_df(results[m], pct=is_p), width="stretch", height=(len(h_labels)+1)*35+15)

st.divider()
viz_h = st.selectbox("Select Years for Plots", options=h_labels, index=len(h_labels)-2)
col_v1, col_v2 = st.columns(2)
with col_v1:
    st.write(f"**Risk-Return Plot ({viz_h})**")
    rr_ui = pd.DataFrame({"Return (%)": results["Annualized Return (%)"].loc[viz_h], "Volatility (%)": results["Annualized Volatility (%)"].loc[viz_h]}).apply(pd.to_numeric).dropna()
    fig_rr, ax_rr = plt.subplots(figsize=(8, 5))
    ax_rr.scatter(rr_ui["Volatility (%)"], rr_ui["Return (%)"], color='teal')
    for i, txt in enumerate(rr_ui.index): ax_rr.annotate(txt, (rr_ui.iloc[i,1], rr_ui.iloc[i,0]), fontsize=9, fontweight='bold')
    ax_rr.set_xlabel("Volatility (%)"); ax_rr.set_ylabel("Return (%)"); ax_rr.grid(True, ls=':'); st.pyplot(fig_rr)

with col_v2:
    b_cap = st.selectbox("Benchmark for Capture", options=manager_cols, index=len(manager_cols)-1)
    st.write(f"**Capture Matrix ({viz_h}) — Benchmark: {b_cap}**")
    caps_ui = []
    for m in manager_cols:
        y_c = int(viz_h.split()[0]) if "Year" in viz_h else None
        s_m = cleaned_df[m].tail(y_c*12) if y_c else cleaned_df[m]
        s_b = cleaned_df[b_cap].tail(y_c*12) if y_c else cleaned_df[b_cap]
        u, d = get_cap(s_m.dropna(), s_b.dropna()); caps_ui.append({"Manager": m, "Upside": u, "Downside": d})
    cap_ui_df = pd.DataFrame(caps_ui).set_index("Manager")
    fig_cap, ax_cap = plt.subplots(figsize=(8, 5))
    ax_cap.scatter(cap_ui_df["Downside"], cap_ui_df["Upside"], color='darkred')
    for i, txt in enumerate(cap_ui_df.index): ax_cap.annotate(txt, (cap_ui_df.iloc[i,1], cap_ui_df.iloc[i,0]), fontsize=9, fontweight='bold')
    ax_cap.axvline(100, color='black', lw=1); ax_cap.axhline(100, color='black', lw=1); st.pyplot(fig_cap)

# Calendar & Alpha UI
st.divider()
cal_base = cleaned_df.set_index('Date')[manager_cols]
cal_ret = cal_base.groupby(cal_base.index.year).apply(lambda x: (np.prod(x + 1, axis=0) - 1) * 100).sort_index(ascending=False)
st.write("**Calendar Returns (%)**")
st.dataframe(style_df(cal_ret), width="stretch")

bench_diff = st.selectbox("Select Alpha Benchmark", options=manager_cols, index=len(manager_cols)-1)
cal_diff = cal_ret.subtract(cal_ret[bench_diff], axis=0)
st.write(f"**Calendar Difference (Alpha vs {bench_diff}) %**")
st.dataframe(style_df(cal_diff), width="stretch")

st.divider()
alpha_fund = st.selectbox("Select Fund for Alpha Matrix", options=manager_cols, index=0)
alpha_bench = st.selectbox("Select Benchmark for Alpha Matrix", options=manager_cols, index=len(manager_cols)-1)
st.write(f"**Alpha Over benchmark (Yearly View): {alpha_fund} vs {alpha_bench} (%)**")

alpha_matrix = pd.DataFrame(index=cleaned_df.index)
for y in range(1, 21):
    f_r = cleaned_df[alpha_fund].rolling(window=y*12).apply(lambda x: (np.prod(x + 1)**(12/len(x)) - 1) * 100)
    b_r = cleaned_df[alpha_bench].rolling(window=y*12).apply(lambda x: (np.prod(x + 1)**(12/len(x)) - 1) * 100)
    alpha_matrix[f"{y}Y"] = f_r - b_r

alpha_matrix['Date'] = cleaned_df['Date']
alpha_disp_year = alpha_matrix.groupby(alpha_matrix['Date'].dt.year).tail(1).copy()
alpha_disp_year.set_index(alpha_disp_year['Date'].dt.year, inplace=True)
alpha_disp_year = alpha_disp_year.drop(columns=['Date']).sort_index(ascending=False)
st.dataframe(style_df(alpha_disp_year), width="stretch", height=600)

st.divider()
st.subheader("36-Monthly Returns of the selected funds")
m_36_ui = cleaned_df.set_index('Date')[manager_cols].tail(36).iloc[::-1]
m_36_ui.index = m_36_ui.index.strftime('%b-%Y')
st.dataframe(style_df(m_36_ui * 100), width="stretch", height=1300)

# =====================================================
# 5. AI STRATEGIC CONVERSATION
# =====================================================
st.divider()
st.subheader("Jarir AI Strategic Advisor")

full_results_text = ""
for metric_name, df_res in results.items():
    full_results_text += f"\n--- {metric_name} ---\n{df_res.to_string()}\n"

chat_context = f"""
Portfolio Window: {start_label} to {end_label}
Managers: {', '.join(manager_cols)}
Alpha Check: {alpha_fund} vs {alpha_bench}
[DATASET]\n{full_results_text}\n{alpha_disp_year.to_string()}
"""

with st.expander("Activate AI Conversation", expanded=False):
    key_input = st.text_input("Enter OpenAI API Key", type="password")
    
    if st.button(" Generate insights"):
        if not key_input:
            st.error("Please provide an API key.")
        else:
            with st.spinner("Generating Insights..."):
                try:
                    client = openai.OpenAI(api_key=key_input)
                    sys_role = "You are acting as a senior investment analyst performing a comprehensive evaluation of fund manager performance for institutional portfolio decisions. I will provide manager-level performance data including returns, volatility, upside and downside deviations, Sharpe and Sortino ratios, drawdowns, capture ratios, calendar returns, alpha metrics, and risk-return visualizations. Your objective is to interpret the data the way an experienced investment analyst would—by identifying what is really driving performance, when it occurred, and whether it is repeatable. Examine whether returns are generated through genuine manager skill or through exposure to favorable market regimes, elevated risk-taking, or beta concentration. Assess return consistency across time by identifying periods of performance concentration, regime dependence, and month-year-specific inflection points such as market stress, drawdowns, recoveries, or rallies. Evaluate downside risk by analyzing drawdowns, downside deviation, recovery speed, and Sortino behavior to determine capital preservation capability. Analyze upside versus downside capture to understand payoff asymmetry and to distinguish convex return profiles from leveraged or directional exposure. Use the risk-return positioning to identify efficiency, dominance, and risk-adjusted attractiveness relative to peers. Leverage alpha and relative metrics to test persistence, robustness, and benchmark independence, flagging statistically fragile or benchmark-hugging strategies. Synthesize these findings into actionable conclusions by classifying managers as core, satellite, tactical, or unsuitable; identifying complementary pairings based on risk and asymmetry; highlighting red flags and monitoring triggers; and presenting insights in clear, decision-focused language suitable for investment committee review, explicitly referencing relevant time periods and market contexts rather than relying on generic performance summaries."
                    user_prompt = f"Using this EXACT DATA: {chat_context}\n\nTask: You are acting as a senior investment analyst performing a comprehensive evaluation of fund manager performance for institutional portfolio decisions. I will provide manager-level performance data including returns, volatility, upside and downside deviations, Sharpe and Sortino ratios, drawdowns, capture ratios, calendar returns, alpha metrics, and risk-return visualizations. Your objective is to interpret the data the way an experienced investment analyst would—by identifying what is really driving performance, when it occurred, and whether it is repeatable. Examine whether returns are generated through genuine manager skill or through exposure to favorable market regimes, elevated risk-taking, or beta concentration. Assess return consistency across time by identifying periods of performance concentration, regime dependence, and month-year-specific inflection points such as market stress, drawdowns, recoveries, or rallies. Evaluate downside risk by analyzing drawdowns, downside deviation, recovery speed, and Sortino behavior to determine capital preservation capability. Analyze upside versus downside capture to understand payoff asymmetry and to distinguish convex return profiles from leveraged or directional exposure. Use the risk-return positioning to identify efficiency, dominance, and risk-adjusted attractiveness relative to peers. Leverage alpha and relative metrics to test persistence, robustness, and benchmark independence, flagging statistically fragile or benchmark-hugging strategies. Synthesize these findings into actionable conclusions by classifying managers as core, satellite, tactical, or unsuitable; identifying complementary pairings based on risk and asymmetry; highlighting red flags and monitoring triggers; and presenting insights in clear, decision-focused language suitable for investment committee review, explicitly referencing relevant time periods and market contexts rather than relying on generic performance summaries."
                    response = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "system", "content": sys_role}, {"role": "user", "content": user_prompt}])
                    st.session_state.insights = response.choices[0].message.content
                    st.session_state.chat_history = [{"role": "assistant", "content": st.session_state.insights}]
                except Exception as e: st.error(f"AI Error: {e}")

    if 'insights' in st.session_state:
        st.write(st.session_state.insights)
        st.divider()
        st.write(" **Chat with Jarir AI to get more insights:**")
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if user_query := st.chat_input("Ask about the data..."):
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"): st.write(user_query)
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("🤓 Investment Analyst is thinking...")
                try:
                    client = openai.OpenAI(api_key=key_input)
                    chat_resp = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "system", "content": f"Senior Analyst. Context: {chat_context}"}, *st.session_state.chat_history])
                    reply = chat_resp.choices[0].message.content
                    thinking_placeholder.empty()
                    st.write(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as e: st.error(f"Chat Error: {e}")

# =====================================================
# 6. PDF EXPORT (A4 OPTIMIZED)
# =====================================================
def get_pdf_img(fig):
    img_buf = io.BytesIO(); fig.savefig(img_buf, format='png', bbox_inches='tight'); img_buf.seek(0); plt.close(fig); return img_buf

def generate_pdf(res, df, mgrs, b_name, start_l, end_l, cal_ret_df, cal_diff_df, alpha_df, a_fund, a_bench):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=20, rightMargin=20, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet(); elems = []
    page_width = landscape(A4)[0] - 40

    def add_table_pdf(df_p, title, is_pct=True):
        elems.append(Paragraph(title, styles['Heading2']))
        num_cols = len(df_p.columns) + 1
        font_size = 7 if num_cols <= 10 else 5.5
        index_w = 80 if num_cols <= 10 else 60
        col_widths = [index_w] + [(page_width - index_w) / (num_cols - 1)] * (num_cols - 1)
        data = [["Manager"] + list(df_p.columns)]
        style = [('BACKGROUND',(0,0),(-1,0),colors.HexColor("#4F6228")), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke), 
                 ('GRID',(0,0),(-1,-1),0.5,colors.grey), ('FONTSIZE',(0,0),(-1,-1), font_size), ('ALIGN',(0,0),(-1,-1),'CENTER')]
        
        for r_i, (idx, row) in enumerate(df_p.iterrows(), 1):
            row_vals = [str(idx)]
            for c_i, v in enumerate(row, 1):
                # NAN FIX: Handle empty strings for NaN in PDF
                if pd.isna(v):
                    row_vals.append("")
                else:
                    fmt = "{:.2f}%" if is_pct else "{:.2f}"
                    row_vals.append(fmt.format(v))
                    if v < 0: style.append(('TEXTCOLOR',(c_i, r_i),(c_i, r_i),colors.red))
            data.append(row_vals)
        t = LongTable(data, repeatRows=1, colWidths=col_widths); t.setStyle(TableStyle(style)); elems.append(t); elems.append(Spacer(1,15))

    elems.append(Paragraph(f"Jarir Investments Quant Performance Report: {start_l} to {end_l}", styles['Title']))
    add_table_pdf(res["Annualized Return (%)"], "Annualized Return (%)"); add_table_pdf(res["Annualized Volatility (%)"], "Annualized Volatility (%)"); elems.append(PageBreak())
    add_table_pdf(res["Upward Deviation (%)"], "Upward Deviation (%)"); add_table_pdf(res["Downward Deviation (%)"], "Downward Deviation (%)"); elems.append(PageBreak())
    add_table_pdf(res["Sharpe Ratio"], "Sharpe Ratio", False); add_table_pdf(res["Sortino Ratio"], "Sortino Ratio", False); elems.append(PageBreak())
    add_table_pdf(res["Max Drawdown (%)"], "Max Drawdown (%)"); elems.append(PageBreak())

    for yr in [3, 5, 8]:
        lbl = f"{yr} Year"
        if lbl in res["Annualized Return (%)"].index:
            elems.append(Paragraph(f"Risk-Return Plot: {lbl}", styles['Heading2']))
            f, ax = plt.subplots(figsize=(10, 4))
            ax.scatter(res["Annualized Volatility (%)"].loc[lbl], res["Annualized Return (%)"].loc[lbl], color='teal')
            for n in mgrs: ax.annotate(n, (res["Annualized Volatility (%)"].at[lbl, n], res["Annualized Return (%)"].at[lbl, n]), fontsize=9)
            elems.append(RLImage(get_pdf_img(f), width=500, height=220)); elems.append(PageBreak())

    for yr in [3, 5, 8]:
        if len(df) >= yr*12:
            elems.append(Paragraph(f"Upside Downside Capture: {yr} Year — Benchmark: {b_name}", styles['Heading2']))
            c_l = []
            for m in mgrs:
                u, d = get_cap(df[m].tail(yr*12), df[b_name].tail(yr*12)); c_l.append({"m": m, "u": u, "d": d})
            f, ax = plt.subplots(figsize=(10, 4))
            cp_df = pd.DataFrame(c_l)
            ax.scatter(cp_df["d"], cp_df["u"], color='darkred')
            for _, r in cp_df.iterrows(): ax.annotate(r["m"], (r["d"], r["u"]), fontsize=9)
            ax.axvline(100, color='black', lw=1); ax.axhline(100, color='black', lw=1)
            elems.append(RLImage(get_pdf_img(f), width=500, height=220)); elems.append(PageBreak())

    for yr in [3, 5, 8, 10]:
        if len(df) >= yr*12: add_table_pdf(df[mgrs].tail(yr*12).corr(), f"Correlation Matrix: {yr} Year", False); elems.append(PageBreak())

    add_table_pdf(cal_ret, "Calendar Year Returns"); add_table_pdf(cal_diff, f"Calendar Difference vs {bench_diff} "); elems.append(PageBreak())
    add_table_pdf(alpha_df, f"Alpha Over Benchmark on selected funds: {a_fund} vs {a_bench}"); elems.append(PageBreak())
    m_36_p = df.set_index('Date')[mgrs].tail(36).iloc[::-1]
    m_36_p.index = m_36_p.index.strftime('%b-%Y')
    add_table_pdf(m_36_p*100, "Trailing 36-Month Returns (%)")
    doc.build(elems); buf.seek(0); return buf

st.sidebar.divider()
if st.sidebar.button(" Generate Final Master PDF"):
    with st.spinner("Generating 20-Page Professional Report..."):
        p_buf = generate_pdf(results, cleaned_df, manager_cols, b_cap, start_label, end_label, cal_ret, cal_diff, alpha_disp_year, alpha_fund, alpha_bench)
        today_date = dt.date.today().strftime("%Y-%m-%d")
        file_name_final = f"{today_date}_jarir_quant analysis.pdf"
        st.sidebar.download_button(" Download PDF", data=p_buf, file_name=file_name_final)