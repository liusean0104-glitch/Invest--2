"""
streamlit_app.py — DCF 估值引擎的互動式前端

部署到 Streamlit Cloud:把這支檔、dcf_engine.py、requirements.txt
放在同一個 GitHub repo 根目錄,App 進入點選 streamlit_app.py 即可。
"""

import streamlit as st
import pandas as pd
import numpy as np

from dcf_engine import WACCInputs, DCFAssumptions, DCFModel

st.set_page_config(page_title="DCF 估值引擎", page_icon="📈", layout="wide")

st.title("📈 DCF 估值引擎")
st.caption("FCFF 折現模型 · Gordon Growth 與 Exit Multiple 雙終值法 · 含 WACC×g 敏感度分析")


# ──────────────────────────────────────────────────────────────
# 側邊欄:全域假設
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("假設輸入")

    st.subheader("公司結構")
    base_revenue = st.number_input("最近一年營收 (Year 0)", value=5000.0, step=100.0, format="%.1f")
    net_debt = st.number_input("淨負債（總負債 − 現金 + 少數股權）", value=1500.0, step=100.0, format="%.1f")
    shares = st.number_input("流通股數", value=1000.0, step=10.0, min_value=0.0001, format="%.1f")
    tax_rate = st.number_input("稅率 (%)", value=20.0, min_value=0.0, max_value=60.0, step=1.0) / 100
    n_years = st.slider("明確預測期（年）", 3, 10, 5)

    st.subheader("折現率 WACC")
    wacc_mode = st.radio("WACC 來源", ["直接輸入", "用 CAPM 建構"], horizontal=True)
    if wacc_mode == "直接輸入":
        wacc = st.number_input("WACC (%)", value=8.84, min_value=0.1, step=0.1) / 100
    else:
        rf = st.number_input("無風險利率 rf (%)", value=4.0, step=0.1) / 100
        beta = st.number_input("Beta（levered）", value=1.10, step=0.05)
        erp = st.number_input("市場風險溢酬 ERP (%)", value=5.5, step=0.1) / 100
        kd = st.number_input("稅前債務成本 kd (%)", value=5.0, step=0.1) / 100
        mcap = st.number_input("股權市值 E", value=8000.0, step=100.0)
        debt = st.number_input("總負債 D", value=2000.0, step=100.0, min_value=0.0)
        w_in = WACCInputs(risk_free_rate=rf, beta=beta, equity_risk_premium=erp,
                          cost_of_debt_pretax=kd, tax_rate=tax_rate,
                          market_cap=mcap, total_debt=debt)
        wacc = w_in.wacc()
        st.caption(f"Cost of Equity：{w_in.cost_of_equity():.2%}　|　**WACC：{wacc:.2%}**")

    st.subheader("終值")
    terminal_growth = st.number_input("永續成長率 g (%)", value=2.5, step=0.1) / 100
    exit_mult = st.number_input("Exit EV/EBITDA (×)", value=12.0, min_value=0.0, step=0.5)
    normalize_terminal = st.checkbox(
        "終值常態化（CapEx ≈ D&A）", value=True,
        help="關閉會把擴張期的高 CapEx 永遠帶進終值,通常系統性低估 TV")
    mid_year = st.checkbox("期中折現慣例（mid-year）", value=False)

    st.subheader("市價對照（選填）")
    current_price = st.number_input("目前股價（留 0 表示不比較）", value=0.0, min_value=0.0, step=1.0)


# ──────────────────────────────────────────────────────────────
# 主畫面:逐年營運驅動因子（可編輯表格）
# ──────────────────────────────────────────────────────────────
def default_drivers(n: int) -> pd.DataFrame:
    g = np.linspace(12, 5, n)  # 成長率預設從 12% 漸降到 5%
    return pd.DataFrame({
        "Year": list(range(1, n + 1)),
        "營收成長 (%)": np.round(g, 1),
        "EBIT margin (%)": [22.0] * n,
        "D&A (% rev)": [5.0] * n,
        "CapEx (% rev)": [6.0] * n,
        "NWC (% rev)": [10.0] * n,
    })

st.subheader("逐年營運假設")
st.caption("直接點選表格即可編輯每一年的驅動因子（單位皆為 %）")
drivers = st.data_editor(
    default_drivers(n_years),
    key=f"drivers_{n_years}",
    hide_index=True,
    width='stretch',
    disabled=["Year"],
)


# ──────────────────────────────────────────────────────────────
# 建模
# ──────────────────────────────────────────────────────────────
def pct(col):
    return (drivers[col] / 100).tolist()

assumptions = DCFAssumptions(
    base_revenue=base_revenue,
    revenue_growth=pct("營收成長 (%)"),
    ebit_margin=pct("EBIT margin (%)"),
    tax_rate=tax_rate,
    da_pct_revenue=pct("D&A (% rev)"),
    capex_pct_revenue=pct("CapEx (% rev)"),
    nwc_pct_revenue=pct("NWC (% rev)"),
    wacc=wacc,
    terminal_growth=terminal_growth,
    net_debt=net_debt,
    shares_outstanding=shares,
    forecast_years=n_years,
    exit_ev_ebitda=exit_mult,
    mid_year_convention=mid_year,
    normalize_terminal=normalize_terminal,
)

try:
    model = DCFModel(assumptions)
except Exception as e:  # noqa: BLE001
    st.error(f"模型建構失敗：{e}")
    st.stop()


# ──────────────────────────────────────────────────────────────
# 結果
# ──────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("WACC", f"{wacc:.2%}")
m2.metric("永續成長率 g", f"{terminal_growth:.2%}")
m3.metric("預測期", f"{n_years} 年")

st.subheader("現金流預測")
proj = model.proj[["Revenue", "EBIT", "EBITDA", "NOPAT", "D&A",
                   "CapEx", "ΔNWC", "FCFF", "DiscFactor", "PV_FCFF"]]
fmt = {c: "{:,.1f}" for c in proj.columns}
fmt["DiscFactor"] = "{:.3f}"
st.dataframe(proj.style.format(fmt), width='stretch')

st.subheader("估值結果")

def render_value(col, method: str, title: str):
    try:
        v = model.value(method)
    except Exception as e:  # noqa: BLE001
        col.error(f"{title}：{e}")
        return
    col.markdown(f"### {title}")
    if current_price > 0:
        upside = v["Value_per_Share"] / current_price - 1
        col.metric("每股價值", f"{v['Value_per_Share']:,.2f}",
                   delta=f"{upside:+.1%} vs 現價")
    else:
        col.metric("每股價值", f"{v['Value_per_Share']:,.2f}")
    col.write(f"企業價值 EV：**{v['Enterprise_Value']:,.0f}**")
    col.write(f"權益價值：**{v['Equity_Value']:,.0f}**")
    col.write(f"終值佔 EV：**{v['TV_pct_of_EV']:.1%}**")
    if v["TV_pct_of_EV"] > 0.80:
        col.warning("終值佔比 > 80%，估值高度依賴永續假設")

cG, cE = st.columns(2)
render_value(cG, "gordon", "Gordon Growth")
render_value(cE, "exit_multiple", "Exit Multiple")

st.subheader("敏感度分析：每股價值（WACC × g）")
wacc_grid = np.arange(wacc - 0.01, wacc + 0.011, 0.005)
g_grid = np.arange(terminal_growth - 0.01, terminal_growth + 0.011, 0.005)
try:
    sens = model.sensitivity(wacc_grid, g_grid, "gordon")
    st.dataframe(
        sens.style.format("{:,.1f}").background_gradient(cmap="RdYlGn", axis=None),
        width='stretch',
    )
except Exception as e:  # noqa: BLE001
    st.error(f"敏感度計算失敗：{e}")

st.subheader("價值組成（PV 拆解）")
try:
    gv = model.value("gordon")
    comp = pd.DataFrame(
        {"PV": list(model.proj["PV_FCFF"].values) + [gv["PV_of_TV"]]},
        index=[f"Y{y}" for y in model.proj.index] + ["終值 TV"],
    )
    st.bar_chart(comp)
except Exception:  # noqa: BLE001
    pass

st.divider()
st.caption("教學/作品集用途,非投資建議。所有假設可於左側與表格中自行調整。")
