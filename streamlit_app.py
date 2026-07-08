"""
streamlit_app.py — 台股 DCF 估值工作台(多公司版)

流程:輸入台股代號 → 自動抓 FinMind 財報 → 產生歷史指標表與建議假設
     → 使用者微調 → DCF 估值 → 已載入公司可隨時切換、底部跨公司對照。

repo 結構(Streamlit Cloud 部署):
  streamlit_app.py / dcf_engine.py / data.py / requirements.txt 同層即可。
"""

import streamlit as st
import pandas as pd
import numpy as np

from dcf_engine import DCFAssumptions, DCFModel
import data as D

st.set_page_config(page_title="台股 DCF 工作台", page_icon="📈", layout="wide")
st.title("📈 台股 DCF 估值工作台")
st.caption("輸入代號自動抓取財報 → 建議假設 → 估值。單位:億元 / 億股 / 元。教學用途,非投資建議。")

ss = st.session_state
ss.setdefault("companies", {})     # {ticker: load_company() 的結果}
ss.setdefault("results", {})       # {ticker: 最後一次估值摘要}


def get_finmind_token() -> str:
    """從 st.secrets 讀 FinMind token。沒設定 secrets 檔也不會壞,回空字串(匿名使用)。"""
    try:
        return st.secrets.get("FINMIND_TOKEN", "")
    except Exception:  # 本機沒有 .streamlit/secrets.toml 時 st.secrets 會拋例外
        return ""


# ──────────────────────────────────────────────────────────────
# 載入公司
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def cached_load(ticker: str, token: str):
    return D.load_company(ticker, token)

with st.sidebar:
    st.header("資料來源")
    token = get_finmind_token()
    if token:
        st.success("已從 secrets 讀取 FinMind token")
    else:
        st.warning("未設定 FinMind token —— 可匿名使用,但流量上限低,多載幾家易被擋。")
        st.caption(
            "設定方式:\n"
            "• Streamlit Cloud:App → Settings → Secrets 貼上\n"
            "  `FINMIND_TOKEN = \"你的token\"`\n"
            "• 本機:專案內建立 `.streamlit/secrets.toml`,同樣寫一行"
        )

c1, c2 = st.columns([3, 1])
ticker_in = c1.text_input("台股代號(例:2308、2330、3017)", value="", placeholder="輸入代號後按載入")
if c2.button("載入財報", type="primary", use_container_width=True) and ticker_in.strip():
    t = ticker_in.strip()
    try:
        with st.spinner(f"抓取 {t} 財報中…"):
            ss.companies[t] = cached_load(t, token)
        st.success(f"{t} 載入完成")
    except Exception as e:  # noqa: BLE001
        st.error(f"載入失敗:{e}")
        st.caption("常見原因:代號錯誤、FinMind 流量上限(可填 token)、或欄位對應需調整(見下方診斷)。")

if not ss.companies:
    st.info("先在上方輸入台股代號並載入財報。載入後即可估值,已載入的公司都會留在此頁可切換。")
    st.stop()

active = st.selectbox("目前公司", list(ss.companies.keys()),
                      index=len(ss.companies) - 1)
co = ss.companies[active]
sug = co["suggest"]

for w in co.get("warnings", []):
    st.error(f"⚠️ 資料異常:{w}")


# ──────────────────────────────────────────────────────────────
# 歷史指標(建議假設的依據,讓使用者能檢查)
# ──────────────────────────────────────────────────────────────
st.subheader(f"{active} 歷史財務指標(億元 / %)")
hist_cols = [c for c in ["Revenue", "RevenueGrowth%", "OpMargin%", "DA%",
                         "CapEx%", "NWC%", "TaxRate%"] if c in co["history"].columns]
st.dataframe(co["history"][hist_cols].style.format("{:,.1f}"), width="stretch")

with st.expander("診斷:本次抓到的原始科目(欄位對不上時看這裡)"):
    st.write(co["diagnostics"])


# ──────────────────────────────────────────────────────────────
# 假設(以建議值預填;widget key 綁 ticker → 各公司各自記住修改)
# ──────────────────────────────────────────────────────────────
st.subheader("估值假設(已用歷史資料預填,可修改)")

k = lambda name: f"{name}_{active}"
a1, a2, a3, a4 = st.columns(4)
wacc = a1.number_input("WACC (%)", value=9.0, min_value=0.1, step=0.1, key=k("wacc")) / 100
tg   = a2.number_input("永續成長率 g (%)", value=2.5, step=0.1, key=k("tg")) / 100
tax  = a3.number_input("稅率 (%)", value=float(sug["tax_rate_pct"]), step=0.5, key=k("tax")) / 100
exit_mult = a4.number_input("Exit EV/EBITDA (×)", value=12.0, step=0.5, key=k("exit"))

b1, b2, b3, b4 = st.columns(4)
n_years = b1.slider("預測年數", 3, 10, 5, key=k("ny"))
mid_year = b2.checkbox("期中折現", value=True, key=k("mid"))
norm_tv = b3.checkbox("終值常態化 CapEx≈D&A", value=True, key=k("norm"))
price = b4.number_input("現價(元)", value=float(co["price"] or 0.0), step=1.0, key=k("px"))

c1, c2, c3 = st.columns(3)
net_debt = c1.number_input("淨負債(億元)", value=float(co["net_debt"]), step=10.0, key=k("nd"))
minority = c2.number_input("少數股權(億元)", value=0.0, step=10.0, key=k("mi"),
                           help="子公司非 100% 持股時必填,否則高估每股價值")
shares = c3.number_input("流通股數(億股)", value=float(co["shares"]),
                         min_value=0.0001, step=0.1, key=k("sh"))

st.markdown("**逐年驅動因子(%,可直接編輯)**")
def _default_drivers():
    g = sug["growth_pct"][:n_years]
    g = g + [g[-1]] * (n_years - len(g))
    return pd.DataFrame({
        "Year": range(1, n_years + 1),
        "營收成長 (%)": np.round(g, 1),
        "營業利益率 (%)": [sug["op_margin_pct"]] * n_years,
        "D&A (% rev)": [sug["da_pct"]] * n_years,
        "CapEx (% rev)": [sug["capex_pct"]] * n_years,
        "NWC (% rev)": [sug["nwc_pct"]] * n_years,
    })
drivers = st.data_editor(_default_drivers(), key=k(f"drv{n_years}"),
                         hide_index=True, width="stretch", disabled=["Year"])


# ──────────────────────────────────────────────────────────────
# 建模與結果
# ──────────────────────────────────────────────────────────────
pct = lambda col: (drivers[col] / 100).tolist()
try:
    model = DCFModel(DCFAssumptions(
        base_revenue=co["base_revenue"],
        revenue_growth=pct("營收成長 (%)"), ebit_margin=pct("營業利益率 (%)"),
        tax_rate=tax, da_pct_revenue=pct("D&A (% rev)"),
        capex_pct_revenue=pct("CapEx (% rev)"), nwc_pct_revenue=pct("NWC (% rev)"),
        wacc=wacc, terminal_growth=tg,
        net_debt=net_debt, minority_interest=minority, shares_outstanding=shares,
        base_nwc=co["base_nwc"], forecast_years=n_years,
        exit_ev_ebitda=exit_mult, mid_year_convention=mid_year,
        normalize_terminal=norm_tv,
    ))
except Exception as e:  # noqa: BLE001
    st.error(f"模型建構失敗:{e}")
    st.stop()

st.subheader("估值結果")
res_cols = st.columns(2)
summary = {}
for col, (meth, title) in zip(res_cols, [("gordon", "Gordon Growth"),
                                         ("exit_multiple", "Exit Multiple")]):
    try:
        v = model.value(meth)
    except Exception as e:  # noqa: BLE001
        col.error(f"{title}:{e}")
        continue
    col.markdown(f"### {title}")
    if price > 0:
        col.metric("每股價值(元)", f"{v['Value_per_Share']:,.0f}",
                   delta=f"{v['Value_per_Share']/price-1:+.1%} vs 現價")
    else:
        col.metric("每股價值(元)", f"{v['Value_per_Share']:,.0f}")
    col.write(f"EV **{v['Enterprise_Value']:,.0f}** 億 | 權益 **{v['Equity_Value']:,.0f}** 億 "
              f"| 終值佔 EV **{v['TV_pct_of_EV']:.0%}**")
    if v["TV_pct_of_EV"] > 0.80:
        col.warning("終值佔比 > 80%,估值高度依賴永續假設")
    summary[meth] = v["Value_per_Share"]

ss.results[active] = {
    "Gordon(元)": summary.get("gordon"),
    "Exit(元)": summary.get("exit_multiple"),
    "現價(元)": price if price > 0 else None,
    "上檔空間": (summary.get("gordon") / price - 1) if (price and summary.get("gordon")) else None,
}

with st.expander("現金流預測明細"):
    proj = model.proj[["Revenue", "EBIT", "EBITDA", "NOPAT", "D&A", "CapEx", "ΔNWC",
                       "FCFF", "PV_FCFF"]]
    st.dataframe(proj.style.format("{:,.1f}"), width="stretch")

st.subheader("敏感度:每股價值(WACC × g)")
try:
    sens = model.sensitivity(np.arange(wacc - 0.01, wacc + 0.011, 0.005),
                             np.arange(tg - 0.01, tg + 0.011, 0.005))
    st.dataframe(sens.style.format("{:,.0f}").background_gradient(cmap="RdYlGn", axis=None),
                 width="stretch")
except Exception as e:  # noqa: BLE001
    st.error(f"敏感度計算失敗:{e}")


# ──────────────────────────────────────────────────────────────
# 跨公司對照(同一頁比較所有已載入的公司)
# ──────────────────────────────────────────────────────────────
if len(ss.results) > 1:
    st.divider()
    st.subheader("跨公司對照(各公司最後一次計算)")
    comp = pd.DataFrame(ss.results).T
    st.dataframe(comp.style.format({"Gordon(元)": "{:,.0f}", "Exit(元)": "{:,.0f}",
                                    "現價(元)": "{:,.0f}", "上檔空間": "{:+.1%}"}),
                 width="stretch")

st.divider()
st.caption("資料來源:FinMind。教學/作品集用途,非投資建議。")
