"""
streamlit_app.py — 台股 DCF 估值工作台(投行級版面)

流程:輸入台股代號 → 自動抓 FinMind 財報 → 歷史指標與建議假設
     → CAPM 推導 WACC → DCF 估值(Gordon / Exit 交叉檢核)
     → Football Field、EV→權益 bridge、情境與敏感度分析。

repo 結構(Streamlit Cloud 部署):
  streamlit_app.py / dcf_engine.py / data.py / requirements.txt 同層即可。
"""

import dataclasses

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

import dcf_engine as dcf_engine_module
from dcf_engine import DCFAssumptions, DCFModel, WACCInputs
import data as D

st.set_page_config(page_title="台股 DCF 工作台", page_icon="📈", layout="wide")

# ── JPMorganChase 風格主題:米白底、黑色襯線字、白色分欄搜尋框 ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@600;700;900&display=swap');

/* 整頁米白底 */
.stApp { background-color: #F2EEE4; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background-color: #ECE6D8; }

/* 標題與品牌字:黑色襯線 */
h1, h2, h3, .brandmark,
[data-testid="stMetricLabel"] {
  font-family: "Noto Serif TC", Georgia, "Times New Roman", serif !important;
  color: #141414;
}
.brandmark {
  text-align: center; font-size: 3.3rem; font-weight: 900;
  letter-spacing: 0.005em; line-height: 1.15; margin: 1.2rem 0 0.15rem 0;
}
.brand-sub {
  text-align: center; letter-spacing: 0.38em; font-size: 0.78rem;
  color: #6E675C; margin-bottom: 2.2rem; text-transform: uppercase;
}

/* 搜尋框內的小型全大寫標籤 */
.search-label {
  font-size: 0.72rem; letter-spacing: 0.16em; font-weight: 700;
  color: #141414; margin: 0.55rem 0 0 0.15rem; text-transform: uppercase;
  font-family: Georgia, serif;
}

/* 白色分欄搜尋框(仿 JPMC Find Jobs bar) */
.st-key-jpmc_search {
  background: #FFFFFF; border: 1px solid #C8C1B2;
  padding: 0.35rem 0.35rem 0.85rem 1.2rem; margin-bottom: 1.4rem;
}
.st-key-jpmc_search [data-testid="stColumn"]:nth-of-type(2) {
  border-left: 1px solid #C8C1B2; padding-left: 1.2rem;
}
.st-key-jpmc_search [data-testid="stColumn"]:nth-of-type(3) {
  display: flex; align-items: stretch;
}
/* 內部輸入元件去邊框、透明底 */
.st-key-jpmc_search [data-baseweb="input"],
.st-key-jpmc_search [data-baseweb="select"] > div,
.st-key-jpmc_search input {
  border: none !important; box-shadow: none !important;
  background: transparent !important; font-size: 1.12rem !important;
}
.st-key-jpmc_search [data-testid="stTextInput"] > div,
.st-key-jpmc_search [data-testid="stSelectbox"] > div {
  border: none !important; box-shadow: none !important; background: transparent !important;
}
/* 放大鏡按鈕:米色方塊 + 棕色圖示 */
.st-key-jpmc_search .stButton > button {
  background: #E7E0CE !important; color: #8A5A32 !important;
  border: none !important; border-radius: 0 !important;
  height: 3.6rem; width: 100%; font-size: 1.35rem; margin-top: 0.55rem;
}
.st-key-jpmc_search .stButton > button:hover { background: #DFD6BF !important; }

/* 分頁籤與主要按鈕統一風格 */
.stTabs [data-baseweb="tab"] {
  font-family: "Noto Serif TC", Georgia, serif; letter-spacing: 0.04em;
}
.stButton > button[kind="primary"] {
  background: #141414; color: #FFFFFF; border-radius: 0; border: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="brandmark">台股 DCF 估值工作台</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Taiwan Equity Valuation Workbench</div>', unsafe_allow_html=True)
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
    st.divider()
    with st.expander("🔧 引擎版本診斷"):
        st.caption(f"dcf_engine 載入路徑:\n`{dcf_engine_module.__file__}`")
        _f = {f.name for f in dataclasses.fields(DCFAssumptions)}
        if "non_operating_assets" in _f:
            st.success("dcf_engine.py 為最新版(含 non_operating_assets)")
        else:
            st.error("dcf_engine.py 為舊版!請 push 最新檔並 Reboot app")
    if st.button("🔄 清除快取並重抓資料", width="stretch",
                 help="data.py 邏輯更新後 st.cache_data 不會自動失效,改完程式務必按一次"):
        cached_load.clear()
        ss.companies, ss.results = {}, {}
        st.rerun()

# ── 仿 JPMC 搜尋列:左「輸入代號」/ 中「已載入公司」/ 右 放大鏡載入 ──
with st.container(key="jpmc_search"):
    col_a, col_b, col_c = st.columns([5, 4, 1])
    with col_a:
        st.markdown('<div class="search-label">Find Ticker</div>', unsafe_allow_html=True)
        ticker_in = st.text_input("台股代號", value="", label_visibility="collapsed",
                                  placeholder="台股代號:2308、2330、3017 …")
    with col_b:
        st.markdown('<div class="search-label">Loaded Companies ▾</div>', unsafe_allow_html=True)
        _opts = list(ss.companies.keys())
        if _opts:
            active = st.selectbox("目前公司", _opts, index=len(_opts) - 1,
                                  label_visibility="collapsed")
        else:
            st.selectbox("目前公司", ["尚未載入公司"], disabled=True,
                         label_visibility="collapsed")
            active = None
    with col_c:
        do_load = st.button("🔍", key="load_btn", help="載入財報")

if do_load and ticker_in.strip():
    t = ticker_in.strip()
    try:
        with st.spinner(f"抓取 {t} 財報中…"):
            ss.companies[t] = cached_load(t, token)
        st.success(f"{t} 載入完成")
        active = t
    except Exception as e:  # noqa: BLE001
        st.error(f"載入失敗:{e}")
        st.caption("常見原因:代號錯誤、FinMind 流量上限(可填 token)、或欄位對應需調整(見診斷分頁)。")

if not ss.companies:
    st.info("先在上方搜尋列輸入台股代號並按 🔍 載入財報。載入後即可估值,已載入的公司都會留在下拉選單可切換。")
    st.stop()

if active is None or active not in ss.companies:
    active = list(ss.companies.keys())[-1]
co = ss.companies[active]
sug = co["suggest"]
k = lambda name: f"{name}_{active}"

for w in co.get("warnings", []):
    st.error(f"⚠️ 資料異常:{w}")

tab_hist, tab_assume, tab_value, tab_sens = st.tabs(
    ["📊 歷史資料", "⚙️ 估值假設", "💰 估值結果", "🎯 敏感度與情境"])


# ──────────────────────────────────────────────────────────────
# Tab 1:歷史指標(建議假設的依據,讓使用者能檢查)
# ──────────────────────────────────────────────────────────────
with tab_hist:
    st.subheader(f"{active} 歷史財務指標(億元 / %)")
    hist_cols = [c for c in ["Revenue", "RevenueGrowth%", "OpMargin%", "DA%",
                             "CapEx%", "NWC%", "TaxRate%"] if c in co["history"].columns]
    st.dataframe(co["history"][hist_cols].style.format("{:,.1f}"), width="stretch")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("現價(元)", f"{co['price']:,.1f}" if co.get("price") else "—")
    m2.metric("流通股數(億股)", f"{co['shares']:,.2f}")
    m3.metric("淨負債・不含租賃(億元)", f"{co['net_debt']:,.0f}")
    m4.metric("租賃負債 IFRS 16(億元)", f"{co.get('lease_liab', 0):,.0f}")

    with st.expander("診斷:本次抓到的原始科目(欄位對不上時看這裡)"):
        st.write(co["diagnostics"])


# ──────────────────────────────────────────────────────────────
# Tab 2:假設(以建議值預填;widget key 綁 ticker → 各公司各自記住修改)
# ──────────────────────────────────────────────────────────────
with tab_assume:
    # ── WACC:直接輸入或 CAPM 推導 ──
    st.subheader("折現率(WACC)")
    wacc_mode = st.radio("WACC 決定方式", ["CAPM 推導", "直接輸入"],
                         horizontal=True, key=k("wmode"))
    if wacc_mode == "直接輸入":
        wacc = st.number_input("WACC (%)", value=9.0, min_value=0.1, step=0.1,
                               key=k("wacc")) / 100
    else:
        w1, w2, w3, w4 = st.columns(4)
        rf   = w1.number_input("無風險利率 rf (%)", value=1.6, step=0.1, key=k("rf"),
                               help="台灣 10 年期公債殖利率") / 100
        beta = w2.number_input("Beta(槓桿後)", value=0.90, step=0.05, key=k("beta"))
        erp  = w3.number_input("市場風險溢酬 ERP (%)", value=5.5, step=0.1, key=k("erp")) / 100
        kd   = w4.number_input("稅前債務成本 kd (%)", value=2.5, step=0.1, key=k("kd")) / 100

        default_e = float((co.get("price") or 0) * co["shares"]) or 1000.0
        default_d = float(co.get("total_debt", 0) + co.get("lease_liab", 0))
        w5, w6, w7 = st.columns(3)
        mcap = w5.number_input("股權市值 E(億元)", value=round(default_e, 0),
                               min_value=1.0, step=100.0, key=k("mcap"))
        debt = w6.number_input("債務市值 D(億元)", value=round(default_d, 0),
                               min_value=0.0, step=50.0, key=k("dval"),
                               help="無市價時以帳面值近似;含租賃負債")
        tax_wacc = w7.number_input("邊際稅率 (%)", value=20.0, step=0.5, key=k("wtax")) / 100

        wi = WACCInputs(risk_free_rate=rf, beta=beta, equity_risk_premium=erp,
                        cost_of_debt_pretax=kd, tax_rate=tax_wacc,
                        market_cap=mcap, total_debt=max(debt, 0.0))
        wacc = wi.wacc()
        st.info(f"ke (CAPM) = {wi.cost_of_equity():.2%} | 稅後 kd = {wi.cost_of_debt_aftertax():.2%} "
                f"| E/V = {mcap/(mcap+debt):.0%} → **WACC = {wacc:.2%}**")

    # ── 終值與其他核心假設 ──
    st.subheader("終值與稅務")
    a1, a2, a3, a4 = st.columns(4)
    tg   = a1.number_input("永續成長率 g (%)", value=2.5, step=0.1, key=k("tg"),
                           help="上限應貼近長期名目 GDP 成長") / 100
    tax  = a2.number_input("有效稅率 (%)", value=float(sug["tax_rate_pct"]), step=0.5,
                           key=k("tax")) / 100
    exit_mult = a3.number_input("Exit EV/EBITDA (×)", value=12.0, step=0.5, key=k("exit"),
                                help="應以同業可比公司的交易倍數為錨")
    ronic_in = a4.number_input("永續期 RONIC (%)(0 = 不使用)", value=0.0, step=0.5,
                               key=k("ronic"),
                               help="給定則以 Damodaran 再投資率法算終值 FCFF,"
                                    "優先於 CapEx≈D&A 常態化")

    b1, b2, b3, b4 = st.columns(4)
    n_years  = b1.slider("預測年數", 3, 10, 5, key=k("ny"))
    mid_year = b2.checkbox("期中折現", value=True, key=k("mid"))
    norm_tv  = b3.checkbox("終值常態化 CapEx≈D&A", value=True, key=k("norm"))
    price    = b4.number_input("現價(元)", value=float(co["price"] or 0.0), step=1.0, key=k("px"))

    # ── EV → 權益 bridge 項目 ──
    st.subheader("EV → 權益 bridge")
    lease = float(co.get("lease_liab", 0.0))
    incl_lease = st.checkbox(
        f"淨負債含租賃負債 IFRS 16(+{lease:,.0f} 億)", value=lease > 0, key=k("lease"),
        help="IFRS 16 後租金移出營業費用 → EBIT/EBITDA 被墊高。"
             "分子(現金流)已含租賃利益,分母端的淨負債就必須含租賃負債,否則高估權益。")
    nd_default = float(co["net_debt"]) + (lease if incl_lease else 0.0)

    c1_, c2_, c3_, c4_ = st.columns(4)
    net_debt = c1_.number_input("淨負債(億元)", value=nd_default, step=10.0,
                                key=k(f"nd{int(incl_lease)}"))
    minority = c2_.number_input("少數股權(億元)", value=0.0, step=10.0, key=k("mi"),
                                help="子公司非 100% 持股時必填,否則高估每股價值")
    preferred = c3_.number_input("特別股(億元)", value=0.0, step=10.0, key=k("pf"))
    non_op = c4_.number_input("非營運資產(億元)", value=0.0, step=10.0, key=k("nonop"),
                              help="長期股權投資、閒置土地等:收益不在 FCFF 內 → 須加回")
    shares = st.number_input("流通股數(億股)", value=float(co["shares"]),
                             min_value=0.0001, step=0.1, key=k("sh"),
                             help="嚴謹版應用完全稀釋股數(員工認股權、可轉債)")

    # ── 逐年驅動因子 ──
    st.subheader("逐年驅動因子(%,可直接編輯)")
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
# 建模(在分頁外執行,結果分頁與敏感度分頁共用)
# ──────────────────────────────────────────────────────────────
pct = lambda col: (drivers[col] / 100).tolist()
base_kwargs = dict(
    base_revenue=co["base_revenue"],
    revenue_growth=pct("營收成長 (%)"), ebit_margin=pct("營業利益率 (%)"),
    tax_rate=tax, da_pct_revenue=pct("D&A (% rev)"),
    capex_pct_revenue=pct("CapEx (% rev)"), nwc_pct_revenue=pct("NWC (% rev)"),
    wacc=wacc, terminal_growth=tg,
    net_debt=net_debt, minority_interest=minority, preferred_equity=preferred,
    non_operating_assets=non_op, shares_outstanding=shares,
    base_nwc=co["base_nwc"], forecast_years=n_years,
    exit_ev_ebitda=exit_mult, mid_year_convention=mid_year,
    normalize_terminal=norm_tv,
    terminal_ronic=(ronic_in / 100) if ronic_in > 0 else None,
)

# ── 版本防呆:部署環境的 dcf_engine.py 若是舊版(缺欄位),
#    不整頁崩潰,改為「略過該參數 + 明確告警」,並指出載入的檔案路徑。──
_engine_fields = {f.name for f in dataclasses.fields(DCFAssumptions)}
_dropped = sorted(set(base_kwargs) - _engine_fields)
if _dropped:
    st.error(
        f"⚠️ 偵測到舊版 dcf_engine.py:缺少欄位 {', '.join(_dropped)}。"
        f"以下計算已**略過**這些參數(視為 0),結果僅供參考。\n\n"
        f"目前載入的引擎檔案:`{dcf_engine_module.__file__}`\n\n"
        f"修復:把最新 dcf_engine.py push 到 repo → Streamlit Cloud 選單 **Reboot app**。"
    )
    base_kwargs = {k_: v_ for k_, v_ in base_kwargs.items() if k_ in _engine_fields}

try:
    model = DCFModel(DCFAssumptions(**base_kwargs))
except Exception as e:  # noqa: BLE001
    st.error(f"模型建構失敗:{e}")
    st.caption(f"引擎檔案路徑:`{dcf_engine_module.__file__}`(若非 repo 內路徑,代表載入到別的舊檔)")
    st.stop()


# ──────────────────────────────────────────────────────────────
# Tab 3:估值結果 + 交叉檢核 + Football Field + Bridge
# ──────────────────────────────────────────────────────────────
with tab_value:
    res_cols = st.columns(2)
    summary, values = {}, {}
    for col, (meth, title) in zip(res_cols, [("gordon", "Gordon Growth"),
                                             ("exit_multiple", "Exit Multiple")]):
        try:
            v = model.value(meth)
        except Exception as e:  # noqa: BLE001
            col.error(f"{title}:{e}")
            continue
        values[meth] = v
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
        # 交叉檢核:各法終值反推另一法的隱含參數
        if meth == "gordon":
            col.caption(f"↔ 隱含 Exit EV/EBITDA:**{model.implied_exit_multiple():.1f}×**"
                        f"(vs 你輸入的 {exit_mult:.1f}×)")
        else:
            g_imp = model.implied_terminal_growth()
            col.caption("↔ 隱含永續成長率 g:" +
                        (f"**{g_imp:.2%}**(vs 你輸入的 {tg:.2%})" if g_imp is not None
                         else "無解(倍數與 WACC 假設下 Gordon 無法達到此終值)"))
        summary[meth] = v["Value_per_Share"]

    # 兩法差距過大 → 明確提示
    if len(summary) == 2:
        lo_, hi_ = sorted(summary.values())
        if hi_ / lo_ - 1 > 0.25:
            st.warning(f"兩種終值法差距 {hi_/lo_-1:.0%} —— 代表 (WACC − g) 隱含的倍數"
                       f"與你輸入的 Exit multiple 不一致。看上方「隱含指標」判斷哪邊假設偏離市場。")

    ss.results[active] = {
        "Gordon(元)": summary.get("gordon"),
        "Exit(元)": summary.get("exit_multiple"),
        "現價(元)": price if price > 0 else None,
        "上檔空間": (summary.get("gordon") / price - 1) if (price and summary.get("gordon")) else None,
    }

    # ── Football Field ──
    st.subheader("Football Field:估值區間 vs 現價")
    try:
        sens_g = model.sensitivity(np.arange(wacc - 0.01, wacc + 0.011, 0.005),
                                   np.arange(tg - 0.005, tg + 0.0051, 0.005))
        sens_x = model.sensitivity_exit(np.arange(wacc - 0.01, wacc + 0.011, 0.005),
                                        np.arange(exit_mult - 2, exit_mult + 2.1, 1.0))
        bands = [
            {"方法": "DCF – Gordon(WACC ±1%, g ±0.5%)",
             "low": float(np.nanmin(sens_g.values)), "high": float(np.nanmax(sens_g.values))},
            {"方法": "DCF – Exit Multiple(WACC ±1%, ±2×)",
             "low": float(np.nanmin(sens_x.values)), "high": float(np.nanmax(sens_x.values))},
        ]
        if co.get("price_52w"):
            lo52, hi52 = co["price_52w"]
            bands.append({"方法": "52 週股價區間", "low": lo52, "high": hi52})
        ff = pd.DataFrame(bands)
        bars = alt.Chart(ff).mark_bar(height=26, cornerRadius=4).encode(
            x=alt.X("low:Q", title="每股價值(元)"), x2="high:Q",
            y=alt.Y("方法:N", title=None, sort=None),
            color=alt.Color("方法:N", legend=None),
            tooltip=[alt.Tooltip("方法:N"),
                     alt.Tooltip("low:Q", format=",.0f", title="低"),
                     alt.Tooltip("high:Q", format=",.0f", title="高")])
        chart = bars
        if price > 0:
            rule = alt.Chart(pd.DataFrame({"px": [price]})).mark_rule(
                color="crimson", strokeDash=[6, 3], size=2).encode(x="px:Q")
            label = alt.Chart(pd.DataFrame({"px": [price], "t": [f"現價 {price:,.0f}"]})
                              ).mark_text(dy=-8, color="crimson", fontWeight="bold"
                              ).encode(x="px:Q", text="t:N")
            chart = bars + rule + label
        st.altair_chart(chart.properties(height=170), width="stretch")
    except Exception as e:  # noqa: BLE001
        st.error(f"Football field 產生失敗:{e}")

    # ── EV → 權益 bridge 瀑布圖 ──
    if "gordon" in values:
        st.subheader("EV → 權益 bridge(Gordon 法)")
        v = values["gordon"]
        steps = [("企業價值 EV", v["Enterprise_Value"]),
                 ("− 淨負債", -v.get("Net_Debt", 0.0)),
                 ("− 少數股權", -v.get("Minority_Interest", 0.0)),
                 ("− 特別股", -v.get("Preferred_Equity", 0.0)),
                 ("+ 非營運資產", v.get("Non_Operating_Assets", 0.0))]
        steps = [(n_, x) for n_, x in steps if abs(x) > 1e-9 or n_ == "企業價值 EV"]
        rows, cum = [], 0.0
        for i, (name, amt) in enumerate(steps):
            start = 0.0 if i == 0 else cum
            cum = amt if i == 0 else cum + amt
            rows.append({"項目": name, "start": min(start, cum), "end": max(start, cum),
                         "類型": "EV" if i == 0 else ("減項" if amt < 0 else "加項"),
                         "金額": amt, "order": i})
        rows.append({"項目": "普通股權益", "start": 0.0, "end": v["Equity_Value"],
                     "類型": "權益", "金額": v["Equity_Value"], "order": len(steps)})
        wf = pd.DataFrame(rows)
        st.altair_chart(
            alt.Chart(wf).mark_bar().encode(
                x=alt.X("項目:N", sort=alt.SortField("order"), title=None,
                        axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("start:Q", title="億元"), y2="end:Q",
                color=alt.Color("類型:N",
                                scale=alt.Scale(domain=["EV", "減項", "加項", "權益"],
                                                range=["#4C78A8", "#E45756", "#54A24B", "#2E5A88"]),
                                legend=None),
                tooltip=[alt.Tooltip("項目:N"), alt.Tooltip("金額:Q", format=",.0f")]
            ).properties(height=280), width="stretch")
        st.caption(f"權益 {v['Equity_Value']:,.0f} 億 ÷ 股數 {shares:,.2f} 億股 "
                   f"= **每股 {v['Value_per_Share']:,.0f} 元**")

    with st.expander("現金流預測明細"):
        proj = model.proj[["Revenue", "EBIT", "EBITDA", "NOPAT", "D&A", "CapEx", "ΔNWC",
                           "FCFF", "PV_FCFF"]]
        st.dataframe(proj.style.format("{:,.1f}"), width="stretch")


# ──────────────────────────────────────────────────────────────
# Tab 4:敏感度與情境
# ──────────────────────────────────────────────────────────────
with tab_sens:
    s1, s2 = st.columns(2)
    s1.markdown("**Gordon:每股價值(WACC × g)**")
    try:
        sens = model.sensitivity(np.arange(wacc - 0.01, wacc + 0.011, 0.005),
                                 np.arange(tg - 0.01, tg + 0.011, 0.005))
        s1.dataframe(sens.style.format("{:,.0f}")
                     .background_gradient(cmap="RdYlGn", axis=None), width="stretch")
    except Exception as e:  # noqa: BLE001
        s1.error(f"敏感度計算失敗:{e}")

    s2.markdown("**Exit:每股價值(WACC × EV/EBITDA)**")
    try:
        sens2 = model.sensitivity_exit(np.arange(wacc - 0.01, wacc + 0.011, 0.005),
                                       np.arange(exit_mult - 2, exit_mult + 2.1, 1.0))
        s2.dataframe(sens2.style.format("{:,.0f}")
                     .background_gradient(cmap="RdYlGn", axis=None), width="stretch")
    except Exception as e:  # noqa: BLE001
        s2.error(f"敏感度計算失敗:{e}")

    # ── 情境分析:Bear / Base / Bull ──
    st.subheader("情境分析(Gordon 法)")
    st.caption("Bear:每年成長 −2pp、利益率 −1pp、WACC +0.5pp;Bull 反向。可依產業自行調整幅度。")
    scen_rows = []
    for name, dg, dm, dw in [("🐻 Bear", -2.0, -1.0, +0.005),
                             ("📊 Base", 0.0, 0.0, 0.0),
                             ("🐂 Bull", +2.0, +1.0, -0.005)]:
        kw = dict(base_kwargs)
        kw["revenue_growth"] = [(x + dg) / 100 for x in drivers["營收成長 (%)"]]
        kw["ebit_margin"]    = [(x + dm) / 100 for x in drivers["營業利益率 (%)"]]
        kw["wacc"] = wacc + dw
        try:
            vps = DCFModel(DCFAssumptions(**kw)).value("gordon")["Value_per_Share"]
            scen_rows.append({"情境": name, "每股價值(元)": vps,
                              "vs 現價": (vps / price - 1) if price > 0 else np.nan})
        except Exception:  # noqa: BLE001
            scen_rows.append({"情境": name, "每股價值(元)": np.nan, "vs 現價": np.nan})
    st.dataframe(pd.DataFrame(scen_rows).set_index("情境")
                 .style.format({"每股價值(元)": "{:,.0f}", "vs 現價": "{:+.1%}"}),
                 width="stretch")


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
