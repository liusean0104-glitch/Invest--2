"""
data.py — 台股財報資料層(FinMind)

設計:
  抓取(_fetch)與解析(annual_income / annual_cashflow / yearend_balance)分離,
  解析函式是純函式 → 可以離線用合成資料測試,不依賴網路。

單位約定:FinMind 財報數字為「千元」;本模組對外一律轉成「億元」,
  股數轉成「億股」(面額 10 元:股數 = 股本 / 10)→ 每股價值自然是「元」。

科目對應採「type 英文鍵 + origin_name 中文關鍵字」雙重比對,
  任何一邊命中即可,降低 FinMind 欄位命名變動的風險。
"""

from __future__ import annotations
import requests
import pandas as pd
import numpy as np

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
DATA_UNIT_TO_YI = 1e8         # 元 → 億元(FinMind 財報 value 欄位單位為「元」,非「千元」;
                              # 1 億 = 1e8 元。此假設已用真實資料反推驗證,見 CHANGELOG。)


# ──────────────────────────────────────────────────────────────
# 抓取層(需要網路;token 免費申請可提高流量上限)
# ──────────────────────────────────────────────────────────────
def _fetch(dataset: str, stock_id: str, start_date: str, token: str = "") -> pd.DataFrame:
    r = requests.get(FINMIND_URL, params={
        "dataset": dataset, "data_id": stock_id,
        "start_date": start_date, "token": token,
    }, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != 200:
        raise RuntimeError(f"FinMind 回應異常:{j.get('msg')}")
    return pd.DataFrame(j["data"])


# ──────────────────────────────────────────────────────────────
# 解析層(純函式,離線可測)
# ──────────────────────────────────────────────────────────────
def _pick(df: pd.DataFrame, type_keys: list[str], name_keywords: list[str]) -> pd.Series:
    """依 type 或 origin_name 關鍵字取出科目,回傳 date → value(千元)。"""
    if df.empty:
        return pd.Series(dtype=float)
    mask = df["type"].isin(type_keys) if "type" in df else pd.Series(False, index=df.index)
    if "origin_name" in df and name_keywords:
        mask |= df["origin_name"].fillna("").str.contains("|".join(name_keywords))
    sub = df[mask].drop_duplicates(subset=[c for c in ("date", "type") if c in df])
    return sub.groupby("date")["value"].sum().sort_index()


def annual_income(df: pd.DataFrame) -> pd.DataFrame:
    """損益表(FinMind 為單季)→ 年度加總;不足 4 季的年度剔除。

    回傳(億元):Revenue, OperatingIncome, PreTaxIncome, Tax
    """
    items = {
        "Revenue":         (["Revenue"], ["營業收入"]),
        "OperatingIncome": (["OperatingIncome"], ["營業利益"]),
        "PreTaxIncome":    (["PreTaxIncome", "IncomeBeforeIncomeTax"], ["稅前淨利", "稅前純益"]),
        "Tax":             (["TAX", "IncomeTax", "TaxExpense"], ["所得稅"]),
    }
    cols = {}
    for name, (tk, kw) in items.items():
        s = _pick(df, tk, kw)
        if s.empty:
            continue
        q = s.to_frame("v")
        q["year"] = q.index.str[:4]
        g = q.groupby("year")["v"]
        full = g.count() == 4                      # 只留完整年度
        cols[name] = (g.sum()[full]) / DATA_UNIT_TO_YI
    return pd.DataFrame(cols).dropna(how="all")


def annual_cashflow(df: pd.DataFrame) -> pd.DataFrame:
    """現金流量表(FinMind 為年內累計 YTD)→ 取各年度最後一期即為全年。

    回傳(億元,皆為正值):DA(折舊+攤銷), CapEx(取得不動產廠房設備)
    """
    items = {
        "Depreciation": (["Depreciation"], ["折舊"]),
        "Amortization": (["Amortization"], ["攤銷"]),
        "CapEx":        (["PropertyAndPlantAndEquipment", "AcquisitionOfPropertyPlantAndEquipment"],
                         ["取得不動產", "購置不動產"]),
    }
    cols = {}
    for name, (tk, kw) in items.items():
        s = _pick(df, tk, kw)
        if s.empty:
            continue
        q = s.to_frame("v")
        q["year"] = q.index.str[:4]
        # 累計制:同年取最後一期(通常是 12-31)
        last = q.sort_index().groupby("year")["v"].last()
        cols[name] = last.abs() / DATA_UNIT_TO_YI   # CapEx 在現金流出常為負,取絕對值
    out = pd.DataFrame(cols).dropna(how="all")
    if {"Depreciation", "Amortization"} & set(out.columns):
        out["DA"] = out.get("Depreciation", 0).fillna(0) + out.get("Amortization", 0).fillna(0)
    return out


def yearend_balance(df: pd.DataFrame) -> pd.DataFrame:
    """資產負債表(時點值)→ 取各年度最後一期。

    回傳(億元):CurrentAssets, CurrentLiabilities, Cash, DebtST, DebtLT,
                OrdinaryShare;並推導 NWC, NetDebt, Shares(億股)
    """
    items = {
        "CurrentAssets":      (["CurrentAssets"], ["流動資產合計"]),
        "CurrentLiabilities": (["CurrentLiabilities"], ["流動負債合計"]),
        "Cash":               (["CashAndCashEquivalents"], ["現金及約當現金"]),
        "DebtST":             (["ShortTermBorrowings", "ShortTermNotesAndBillsPayable",
                                "LongTermLiabilitiesCurrentPortion"],
                               ["短期借款", "應付短期票券", "一年內到期"]),
        "DebtLT":             (["LongTermBorrowings", "BondsPayable"],
                               ["長期借款", "應付公司債"]),
        "LeaseLiab":          (["LeaseLiabilitiesCurrent", "LeaseLiabilitiesNoncurrent",
                                "LeaseLiabilitiesNonCurrent"],
                               ["租賃負債"]),
        "OrdinaryShare":      (["OrdinaryShare", "CommonStocks"], ["普通股股本", "股本合計"]),
    }
    cols = {}
    for name, (tk, kw) in items.items():
        s = _pick(df, tk, kw)
        if s.empty:
            continue
        q = s.to_frame("v")
        q["year"] = q.index.str[:4]
        cols[name] = q.sort_index().groupby("year")["v"].last() / DATA_UNIT_TO_YI
    out = pd.DataFrame(cols).dropna(how="all")
    if not out.empty:
        get = lambda c: out[c] if c in out else 0.0
        # 營運中 NWC:剔除現金與帶息負債(它們屬融資面,不屬營運面)
        out["NWC"] = (get("CurrentAssets") - get("Cash")) - (get("CurrentLiabilities") - get("DebtST"))
        out["NetDebt"] = get("DebtST") + get("DebtLT") - get("Cash")
        out["Shares"] = get("OrdinaryShare") / 10.0          # 面額 10 元 → 億股
    return out


def latest_close(df: pd.DataFrame) -> float | None:
    if df.empty or "close" not in df:
        return None
    return float(df.sort_values("date")["close"].iloc[-1])


def price_52w_range(df: pd.DataFrame) -> tuple[float, float] | None:
    """近 52 週收盤價區間(football field 用)。"""
    if df.empty or "close" not in df:
        return None
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    s = df[df["date"] >= cutoff]["close"]
    if s.empty:
        return None
    return float(s.min()), float(s.max())


# ──────────────────────────────────────────────────────────────
# 彙整:歷史指標表 + 建議假設
# ──────────────────────────────────────────────────────────────
def build_history(inc: pd.DataFrame, cf: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """合併成年度指標表(比率以營收為分母),是「建議假設」的依據,也給使用者檢查。"""
    h = inc.join(cf, how="inner").join(bs, how="left")
    if h.empty or "Revenue" not in h:
        return pd.DataFrame()
    h["RevenueGrowth%"] = h["Revenue"].pct_change() * 100
    h["OpMargin%"] = h["OperatingIncome"] / h["Revenue"] * 100
    if "DA" in h:      h["DA%"] = h["DA"] / h["Revenue"] * 100
    if "CapEx" in h:   h["CapEx%"] = h["CapEx"] / h["Revenue"] * 100
    if "NWC" in h:     h["NWC%"] = h["NWC"] / h["Revenue"] * 100
    if {"Tax", "PreTaxIncome"} <= set(h.columns):
        h["TaxRate%"] = (h["Tax"] / h["PreTaxIncome"]).clip(0, 0.6) * 100
    return h


def suggest_assumptions(h: pd.DataFrame, n_years: int = 5) -> dict:
    """從歷史指標產生 DCF 預設假設(使用者可再改)。規則刻意保守且透明:
       成長率 = min(近 3 年 CAGR, 25%) 線性遞減到 4%;比率類用近 3 年平均。"""
    def avg3(col, default):
        return float(h[col].dropna().tail(3).mean()) if col in h and h[col].notna().any() else default

    rev = h["Revenue"].dropna()
    if len(rev) >= 3:
        yrs = min(3, len(rev) - 1)
        cagr = (rev.iloc[-1] / rev.iloc[-1 - yrs]) ** (1 / yrs) - 1
    elif len(rev) == 2:
        cagr = rev.iloc[-1] / rev.iloc[0] - 1
    else:
        cagr = 0.08
    start_g = float(np.clip(cagr, 0.0, 0.25))
    growth = np.linspace(start_g * 100, 4.0, n_years).round(1).tolist()

    return {
        "growth_pct": growth,
        "op_margin_pct": round(avg3("OpMargin%", 15.0), 1),
        "da_pct": round(avg3("DA%", 5.0), 1),
        "capex_pct": round(avg3("CapEx%", 6.0), 1),
        "nwc_pct": round(avg3("NWC%", 10.0), 1),
        "tax_rate_pct": round(avg3("TaxRate%", 20.0), 1),
    }


def sanity_check(base_revenue: float, shares: float, net_debt: float, price: float | None) -> list[str]:
    """粗略防呆:數字大到不合理時提醒,通常代表資料來源單位換算有誤(如本次 1000 倍的錯誤)。"""
    warnings = []
    if shares > 2000:  # 全台股本最大的公司股數也在千億股以內
        warnings.append(f"流通股數 {shares:,.0f} 億股明顯過大,疑似單位換算錯誤(通常差 1000 倍)。")
    if abs(net_debt) > base_revenue * 50:
        warnings.append(f"淨負債 {net_debt:,.0f} 億元相對營收比例異常,疑似單位換算錯誤。")
    if price and shares > 0:
        implied_mktcap = price * shares
        if implied_mktcap > 500_0000:  # 500 兆元,遠超台股全市場市值,必為錯誤
            warnings.append("現價 × 股數 隱含市值超過合理範圍,請檢查股數單位。")
    return warnings


def load_company(stock_id: str, token: str = "", start_date: str = "2019-01-01") -> dict:
    """一次抓齊四個資料集並解析。回傳 app 需要的全部東西。"""
    inc_raw = _fetch("TaiwanStockFinancialStatements", stock_id, start_date, token)
    cf_raw  = _fetch("TaiwanStockCashFlowsStatement", stock_id, start_date, token)
    bs_raw  = _fetch("TaiwanStockBalanceSheet", stock_id, start_date, token)
    px_raw  = _fetch("TaiwanStockPrice", stock_id,
                     (pd.Timestamp.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d"), token)

    inc, cf, bs = annual_income(inc_raw), annual_cashflow(cf_raw), yearend_balance(bs_raw)
    hist = build_history(inc, cf, bs)
    if hist.empty:
        raise RuntimeError(
            f"{stock_id} 抓到資料但解析後為空 —— 請看診斷區的原始科目名稱,可能需要調整關鍵字對應。")

    latest_bs = bs.iloc[-1] if not bs.empty else pd.Series(dtype=float)
    price = latest_close(px_raw)
    shares = float(latest_bs["Shares"]) if "Shares" in latest_bs else 1.0
    net_debt = float(latest_bs["NetDebt"]) if "NetDebt" in latest_bs else 0.0
    base_revenue = float(hist["Revenue"].iloc[-1])

    def _bs(col):  # 取最新資產負債表單一科目,缺就 0
        return float(latest_bs[col]) if col in latest_bs and pd.notna(latest_bs[col]) else 0.0

    return {
        "stock_id": stock_id,
        "history": hist,
        "suggest": suggest_assumptions(hist),
        "base_revenue": base_revenue,
        "base_nwc": float(latest_bs["NWC"]) if "NWC" in latest_bs else None,
        "net_debt": net_debt,
        "cash": _bs("Cash"),
        "total_debt": _bs("DebtST") + _bs("DebtLT"),
        "lease_liab": _bs("LeaseLiab"),          # IFRS 16 租賃負債(零售/航空等租賃重的公司很關鍵)
        "shares": shares,
        "price": price,
        "price_52w": price_52w_range(px_raw),
        "warnings": sanity_check(base_revenue, shares, net_debt, price),
        # 診斷:真實欄位名若與關鍵字不合,從這裡看抓到了什麼
        "diagnostics": {
            "income_types": sorted(inc_raw["type"].unique().tolist()) if "type" in inc_raw else [],
            "cashflow_names": sorted(cf_raw["origin_name"].dropna().unique().tolist())[:40]
                              if "origin_name" in cf_raw else [],
            "balance_names": sorted(bs_raw["origin_name"].dropna().unique().tolist())[:40]
                             if "origin_name" in bs_raw else [],
        },
    }
