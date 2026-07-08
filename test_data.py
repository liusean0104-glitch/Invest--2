"""test_data.py — 資料層解析邏輯的離線測試(合成 FinMind 格式)。"""
import pandas as pd
import numpy as np
import pytest

from data import (annual_income, annual_cashflow, yearend_balance,
                  build_history, suggest_assumptions, latest_close)


def _rows(rows):
    return pd.DataFrame(rows, columns=["date", "stock_id", "type", "value", "origin_name"])


# 損益表:單季制 → 年度加總;不足 4 季的年剔除
def test_income_annualization_and_units():
    rows = []
    for q, d in enumerate(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]):
        rows.append([d, "2308", "Revenue", 100_000_000_000, "營業收入合計"])       # 每季 1000 億(元)
        rows.append([d, "2308", "OperatingIncome", 15_000_000_000, "營業利益"])
        rows.append([d, "2308", "PreTaxIncome", 16_000_000_000, "稅前淨利"])
        rows.append([d, "2308", "TAX", 3_200_000_000, "所得稅費用"])
    rows.append(["2025-03-31", "2308", "Revenue", 120_000_000_000, "營業收入合計"])  # 2025 只有 1 季
    inc = annual_income(_rows(rows))
    assert list(inc.index) == ["2024"]                       # 不完整年度被剔除
    assert inc.loc["2024", "Revenue"] == pytest.approx(4000)  # 4×1000 億
    assert inc.loc["2024", "OperatingIncome"] == pytest.approx(600)


# 現金流量表:累計制 → 取各年最後一期;CapEx 負值取絕對值
def test_cashflow_cumulative_takes_yearend():
    rows = [
        ["2024-06-30", "2308", "Depreciation", 5_000_000_000, "折舊費用"],
        ["2024-12-31", "2308", "Depreciation", 11_000_000_000, "折舊費用"],        # 全年累計 110 億
        ["2024-12-31", "2308", "Amortization", 1_000_000_000, "攤銷費用"],
        ["2024-06-30", "2308", "PropertyAndPlantAndEquipment", -20_000_000_000, "取得不動產、廠房及設備"],
        ["2024-12-31", "2308", "PropertyAndPlantAndEquipment", -46_000_000_000, "取得不動產、廠房及設備"],
    ]
    cf = annual_cashflow(_rows(rows))
    assert cf.loc["2024", "DA"] == pytest.approx(120)         # 110 + 10
    assert cf.loc["2024", "CapEx"] == pytest.approx(460)      # 取年末累計、轉正


# 資產負債表:NWC 剔除現金與帶息負債;NetDebt;股本→億股
def test_balance_derivations():
    rows = [
        ["2024-12-31", "2308", "CurrentAssets", 300_000_000_000, "流動資產合計"],
        ["2024-12-31", "2308", "CurrentLiabilities", 180_000_000_000, "流動負債合計"],
        ["2024-12-31", "2308", "CashAndCashEquivalents", 80_000_000_000, "現金及約當現金"],
        ["2024-12-31", "2308", "ShortTermBorrowings", 30_000_000_000, "短期借款"],
        ["2024-12-31", "2308", "LongTermBorrowings", 50_000_000_000, "長期借款"],
        ["2024-12-31", "2308", "OrdinaryShare", 25_975_000_000, "普通股股本"],     # 259.75 億
    ]
    bs = yearend_balance(_rows(rows))
    # NWC = (3000−800) − (1800−300) = 700 億
    assert bs.loc["2024", "NWC"] == pytest.approx(700)
    # NetDebt = 300 + 500 − 800 = 0
    assert bs.loc["2024", "NetDebt"] == pytest.approx(0)
    # 股數 = 259.75/10 = 25.975 億股(台達電實例)
    assert bs.loc["2024", "Shares"] == pytest.approx(25.975)


# 歷史指標表與建議假設:比率與 CAGR 遞減規則
def test_history_and_suggestions():
    inc = pd.DataFrame({"Revenue": [1000., 1200., 1500.],
                        "OperatingIncome": [120., 168., 270.],
                        "PreTaxIncome": [130., 175., 280.],
                        "Tax": [26., 35., 61.6]},
                       index=["2022", "2023", "2024"])
    cf = pd.DataFrame({"DA": [50., 60., 75.], "CapEx": [70., 84., 105.]},
                      index=["2022", "2023", "2024"])
    bs = pd.DataFrame({"NWC": [100., 120., 150.]}, index=["2022", "2023", "2024"])
    h = build_history(inc, cf, bs)
    assert h.loc["2024", "OpMargin%"] == pytest.approx(18.0)
    assert h.loc["2024", "CapEx%"] == pytest.approx(7.0)

    s = suggest_assumptions(h, n_years=5)
    # 2 年 CAGR = (1500/1000)^(1/2) − 1 ≈ 22.47% → 起點,遞減到 4%
    assert s["growth_pct"][0] == pytest.approx(22.5, abs=0.1)
    assert s["growth_pct"][-1] == pytest.approx(4.0)
    assert s["tax_rate_pct"] == pytest.approx(21.0, abs=0.5)


def test_latest_close():
    px = pd.DataFrame({"date": ["2026-07-03", "2026-07-06"], "close": [2150.0, 2175.0]})
    assert latest_close(px) == 2175.0
    assert latest_close(pd.DataFrame()) is None


# ── 回歸測試:鎖死單位換算(曾經 1000 倍換算錯誤,以 2912 真實數字為準) ──
def test_unit_conversion_matches_real_2912_figures():
    # 統一超商實際股本約 208 億元(面額 10 元 → 約 20.8 億股),
    # FinMind 原始 value 單位為「元」→ 208 億元 = 2.08e10 元
    rows = _rows([["2024-12-31", "2912", "OrdinaryShare", 2.08e10, "普通股股本"]])
    bs = yearend_balance(rows)
    assert 19 < bs.loc["2024", "Shares"] < 22          # 應約 20.8 億股,不是 20792 億股

    # 實際淨現金部位約 186 億元(net_debt 為負)→ 原始值約 -1.86e10 元
    rows2 = _rows([
        ["2024-12-31", "2912", "CashAndCashEquivalents", 3.0e10, "現金及約當現金"],
        ["2024-12-31", "2912", "ShortTermBorrowings", 1.14e10, "短期借款"],
    ])
    bs2 = yearend_balance(rows2)
    assert -190 < bs2.loc["2024", "NetDebt"] < -180     # 應約 -186 億元,不是 -186174 億元


# ── 新功能:IFRS 16 租賃負債解析 + 52 週價格區間 ──
def test_lease_liabilities_parsed():
    rows = _rows([
        ["2024-12-31", "2912", "LeaseLiabilitiesCurrent", 3.0e10, "租賃負債-流動"],
        ["2024-12-31", "2912", "LeaseLiabilitiesNoncurrent", 7.5e10, "租賃負債-非流動"],
        ["2024-12-31", "2912", "CashAndCashEquivalents", 6.0e10, "現金及約當現金"],
    ])
    bs = yearend_balance(rows)
    assert bs.loc["2024", "LeaseLiab"] == pytest.approx(1050)   # 300+750 億
    # 租賃負債「不」自動進 NetDebt,由前端決定是否納入
    assert bs.loc["2024", "NetDebt"] == pytest.approx(-600)


def test_price_52w_range():
    from data import price_52w_range
    today = pd.Timestamp.today()
    px = pd.DataFrame({
        "date": [(today - pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (400, 200, 10)],
        "close": [999.0, 205.0, 268.0]})   # 400 天前的 999 應被排除
    lo, hi = price_52w_range(px)
    assert (lo, hi) == (205.0, 268.0)
    assert price_52w_range(pd.DataFrame()) is None
