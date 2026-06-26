"""
backtest.py — 訊號回測引擎(台股交易成本 + 防前視 + 樣本外驗證)

設計原則(這幾點才是重點,不是策略本身):
  1. 防前視偏誤(look-ahead):訊號用截至 t 的資料算,部位 t+1 才生效。
  2. 交易成本內建:台股買進手續費 0.1425%,賣出手續費 0.1425% + 證交稅 0.3%。
     高頻翻倉的策略多半死在成本上 —— 不含成本的回測沒有意義。
  3. 樣本內/外切分:參數在 in-sample 選,績效在 out-of-sample 驗。
     主動回報 IS 與 OOS 的落差,落差大 = 過度配適。
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

# 台股交易成本(可調)
BUY_COST  = 0.001425                 # 券商手續費
SELL_COST = 0.001425 + 0.003         # 手續費 + 證交稅(一般股票 0.3%)


# ──────────────────────────────────────────────────────────────
# 技術指標(都只用截至當下的資料,不偷看未來)
# ──────────────────────────────────────────────────────────────
def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).mean()

def rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ──────────────────────────────────────────────────────────────
# 策略 → 回傳「目標部位」序列(1 = 持有, 0 = 空手)
# 都是經濟邏輯明確的少數策略,不是亂搜參數
# ──────────────────────────────────────────────────────────────
def strat_sma_crossover(prices, short=20, long=60):
    sig = (sma(prices, short) > sma(prices, long)).astype(float)
    return sig.fillna(0)

def strat_momentum(prices, lookback=120):
    sig = (prices / prices.shift(lookback) - 1 > 0).astype(float)
    return sig.fillna(0)

def strat_rsi_meanrev(prices, period=14, low=30, high=70):
    r = rsi(prices, period)
    pos, holding = [], 0
    for v in r:
        if np.isnan(v):
            pos.append(0); continue
        if holding == 0 and v < low:
            holding = 1
        elif holding == 1 and v > high:
            holding = 0
        pos.append(holding)
    return pd.Series(pos, index=prices.index, dtype=float)

def strat_buy_hold(prices):
    return pd.Series(1.0, index=prices.index)


# ──────────────────────────────────────────────────────────────
# 回測核心:防前視 + 成本
# ──────────────────────────────────────────────────────────────
@dataclass
class BTResult:
    equity: pd.Series
    net_ret: pd.Series
    position: pd.Series
    metrics: dict

def backtest(prices: pd.Series, target_pos: pd.Series,
             buy_cost=BUY_COST, sell_cost=SELL_COST) -> BTResult:
    # 關鍵:今天的訊號,明天才成交 → shift(1) 消除前視
    pos = target_pos.shift(1).fillna(0)
    ret = prices.pct_change().fillna(0)
    gross = pos * ret

    chg = pos.diff().fillna(pos)           # 部位變化
    cost = np.where(chg > 0, chg * buy_cost,
            np.where(chg < 0, -chg * sell_cost, 0.0))
    cost = pd.Series(cost, index=prices.index)

    net = gross - cost
    equity = (1 + net).cumprod()
    return BTResult(equity, net, pos, compute_metrics(net, equity))


def compute_metrics(net_ret: pd.Series, equity: pd.Series, ppy: int = 252) -> dict:
    n = len(net_ret)
    if n == 0 or equity.iloc[-1] <= 0:
        return dict(total_return=np.nan, cagr=np.nan, sharpe=np.nan,
                    max_drawdown=np.nan, n_trades=0)
    years = n / ppy
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    sd = net_ret.std()
    sharpe = (net_ret.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    return dict(total_return=total, cagr=cagr, sharpe=sharpe, max_drawdown=dd)


# ──────────────────────────────────────────────────────────────
# 樣本內選參、樣本外驗證(誠實揭露過配)
# ──────────────────────────────────────────────────────────────
def optimize_and_validate(prices: pd.Series, strategy: str,
                          param_grid: dict, split: float = 0.65) -> dict:
    """在 in-sample 用 Sharpe 選最佳參數,回報 IS 與 OOS 績效。"""
    cut = int(len(prices) * split)
    is_px, oos_px = prices.iloc[:cut], prices.iloc[cut:]

    builders = {
        "sma": lambda px, p: strat_sma_crossover(px, **p),
        "momentum": lambda px, p: strat_momentum(px, **p),
        "rsi": lambda px, p: strat_rsi_meanrev(px, **p),
    }
    build = builders[strategy]

    # 展開參數網格
    keys = list(param_grid.keys())
    from itertools import product
    combos = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]

    best, best_sharpe = None, -np.inf
    for p in combos:
        m = backtest(is_px, build(is_px, p)).metrics
        s = m["sharpe"]
        if s is not None and not np.isnan(s) and s > best_sharpe:
            best_sharpe, best = s, p

    is_m = backtest(is_px, build(is_px, best)).metrics
    oos_m = backtest(oos_px, build(oos_px, best)).metrics
    return {"best_params": best, "in_sample": is_m, "out_of_sample": oos_m,
            "overfit_gap_sharpe": (is_m["sharpe"] or 0) - (oos_m["sharpe"] or 0)}


def compare_strategies(prices: pd.Series) -> pd.DataFrame:
    """全期、含成本,各策略 vs 買進持有的誠實對照。"""
    rows = {
        "SMA crossover (20/60)": strat_sma_crossover(prices),
        "Momentum (120d)":       strat_momentum(prices),
        "RSI mean-rev (14)":     strat_rsi_meanrev(prices),
        "Buy & Hold":            strat_buy_hold(prices),
    }
    out = {}
    for name, sig in rows.items():
        out[name] = backtest(prices, sig).metrics
    df = pd.DataFrame(out).T[["total_return", "cagr", "sharpe", "max_drawdown"]]
    return df
