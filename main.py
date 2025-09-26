from AlgorithmImports import *
import io, math
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict


class NdxTopN_ScanManyN(QCAlgorithm):

    def Initialize(self):
        # -------- 固定扫描 N=1..100 --------
        self.NS = list(range(1, 101))

        # -------- 其它参数--------
        self.rebalance_freq = (self.GetParameter("REBALANCE_FREQ") or "A").upper()   # A/Q/M
        self.weight_scheme  = (self.GetParameter("WEIGHT_SCHEME")  or "value").lower()  # value/equal
        self.csv_name       = self.GetParameter("CONSTITUENTS_CSV") or "ndx_constituents_2007_2025.csv"

        # -------- 使用“总回报”（含分红复权）价格 --------
        def _sec_init(sec: Security):
            sec.SetDataNormalizationMode(DataNormalizationMode.TotalReturn)
            if sec.Type == SecurityType.Equity:
                sec.SetLeverage(1)
        self.SetSecurityInitializer(_sec_init)

        # -------- 载入历史成分（ObjectStore）--------
        self.constituents = self._load_constituents_csv(self.csv_name)
        if self.constituents.empty:
            self.Debug(f"Failed to load constituents CSV: {self.csv_name}")
            self.Quit()
            return

        # date -> set(symbol)
        self.members_by_date = (
            self.constituents.groupby("date")["symbol"]
            .apply(lambda s: set(s.tolist())).to_dict()
        )

        # -------- 回测起点对齐到第一份快照 --------
        first_snap = min(self.members_by_date.keys())
        self.SetStartDate(first_snap.year, first_snap.month, first_snap.day)

        # -------- 基本设置 / 基准（QQQ 也设为 TR）--------
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        qqq = self.AddEquity("QQQ", Resolution.Daily)
        qqq.SetDataNormalizationMode(DataNormalizationMode.TotalReturn)
        self.qqq = qqq.Symbol
        self.SetBenchmark(self.qqq)

        # -------- 调仓与持仓集合 --------
        self.current_set: set = set()
        self.next_rebalance = first_snap

        # -------- 多 N 的虚拟组合账本 --------
        self.vp_value:   Dict[int, float]     = {N: 100000.0 for N in self.NS}  # 初始净值 10 万
        self.vp_weights: Dict[int, Dict[Symbol, float]] = {N: {} for N in self.NS}  # N -> {Symbol: weight}
        self.vp_series:  Dict[int, list]      = {N: [] for N in self.NS}        # N -> [(time, value)]

        # 记录“上一交易日收盘价”，用来算当日简单收益
        self.prev_close: Dict[Symbol, float] = {}

        self.SetWarmUp(10, Resolution.Daily)
        self.Debug(f"Scanning N=1..100, REBALANCE={self.rebalance_freq}, WEIGHT={self.weight_scheme}, CSV={self.csv_name}")
        self.Debug(f"First snapshot: {first_snap}, backtest starts here")


    # ====================== CSV Loader ======================
    def _load_constituents_csv(self, name: str) -> pd.DataFrame:
        from io import StringIO
        try:
            b = self.ObjectStore.ReadBytes(name)
        except Exception as e:
            self.Debug(f"ObjectStore.ReadBytes failed: {e}")
            return pd.DataFrame()
        if not b:
            return pd.DataFrame()

        text = bytes(list(b)).decode("utf-8")
        df = pd.read_csv(StringIO(text))
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns or "symbol" not in df.columns:
            return pd.DataFrame()

        df["date"]   = pd.to_datetime(df["date"]).dt.date
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=["date", "symbol"]).sort_values(["date", "symbol"])
        return df


    # ====================== Universe ======================
    def CoarseSelectionFunction(self, coarse: List[CoarseFundamental]) -> List[Symbol]:
        desired = self._membership_for_date(self.Time.date())
        self.current_set = desired
        if not desired:
            return []
        return [c.Symbol for c in coarse if c.HasFundamentalData and c.Symbol.Value in desired]

    def FineSelectionFunction(self, fine: List[FineFundamental]) -> List[Symbol]:
        return [f.Symbol for f in fine if f.Symbol.Value in self.current_set]

    def _membership_for_date(self, d: date) -> set:
        keys = [k for k in self.members_by_date.keys() if k <= d]
        if not keys:
            return set()
        return set(self.members_by_date[max(keys)])


    # ====================== 调仓日期推进 ======================
    def _next_rebalance_date(self, dt_in, freq: str) -> date:
        d = dt_in.date() if isinstance(dt_in, datetime) else dt_in
        base = datetime(d.year, d.month, 1)
        if freq == "M":
            nxt = base + relativedelta(months=1)
        elif freq == "Q":
            nxt = base + relativedelta(months=3)
        else:  # "A"
            nxt = datetime(d.year + 1, 1, 1)
        return nxt.date()


    # ====================== 主循环（并行更新所有 N） ======================
    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            # 预热阶段只同步 prev_close
            for sym, bar in data.Bars.items():
                self.prev_close[sym] = float(bar.Close)
            return

        # 到达调仓日：按市值/等权，生成每个 N 的新权重
        if self.Time.date() >= self.next_rebalance:
            self._rebalance_all_N()
            self.next_rebalance = self._next_rebalance_date(self.next_rebalance, self.rebalance_freq)
            self.Debug(f"Next rebalance scheduled: {self.next_rebalance}")

        # 先用“昨日收盘”算回报，再更新 prev_close=今日收盘
        rets_today: Dict[Symbol, float] = {}
        for sym, bar in data.Bars.items():
            prev = self.prev_close.get(sym)
            if prev is not None and prev > 0 and bar.Close > 0:
                rets_today[sym] = float(bar.Close) / float(prev) - 1.0

        # 更新各 N 的虚拟净值
        for N in self.NS:
            weights = self.vp_weights.get(N, {})
            if not weights:
                self.vp_series[N].append((self.Time, self.vp_value[N]))
                continue
            r = 0.0
            for sym, w in weights.items():
                if sym in rets_today:
                    r += w * rets_today[sym]
            self.vp_value[N] *= (1.0 + r)
            self.vp_series[N].append((self.Time, self.vp_value[N]))

        # 刷新 prev_close 为今日收盘
        for sym, bar in data.Bars.items():
            self.prev_close[sym] = float(bar.Close)


    # ---------- 生成所有 N 的权重 ----------
    def _rebalance_all_N(self):
        cands = []
        for sec in self.Securities.Values:
            sym = sec.Symbol
            if sym.Value not in self.current_set:
                continue
            if not sec.HasData or sec.Fundamentals is None:
                continue
            mcap = sec.Fundamentals.MarketCap
            if mcap is None or mcap <= 0:
                continue
            cands.append((sym, float(mcap)))

        if not cands:
            self.Debug(f"{self.Time.date()} no candidates with fundamentals; skip")
            return

        cands.sort(key=lambda x: x[1], reverse=True)

        for N in self.NS:
            top = cands[:max(1, N)]
            if self.weight_scheme == "value":
                tot = sum(m for _, m in top)
                if tot > 0:
                    weights = {s: (m / tot) for s, m in top}
                else:
                    w = 1.0 / len(top)
                    weights = {s: w for s, _ in top}
            else:  # equal
                w = 1.0 / len(top)
                weights = {s: w for s, _ in top}
            self.vp_weights[N] = weights

        self.Debug(f"Rebalanced ALL at {self.Time.date()} (|cands|={len(cands)})")


    # ====================== 收尾：输出每个 N 的 TR/CAGR & QQQ 对照 ======================
    def OnEndOfAlgorithm(self):
        def bench_stats(t0: pd.Timestamp, t1: pd.Timestamp):
            try:
                bh = self.History([self.qqq], t0, t1, Resolution.Daily)
                if bh is None or bh.empty:
                    return float('nan'), float('nan')
                if isinstance(bh.index, pd.MultiIndex):
                    try:
                        px = bh.loc[self.qqq]['close']
                    except Exception:
                        px = bh.xs(self.qqq, level=0)['close']
                    idx = px.index
                    if isinstance(idx, pd.MultiIndex):
                        idx = idx.get_level_values(-1)
                    px = pd.Series(px.values, index=pd.to_datetime(idx)).sort_index()
                else:
                    px = bh['close'] if 'close' in bh.columns else bh.squeeze()
                    px = pd.Series(px.values, index=pd.to_datetime(px.index)).sort_index()

                if px.empty or float(px.iloc[0]) <= 0:
                    return float('nan'), float('nan')

                px0, px1 = float(px.iloc[0]), float(px.iloc[-1])
                years_b  = (px.index[-1] - px.index[0]).days / 365.25
                tr       = (px1 / px0) - 1.0
                cagr     = (px1 / px0) ** (1/years_b) - 1 if years_b > 0 else float('nan')
                return tr, cagr
            except Exception as e:
                self.Debug(f"History(benchmark) failed: {e}")
                return float('nan'), float('nan')

        rows = []
        best_n, best_cagr = None, -1e9

        for N in self.NS:
            series = self.vp_series.get(N, [])
            if len(series) < 2:
                continue

            times, vals = zip(*series)
            ts = pd.Series(list(vals), index=pd.to_datetime(list(times))).sort_index()
            start_dt, end_dt = ts.index[0], ts.index[-1]
            v0, v1 = float(ts.iloc[0]), float(ts.iloc[-1])
            years  = (end_dt - start_dt).days / 365.25

            if v0 > 0:
                strat_tr   = (v1 / v0) - 1.0
                strat_cagr = (v1 / v0) ** (1/years) - 1 if years > 0 else float('nan')
            else:
                strat_tr = float('nan')
                strat_cagr = float('nan')

            bench_tr, bench_cagr = bench_stats(start_dt, end_dt)

            rows.append([
                N,
                strat_tr, strat_cagr,
                bench_tr, bench_cagr,
                str(start_dt.date()), str(end_dt.date())
            ])

            if not math.isnan(strat_cagr) and strat_cagr > best_cagr:
                best_cagr, best_n = strat_cagr, N

        # --- 保存 CSV ---
        rows.sort(key=lambda r: (r[2] if r[2] == r[2] else -1e9), reverse=True)
        buf = io.StringIO()
        buf.write("N,STRAT_TR,STRAT_CAGR,QQQ_TR,QQQ_CAGR,START,END\n")
        for r in rows:
            buf.write("{},{:.10f},{:.10f},{:.10f},{:.10f},{},{}\n".format(*r))
        key = f"ndx_scan_manyN_{self.rebalance_freq}_{self.weight_scheme}.csv"
        self.ObjectStore.SaveBytes(key, buf.getvalue().encode())
        self.Debug(f"Saved ALL-N results to ObjectStore: {key}")

        # --- Debug 输出 ---
        self.Debug("All N by STRAT_CAGR (desc):")
        for r in rows:
            self.Debug(
                f"N={r[0]:>3d}  "
                f"STRAT_TR={r[1]:.2%}  STRAT_CAGR={r[2]:.2%}  "
                f"QQQ_TR={r[3]:.2%}  QQQ_CAGR={r[4]:.2%}  "
                f"[{r[5]} ~ {r[6]}]"
            )

        if best_n is not None:
            self.Debug(f"Best N by CAGR = {best_n} (CAGR={best_cagr:.2%})")
        else:
            self.Debug("No valid N results.")
