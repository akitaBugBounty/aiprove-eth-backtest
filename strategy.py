"""
AiProve eth-pnl-fixed — SMC / ICT market-structure strategy.

Approach (per the Riky SMC x ICT deck):
  * Read structure first: swing highs/lows (fractals) -> HH/HL = uptrend,
    LH/LL = downtrend. Trade WITH the trend only.
  * Displacement: an impulsive leg (>= min_leg_bps) that breaks structure.
  * Pullback into the leg's fair-value gap, where the FVG overlaps the
    38.2-78.6% Fibonacci retracement zone of the leg (confluence).
  * Liquidity sweep (optional bonus): pullback dips below/above the leg
    origin, then reclaims it.
  * Entry ONLY after confirmation: a with-trend close (or engulfing /
    displacement candle) after the FVG is tapped.
  * Risk: fixed fractional risk per trade, stop beyond the sweep/zone
    extreme, take-profit at RR x risk (or trend flip, whichever first).

The strategy is a pure function of closed bars: signals are computed on
bar close and executed at the NEXT bar open (no phantom fills, no
look-ahead). All parameters live in DEFAULT_PARAMS and are swept by
optimize.py; the final set is in FINAL_PARAMS at the bottom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PARAMS = dict(
    swing_n=3,            # fractal window for swing highs/lows
    atr_len=14,           # ATR length (1-min bars)
    ema_fast=20,          # trend filter EMAs
    ema_slow=50,
    min_leg_bps=35.0,     # minimum displacement-leg size (bps)
    max_leg_bars=60,      # leg must complete within this many bars
    fib_lo=0.382,         # Fib retracement zone (of the displacement leg)
    fib_hi=0.786,         # wider zone: 1-min pullbacks are shallow
    require_sweep=False,  # sweep of leg origin required (else it is a bonus)
    confirm="close_or_engulf",  # bullish/bearish close OR engulfing/displacement
    risk_pct=0.01,        # equity risk per trade
    rr=2.0,               # take-profit = rr x risk distance (0 = no fixed TP)
    tp_mode="fixed_and_trail",  # fixed | trail | fixed_and_trail
    trail_after_r=1.0,    # start trailing once unrealized >= this x risk
    trail_atr_mult=2.0,   # chandelier trail distance (x ATR)
    entry_mode="market",  # market (taker) | limit (maker at FVG edge)
    limit_valid_bars=3,   # bars a limit entry stays valid
    sl_buffer_atr=0.25,   # stop buffer beyond sweep/zone extreme (x ATR)
    min_sl_bps=30.0,      # stop distance floor (avoid stops inside noise)
    max_sl_bps=150.0,     # stop distance cap
    max_hold_bars=180,    # force-close after this many bars in a trade
    cooldown_bars=5,      # bars to wait after an exit before a new entry
    trend_filter=True,    # only trade with the EMA + structure trend
    regime="adx",         # none | adx (require ADX>=adx_min) | ema200
    adx_min=20.0,         # minimum ADX for entries (regime=adx)
    max_pos_notional_pct=1.0,  # cap position notional at this x equity
)

FINAL_PARAMS = dict(
    DEFAULT_PARAMS,
    min_leg_bps=35.0,
    require_sweep=False,
    confirm="close_or_engulf",
    tp_mode="fixed",
    entry_mode="limit",
    rr=2.5,
    risk_pct=0.01,
    min_sl_bps=30.0,
    max_sl_bps=150.0,
    max_hold_bars=240,
    trend_filter=True,
    regime="adx",
    adx_min=30.0,
    max_pos_notional_pct=2.0,
)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    hi, lo, cl = df["high"], df["low"], df["close"]
    prev_cl = cl.shift(1)
    tr = pd.concat(
        [(hi - lo), (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / p["atr_len"], adjust=False).mean()
    df["ema_fast"] = cl.ewm(span=p["ema_fast"], adjust=False).mean()
    df["ema_slow"] = cl.ewm(span=p["ema_slow"], adjust=False).mean()
    df["ema200"] = cl.ewm(span=200, adjust=False).mean()

    # ADX(14) with Wilder smoothing — regime filter: only trade when a
    # real trend exists; 1-min chop without trend is where this
    # strategy bleeds.
    n_adx = 14
    up_move = hi - hi.shift(1)
    dn_move = lo.shift(1) - lo
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr_s = pd.Series(tr).ewm(alpha=1 / n_adx, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / n_adx, adjust=False).mean() / tr_s
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / n_adx, adjust=False).mean() / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / n_adx, adjust=False).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Swing highs / lows: fractal with window n on each side. A swing is
    # only "known" n bars after it forms (it needs n bars to confirm), so
    # shift the series by n — using the raw centered value would be
    # look-ahead bias.
    n = p["swing_n"]
    roll_hi = hi.rolling(2 * n + 1, center=True).max()
    roll_lo = lo.rolling(2 * n + 1, center=True).min()
    df["swing_high"] = np.where(hi == roll_hi, hi, np.nan)
    df["swing_low"] = np.where(lo == roll_lo, lo, np.nan)
    df["swing_high"] = pd.Series(df["swing_high"]).shift(n)
    df["swing_low"] = pd.Series(df["swing_low"]).shift(n)

    # Bullish FVG: low[i] > high[i-2] ; bearish FVG: high[i] < low[i-2].
    df["fvg_bull_top"] = np.where(lo > hi.shift(2), lo, np.nan)
    df["fvg_bull_bot"] = np.where(lo > hi.shift(2), hi.shift(2), np.nan)
    df["fvg_bear_bot"] = np.where(hi < lo.shift(2), hi, np.nan)
    df["fvg_bear_top"] = np.where(hi < lo.shift(2), lo.shift(2), np.nan)
    return df


def _last_valid(s: pd.Series, i: int) -> float | None:
    """Last non-NaN value of series s at or before index i."""
    v = s.iloc[: i + 1].dropna()
    return float(v.iloc[-1]) if len(v) else None


def _last_two_valid(s: pd.Series, i: int) -> list[float]:
    v = s.iloc[: i + 1].dropna()
    return [float(x) for x in v.iloc[-2:]] if len(v) >= 2 else []


def structure_trend(df: pd.DataFrame, i: int) -> int:
    """+1 uptrend (HH/HL), -1 downtrend (LH/LL), 0 range/mixed."""
    sh = _last_two_valid(df["swing_high"], i)
    sl = _last_two_valid(df["swing_low"], i)
    up = len(sh) == 2 and sh[1] > sh[0] and len(sl) == 2 and sl[1] > sl[0]
    dn = len(sh) == 2 and sh[1] < sh[0] and len(sl) == 2 and sl[1] < sl[0]
    if up:
        return 1
    if dn:
        return -1
    return 0


def last_displacement_leg(df: pd.DataFrame, i: int, direction: int, p: dict):
    """Most recent impulsive leg in `direction`.

    Long leg: swing low -> later swing high (or the highest high since
    that low). Returns dict(origin, extreme, leg_bps, end_idx) or None.
    """
    n = p["swing_n"]
    if direction == 1:
        lows = df["swing_low"].iloc[: i + 1].dropna()
        if len(lows) == 0:
            return None
        origin_idx = int(lows.index[-1])
        origin = float(lows.iloc[-1])
        seg = df.iloc[origin_idx : i + 1]
        extreme_idx = int(seg["high"].idxmax())
        extreme = float(seg["high"].max())
        leg_bps = (extreme - origin) / origin * 1e4
        if leg_bps < p["min_leg_bps"] or extreme_idx - origin_idx > p["max_leg_bars"]:
            return None
        return dict(origin=origin, extreme=extreme, leg_bps=leg_bps,
                    start=origin_idx, end=extreme_idx)
    else:
        highs = df["swing_high"].iloc[: i + 1].dropna()
        if len(highs) == 0:
            return None
        origin_idx = int(highs.index[-1])
        origin = float(highs.iloc[-1])
        seg = df.iloc[origin_idx : i + 1]
        extreme_idx = int(seg["low"].idxmin())
        extreme = float(seg["low"].min())
        leg_bps = (origin - extreme) / origin * 1e4
        if leg_bps < p["min_leg_bps"] or extreme_idx - origin_idx > p["max_leg_bars"]:
            return None
        return dict(origin=origin, extreme=extreme, leg_bps=leg_bps,
                    start=origin_idx, end=extreme_idx)


def _is_engulfing(df: pd.DataFrame, i: int, direction: int) -> bool:
    if i < 1:
        return False
    o0, c0 = df["open"].iloc[i], df["close"].iloc[i]
    o1, c1 = df["open"].iloc[i - 1], df["close"].iloc[i - 1]
    if direction == 1:
        return c0 > o0 and c0 > o1 and o0 <= c1 and (c0 - o0) >= (o1 - c1)
    return c0 < o0 and c0 < o1 and o0 >= c1 and (o0 - c0) >= (c1 - o1)


def _is_displacement(df: pd.DataFrame, i: int, direction: int, p: dict) -> bool:
    """Strong impulsive candle: big range vs ATR and close in the
    direction's extreme quartile, closing beyond the prior bar."""
    o, c = df["open"].iloc[i], df["close"].iloc[i]
    h, l = df["high"].iloc[i], df["low"].iloc[i]
    atr = df["atr"].iloc[i]
    if not np.isfinite(atr) or atr <= 0 or (h - l) < 0.8 * atr:
        return False
    if direction == 1:
        return c > o and c >= h - 0.25 * (h - l) and c > df["high"].iloc[i - 1]
    return c < o and c <= l + 0.25 * (h - l) and c < df["low"].iloc[i - 1]


def confirmation(df: pd.DataFrame, i: int, direction: int, p: dict) -> bool:
    mode = p["confirm"]
    eng = _is_engulfing(df, i, direction)
    dis = _is_displacement(df, i, direction, p)
    if mode == "engulf":
        return eng
    if mode == "displacement":
        return dis
    if mode == "engulf_or_displacement":
        return eng or dis
    # close_or_engulf: a with-trend close (close beyond prior close, in
    # the direction's colour) is enough on 1-min bars, where true
    # engulfing candles are rare; engulfing/displacement also count.
    o, c = df["open"].iloc[i], df["close"].iloc[i]
    pc = df["close"].iloc[i - 1] if i >= 1 else c
    if direction == 1:
        return (c > o and c > pc) or eng or dis
    return (c < o and c < pc) or eng or dis


def _latest_fvg(df: pd.DataFrame, i: int, direction: int, leg: dict):
    """Most recent FVG in `direction` formed between the leg start and
    bar i (inclusive). Returns (top, bottom) or None."""
    if direction == 1:
        top_s = df["fvg_bull_top"].iloc[leg["start"] : i + 1].dropna()
        bot_s = df["fvg_bull_bot"].iloc[leg["start"] : i + 1].dropna()
        if len(top_s) == 0:
            return None
        return float(top_s.iloc[-1]), float(bot_s.iloc[-1])
    top_s = df["fvg_bear_top"].iloc[leg["start"] : i + 1].dropna()
    bot_s = df["fvg_bear_bot"].iloc[leg["start"] : i + 1].dropna()
    if len(top_s) == 0:
        return None
    return float(top_s.iloc[-1]), float(bot_s.iloc[-1])


# ---------------------------------------------------------------------------
# Signal generation — pure function of closed bars
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Return df with a `signal` column: +1 long / -1 short / 0 flat,
    plus entry/stop/target columns for the signal bar (execution is at
    the NEXT bar open, handled by the backtest harness).

    One position at a time; a signal is only emitted when flat. The
    harness enforces cooldown / max-hold / exits.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    df = add_indicators(df, p)
    n = len(df)
    sig = np.zeros(n, dtype=int)
    entry = np.full(n, np.nan)
    stop = np.full(n, np.nan)
    target = np.full(n, np.nan)
    entry_limit = np.full(n, np.nan)
    warmup = max(60, p["ema_slow"] + 5)
    used_legs: set = set()  # one signal per displacement leg

    for i in range(warmup, n):
        struct = structure_trend(df, i)
        # Regime gate: skip chop. ADX < adx_min (or price hugging EMA200
        # with no trend) means no trade, however good the setup looks.
        regime = p.get("regime", "none")
        if regime == "adx":
            if not np.isfinite(df["adx"].iloc[i]) or df["adx"].iloc[i] < p.get("adx_min", 20.0):
                continue
        elif regime == "ema200":
            if abs(df["close"].iloc[i] / df["ema200"].iloc[i] - 1) < 0.002:
                continue
        # Candidate directions, trend-filtered when enabled. Trend =
        # EMA alignment, and structure must not clearly oppose it.
        if p["trend_filter"]:
            if df["ema_fast"].iloc[i] > df["ema_slow"].iloc[i] and struct != -1:
                dirs = [1]
            elif df["ema_fast"].iloc[i] < df["ema_slow"].iloc[i] and struct != 1:
                dirs = [-1]
            else:
                dirs = []
        else:
            dirs = [1, -1]
        for direction in dirs:
            leg = last_displacement_leg(df, i, direction, p)
            if leg is None:
                continue
            leg_key = (direction, leg["start"], leg["end"])
            if leg_key in used_legs:
                continue
            # Entry zone = latest FVG of the leg's direction; it must
            # overlap the Fib 38.2-78.6% retracement zone of the leg.
            fvg = _latest_fvg(df, i, direction, leg)
            if fvg is None:
                continue
            fvg_top, fvg_bot = fvg
            if direction == 1:
                fib_hi = leg["extreme"] - p["fib_lo"] * (leg["extreme"] - leg["origin"])
                fib_lo = leg["extreme"] - p["fib_hi"] * (leg["extreme"] - leg["origin"])
            else:
                fib_hi = leg["extreme"] + p["fib_hi"] * (leg["origin"] - leg["extreme"])
                fib_lo = leg["extreme"] + p["fib_lo"] * (leg["origin"] - leg["extreme"])
            # FVG must overlap the Fib zone (confluence), else skip.
            if fvg_top < fib_lo or fvg_bot > fib_hi:
                continue
            # Tap: price traded into the FVG on this bar or the last 3.
            if direction == 1:
                tapped = df["low"].iloc[max(leg["end"], i - 3) : i + 1].min() <= fvg_top
                extreme = df["low"].iloc[leg["end"] : i + 1].min()
                swept = extreme < leg["origin"] and df["close"].iloc[i] > leg["origin"]
            else:
                tapped = df["high"].iloc[max(leg["end"], i - 3) : i + 1].max() >= fvg_bot
                extreme = df["high"].iloc[leg["end"] : i + 1].max()
                swept = extreme > leg["origin"] and df["close"].iloc[i] < leg["origin"]
            if not tapped:
                continue
            if p["require_sweep"] and not swept:
                continue
            if not confirmation(df, i, direction, p):
                continue
            # Stops go beyond the FVG extreme + ATR buffer.
            atr = df["atr"].iloc[i]
            close = df["close"].iloc[i]
            if direction == 1:
                sl = min(extreme, fvg_bot) - p["sl_buffer_atr"] * atr
                sl = close - max(p["min_sl_bps"] / 1e4 * close,
                                 min(p["max_sl_bps"] / 1e4 * close, close - sl))
                tp = close + p["rr"] * (close - sl)
            else:
                sl = max(extreme, fvg_top) + p["sl_buffer_atr"] * atr
                sl = close + max(p["min_sl_bps"] / 1e4 * close,
                                 min(p["max_sl_bps"] / 1e4 * close, sl - close))
                tp = close - p["rr"] * (sl - close)
            sig[i] = direction
            entry[i] = close  # reference price; harness fills at next open
            stop[i] = sl
            target[i] = tp
            # Limit entry at the FVG edge price (maker fill if tapped).
            entry_limit[i] = fvg_top if direction == 1 else fvg_bot
            used_legs.add(leg_key)
            break  # one signal per bar

    df["signal"] = sig
    df["sig_entry_ref"] = entry
    df["sig_stop"] = stop
    df["sig_target"] = target
    df["sig_entry_limit"] = entry_limit
    return df


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "bars.csv"
    raw = pd.read_csv(path)
    out = generate_signals(raw, FINAL_PARAMS)
    print("signals:", {d: int((out['signal'] == d).sum()) for d in (1, -1)})
    print(out.loc[out["signal"] != 0,
                  ["minute", "close", "signal", "sig_stop", "sig_target"]].to_string())
