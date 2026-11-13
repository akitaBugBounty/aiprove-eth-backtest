"""
Backtest harness for the AiProve eth-pnl-fixed bounty.

Costing (documented assumptions — the exact sandbox numbers are NOT
published anywhere in the repo or the bounty brief, which only says
"maker and taker fees, plus slippage scaled by size"):
  * taker fee 10 bps (0.10%) — market orders (entries, stop-market exits)
  * maker fee  2 bps (0.02%) — limit orders (take-profit exits)
  * slippage  = base_slip_bps + slip_per_participation * (qty / bar volume)
    in bps, capped at slip_cap_bps. Participation is measured against the
    volume of the bar in which the fill happens.
  * fills: signals are computed on bar close and executed at the NEXT bar
    open (no phantom fills, no look-ahead). Stop-loss / take-profit are
    checked intrabar against the bar high/low; if both are touched in one
    bar the stop is assumed hit first (conservative).
  * equity starts at `capital` (default $10,000). Position size = risk
    dollars / stop distance, capped at max_pos_notional_pct x equity.
    No leverage (spot-style, notional <= equity).

Sensitivity: run with COST_BASE and COST_HARSH to check the strategy
survives harsher-than-assumed costs.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

COST_BASE = dict(
    taker_bps=10.0,
    maker_bps=2.0,
    base_slip_bps=0.5,
    slip_per_participation_bps=15.0,  # +15 bps per 100% of bar volume
    slip_cap_bps=30.0,
)

COST_HARSH = dict(
    taker_bps=15.0,
    maker_bps=5.0,
    base_slip_bps=1.0,
    slip_per_participation_bps=45.0,
    slip_cap_bps=50.0,
)

COST_ZERO = dict(
    taker_bps=0.0, maker_bps=0.0,
    base_slip_bps=0.0, slip_per_participation_bps=0.0, slip_cap_bps=0.0,
)


def _slip_bps(qty: float, bar_volume: float, cost: dict) -> float:
    part = qty / max(bar_volume, 1e-9)
    return min(cost["slip_cap_bps"],
               cost["base_slip_bps"] + cost["slip_per_participation_bps"] * part)


class Trade:
    def __init__(self, direction, entry_idx, entry_price, qty, stop, target):
        self.direction = direction
        self.entry_idx = entry_idx
        self.entry_price = entry_price
        self.qty = qty
        self.stop = stop
        self.stop_initial = stop
        self.target = target
        self.risk_dollars = qty * abs(entry_price - stop)
        self.best_extreme = entry_price  # best high (long) / low (short)
        self.trail_active = False
        self.exit_idx = None
        self.exit_price = None
        self.exit_reason = None
        self.entry_fee = 0.0
        self.exit_fee = 0.0
        self.entry_slip = 0.0
        self.exit_slip = 0.0

    def gross_pnl(self):
        if self.exit_price is None:
            return 0.0
        return self.direction * (self.exit_price - self.entry_price) * self.qty

    def net_pnl(self):
        return (self.gross_pnl() - self.entry_fee - self.exit_fee
                - self.entry_slip - self.exit_slip)

    def to_dict(self):
        return dict(direction=int(self.direction), entry_idx=int(self.entry_idx),
                    entry_price=round(self.entry_price, 2), qty=round(self.qty, 4),
                    stop=round(self.stop, 2), stop_initial=round(self.stop_initial, 2),
                    target=round(self.target, 2),
                    trail_active=bool(self.trail_active),
                    exit_idx=None if self.exit_idx is None else int(self.exit_idx),
                    exit_price=None if self.exit_price is None else round(self.exit_price, 2),
                    exit_reason=self.exit_reason,
                    gross_pnl=round(self.gross_pnl(), 2),
                    fees=round(self.entry_fee + self.exit_fee, 2),
                    slippage=round(self.entry_slip + self.exit_slip, 2),
                    net_pnl=round(self.net_pnl(), 2))


def backtest(signals: pd.DataFrame, params: dict, cost: dict = COST_BASE,
             capital: float = 10_000.0, verbose: bool = False) -> dict:
    """Replay signals over bars with honest fills and costs.

    `signals` must carry columns from strategy.generate_signals.
    Returns dict with equity curve, trades and metrics.
    """
    df = signals.reset_index(drop=True)
    n = len(df)
    equity = capital
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    equity_curve = np.full(n, capital, dtype=float)
    trades: list[Trade] = []
    position: Trade | None = None
    pending: dict | None = None   # signal waiting for next-bar-open fill
    cooldown_until = -1
    total_fees = 0.0
    total_slip = 0.0

    for i in range(n):
        bar = df.iloc[i]
        # --- mark to market (equity at close) ---
        if position is not None:
            mtm = position.direction * (bar["close"] - position.entry_price) * position.qty
            equity = capital + sum(t.net_pnl() for t in trades) + mtm
        else:
            equity = capital + sum(t.net_pnl() for t in trades)
        equity_curve[i] = equity
        if equity > peak:
            peak = equity
        dd = peak - equity
        max_dd = max(max_dd, dd)
        max_dd_pct = max(max_dd_pct, dd / peak if peak > 0 else 0.0)

        # --- intrabar exit check for an open position ---
        if position is not None:
            # Trailing stop (chandelier): once the trade is up by
            # trail_after_r x risk, trail the stop behind the best
            # extreme by trail_atr_mult x ATR. Only ever moves in favour.
            if params.get("tp_mode", "fixed") in ("trail", "fixed_and_trail"):
                atr_now = bar["atr"]
                if np.isfinite(atr_now) and atr_now > 0:
                    if position.direction == 1:
                        position.best_extreme = max(position.best_extreme, bar["high"])
                        unreal = (bar["close"] - position.entry_price) * position.qty
                    else:
                        position.best_extreme = min(position.best_extreme, bar["low"])
                        unreal = (position.entry_price - bar["close"]) * position.qty
                    if unreal >= params.get("trail_after_r", 1.0) * position.risk_dollars:
                        position.trail_active = True
                    if position.trail_active:
                        if position.direction == 1:
                            trail = position.best_extreme - params.get("trail_atr_mult", 2.0) * atr_now
                            position.stop = max(position.stop, trail)
                        else:
                            trail = position.best_extreme + params.get("trail_atr_mult", 2.0) * atr_now
                            position.stop = min(position.stop, trail)
            hit_sl = hit_tp = False
            if position.direction == 1:
                hit_sl = bar["low"] <= position.stop
                hit_tp = (params.get("tp_mode", "fixed") in ("fixed", "fixed_and_trail")
                          and params.get("rr", 0) > 0 and bar["high"] >= position.target)
            else:
                hit_sl = bar["high"] >= position.stop
                hit_tp = (params.get("tp_mode", "fixed") in ("fixed", "fixed_and_trail")
                          and params.get("rr", 0) > 0 and bar["low"] <= position.target)
            exit_reason = None
            if hit_sl and hit_tp:
                exit_reason = "stop"  # conservative: worst fill first
            elif hit_sl:
                exit_reason = "trail" if position.trail_active and position.stop != position.stop_initial else "stop"
            elif hit_tp:
                exit_reason = "target"
            elif i - position.entry_idx >= params["max_hold_bars"]:
                exit_reason = "max_hold"
            if exit_reason is not None:
                if exit_reason == "target":
                    px = position.target
                    fee_bps = cost["maker_bps"]
                    slip = 0.0  # limit fill at the limit price
                elif exit_reason == "max_hold":
                    px = bar["close"]
                    fee_bps = cost["taker_bps"]
                    slip_bps = _slip_bps(position.qty, bar["volume"], cost)
                    slip = slip_bps / 1e4 * px * position.qty
                    px = px - position.direction * slip_bps / 1e4 * px
                else:  # stop-market (initial stop or trailed stop)
                    px = position.stop
                    fee_bps = cost["taker_bps"]
                    slip_bps = _slip_bps(position.qty, bar["volume"], cost)
                    slip = slip_bps / 1e4 * px * position.qty
                    px = px - position.direction * slip_bps / 1e4 * px
                fee = fee_bps / 1e4 * px * position.qty
                position.exit_idx = i
                position.exit_price = px
                position.exit_reason = exit_reason
                position.exit_fee = fee
                position.exit_slip = slip
                total_fees += fee
                total_slip += slip
                trades.append(position)
                position = None
                cooldown_until = i + params["cooldown_bars"]
                equity = capital + sum(t.net_pnl() for t in trades)
                equity_curve[i] = equity
                if equity > peak:
                    peak = equity
                continue

        # --- fill a pending entry ---
        if pending is not None and position is None:
            if pending["mode"] == "limit":
                # Limit fill if this bar traded through the limit price.
                touched = (bar["low"] <= pending["limit_px"] if pending["direction"] == 1
                           else bar["high"] >= pending["limit_px"])
                if touched:
                    px = pending["limit_px"]
                    qty = pending["qty"]
                    fee = cost["maker_bps"] / 1e4 * px * qty
                    stop, target = pending["stop"], pending["target"]
                    if pending["direction"] == 1:
                        stop = min(stop, px - pending["stop_dist"])
                        target = px + params["rr"] * (px - stop)
                    else:
                        stop = max(stop, px + pending["stop_dist"])
                        target = px - params["rr"] * (stop - px)
                    position = Trade(pending["direction"], i, px, qty, stop, target)
                    position.entry_fee = fee
                    total_fees += fee
                    pending = None
                elif i - pending["born"] >= params.get("limit_valid_bars", 3):
                    pending = None  # unfilled limit expires
            else:  # market fill at the open
                px = bar["open"]
                qty = pending["qty"]
                slip_bps = _slip_bps(qty, bar["volume"], cost)
                px = px + pending["direction"] * slip_bps / 1e4 * px
                slip = slip_bps / 1e4 * px * qty
                fee = cost["taker_bps"] / 1e4 * px * qty
                stop = pending["stop"]
                target = pending["target"]
                if pending["direction"] == 1:
                    stop = min(stop, px - pending["stop_dist"])
                    target = px + params["rr"] * (px - stop)
                else:
                    stop = max(stop, px + pending["stop_dist"])
                    target = px - params["rr"] * (stop - px)
                position = Trade(pending["direction"], i, px, qty, stop, target)
                position.entry_fee = fee
                position.entry_slip = slip
                total_fees += fee
                total_slip += slip
                pending = None

        # --- new signal (only when flat, after cooldown) ---
        if position is None and pending is None and i >= cooldown_until:
            s = int(df["signal"].iloc[i])
            if s != 0 and i + 1 < n:
                ref = df["sig_entry_ref"].iloc[i]
                sl = df["sig_stop"].iloc[i]
                tp = df["sig_target"].iloc[i]
                if np.isfinite(ref) and np.isfinite(sl) and np.isfinite(tp):
                    stop_dist = abs(ref - sl)
                    if stop_dist <= 0:
                        continue
                    risk_dollars = params["risk_pct"] * equity
                    qty = risk_dollars / stop_dist
                    max_qty = (params["max_pos_notional_pct"] * equity) / ref
                    qty = min(qty, max_qty)
                    if qty * ref < 10:  # dust guard: <$10 notional
                        continue
                    limit_px = df["sig_entry_limit"].iloc[i]
                    if not np.isfinite(limit_px):
                        limit_px = ref
                    pending = dict(direction=s, qty=qty, stop=sl, target=tp,
                                   stop_dist=stop_dist, born=i,
                                   mode=params.get("entry_mode", "market"),
                                   limit_px=float(limit_px))

    # --- close any open position at the last close ---
    if position is not None:
        px = df["close"].iloc[-1]
        fee = cost["taker_bps"] / 1e4 * px * position.qty
        slip_bps = _slip_bps(position.qty, df["volume"].iloc[-1], cost)
        slip = slip_bps / 1e4 * px * position.qty
        px = px - position.direction * slip_bps / 1e4 * px
        position.exit_idx = n - 1
        position.exit_price = px
        position.exit_reason = "eod"
        position.exit_fee = fee
        position.exit_slip = slip
        total_fees += fee
        total_slip += slip
        trades.append(position)
        position = None
        equity = capital + sum(t.net_pnl() for t in trades)

    net = sum(t.net_pnl() for t in trades)
    gross = sum(t.gross_pnl() for t in trades)
    wins = [t for t in trades if t.net_pnl() > 0]
    losses = [t for t in trades if t.net_pnl() <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = float(np.mean([t.net_pnl() for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.net_pnl() for t in losses])) if losses else 0.0
    profit_factor = (sum(t.net_pnl() for t in wins) /
                     abs(sum(t.net_pnl() for t in losses))) if losses and sum(
                         t.net_pnl() for t in losses) != 0 else float("inf")
    return dict(
        net_pnl=round(net, 2),
        gross_pnl=round(gross, 2),
        total_fees=round(total_fees, 2),
        total_slippage=round(total_slip, 2),
        final_equity=round(equity, 2),
        return_pct=round((equity / capital - 1) * 100, 2),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd_pct * 100, 2),
        n_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=round(win_rate * 100, 1),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else None,
        equity_curve=[round(float(x), 2) for x in equity_curve],
        trades=[t.to_dict() for t in trades],
    )


def run(df: pd.DataFrame, params: dict, cost: dict = COST_BASE,
        capital: float = 10_000.0) -> dict:
    from strategy import generate_signals

    sig = generate_signals(df, params)
    return backtest(sig, params, cost=cost, capital=capital)


if __name__ == "__main__":
    import sys

    from strategy import FINAL_PARAMS

    raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "bars.csv")
    for name, c in [("base", COST_BASE), ("harsh", COST_HARSH), ("zero", COST_ZERO)]:
        r = run(raw, FINAL_PARAMS, cost=c)
        print(f"--- {name} costs ---")
        for k in ("net_pnl", "gross_pnl", "total_fees", "total_slippage",
                  "return_pct", "max_drawdown", "max_drawdown_pct", "n_trades",
                  "win_rate", "avg_win", "avg_loss", "profit_factor"):
            print(f"  {k}: {r[k]}")
