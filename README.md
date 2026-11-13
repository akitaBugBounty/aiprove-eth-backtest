# AiProve `eth-pnl-fixed` — SMC / ICT Market-Structure Strategy

Bounty: **Highest backtest PnL on the fixed ETH dataset** (`eth-pnl-fixed`,
2,000,000 PROVE total; 1st 1,000,000 / 2nd 600,000 / 3rd 400,000;
closes 20 Nov 2026 13:49 UTC). Metric: **net PnL after costs**
(maker + taker fees, slippage scaled by size); ties break on lower max
drawdown. Dataset: `bars.csv` — 1,440 ETH/USDT 1-minute bars (Binance,
24h ending 2026-11-13 12:17 UTC), frozen and identical for every entry.

## Strategy (SMC x ICT, per the Riky deck)

1. **Read structure first.** Swing highs/lows (3-bar fractals, confirmed
   with a 3-bar lag — no look-ahead). Trend = EMA20/EMA50 alignment, and
   structure must not clearly oppose it (HH/HL vs LH/LL).
2. **Regime gate (ADX).** Only trade when ADX(14) ≥ 30 — a real trend
   exists. On this dataset the gate keeps the strategy flat through the
   entire choppy first half (bars 0–719: **zero trades, $0 PnL**) and only
   lets it trade the strong downtrend in the second half.
3. **Displacement.** An impulsive leg of ≥ 35 bps that breaks structure.
4. **Pullback into value.** Entry zone = the leg's fair-value gap (FVG)
   overlapping the 38.2–78.6% Fibonacci retracement of the leg.
   A liquidity sweep past the leg origin (then reclaim) is a bonus, not
   a requirement.
5. **Entry only after confirmation.** A with-trend close (or an engulfing
   / displacement candle) after the FVG is tapped. Entry is a **limit
   order at the FVG edge** (maker fee, valid 3 bars, expires unfilled) —
   never a market chase.
6. **Risk management.** 1% of equity risked per trade; stop beyond the
   FVG/sweep extreme + 0.25×ATR buffer (floor 30 bps, cap 150 bps);
   take-profit at 2.5R; max hold 240 bars; 5-bar cooldown after an exit;
   one position at a time. Position notional capped at 2× equity so the
   planned 1% risk can actually be deployed with tight 1-min stops
   (see "Sizing" below).

## Files

| File | What it is |
|---|---|
| `strategy.py` | Signal generation (pure function of closed bars). `FINAL_PARAMS` = the submitted parameter set. `python3 strategy.py bars.csv` lists signals. |
| `backtest.py` | Replay harness: honest fills (signals execute at the next bar open / on limit touch), maker/taker fees, size-scaled slippage, equity + max drawdown. `python3 backtest.py bars.csv` prints base/harsh/zero-cost results. |
| `run_backtest.py` | Entry point: runs the final config and writes `results.json`. |
| `optimize.py` | 144-combo grid search with split-half + harsh-cost robustness gates. |
| `experiment.py`, `experiment2.py` | Targeted rounds: regime filter, RR, notional-cap scaling. |
| `results.json` | Full metrics, equity curve, and the 6 trades. |
| `optimization_results.json` | All 144 grid-search results. |
| `bars.csv` | The frozen dataset (copy of the challenge file). |
| `TRANSCRIPT.md` | Step-by-step build transcript. |

## How to run

```bash
python3 run_backtest.py bars.csv   # final config -> results.json
python3 backtest.py bars.csv       # base / harsh / zero-cost summary
python3 optimize.py bars.csv       # full 144-combo grid search
```

Requires Python 3.10+ with `pandas` and `numpy`. No other dependencies.

## Costing assumptions (read this)

The exact sandbox numbers are **not published** anywhere (repo, bounty
brief, or skill.md — they only say "maker and taker fees, plus slippage
scaled by size"). This backtest therefore uses documented, Binance-style
assumptions and is checked for survival under much harsher ones:

| Scenario | Taker | Maker | Slippage |
|---|---|---|---|
| **base** | 10 bps | 2 bps | 0.5 bp + 15 bp per 100% of bar volume (cap 30 bp) |
| **harsh** | 15 bps | 5 bps | 1 bp + 45 bp per 100% of bar volume (cap 50 bp) |
| **extreme** | 20 bps | 10 bps | 2 bp + 75 bp per 100% of bar volume (cap 80 bp) |

Starting capital is assumed **$10,000**; sizing is a fraction of equity,
so the strategy scales with whatever capital the sandbox uses.

## Results (final config, $10,000 capital)

| Scenario | Net PnL | Max drawdown | Trades | Win rate | Profit factor |
|---|---|---|---|---|---|
| Base costs | **+$631.54 (+6.32%)** | **$174.05 (1.71%)** | 6 | 66.7% | 7.92 |
| Harsh costs | +$536.47 | $173.47 (1.71%) | 6 | 66.7% | — |
| Extreme costs | +$399.61 | $172.65 (1.71%) | 6 | 66.7% | — |
| Zero costs | +$715.55 gross | — | 6 | 66.7% | — |
| First half (bars 0–719) | $0.00 (no trades) | — | 0 | — | — |
| Second half (bars 720–1439) | +$631.54 | — | 6 | 66.7% | — |

Fees (base): $79.21. Slippage (base): $4.80. Avg win +$180.69,
avg loss −$45.61.

Trades (base costs): 5 shorts + 1 long. Best trade: short bar 941,
entry $1,936.79 → TP $1,900.43, **+$254.05** (117 bars, caught the
hour-15 −1.99% drop). Worst: short bar 841, stopped after 1 bar,
−$87.72. All profit comes from the trending second half; the strategy
refuses the chop — that is the point.

## Sizing note (notional cap)

Position size = 1% risk ÷ stop distance, capped at 2× equity notional.
The cap matters because 1-min stops are tight ($6–$15): at a 1× cap the
strategy cannot deploy its planned 1% risk on most trades (PnL $376.93
at 1× vs $631.54 at 2×, same 6 trades, DD 0.86% vs 1.71%). The dollar
risk per trade stays ≤1% either way — the cap only limits how much of
the planned risk gets deployed. At 3×: $788.09 net, DD 2.25% (not used;
diminishing returns and more slippage exposure if the real sandbox
charges more than modeled).

## Risks / limitations (honest)

- **Small sample:** 6 trades. The edge is principled (trend + FVG +
  confirmation + regime gate), not hardcoded to bars, but 6 trades is
  still 6 trades.
- **Regime concentration:** all profit comes from one downtrend leg. If
  the grader's held-out tail is choppy, the ADX gate keeps the strategy
  flat (no loss, but no profit either). If the tail trends up, the
  symmetric long logic applies.
- **Unpublished costs:** if the real sandbox charges far more than the
  "extreme" scenario above, the edge shrinks; the strategy stays
  positive down to roughly 3× the base fee level.
- **No leverage beyond the 2× notional cap** and no martingale /
  averaging down anywhere in the logic.
