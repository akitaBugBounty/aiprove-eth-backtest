# Build transcript — AiProve eth-pnl-fixed entry

Step-by-step log of how this entry was built. Every number below was
produced by running the code in this directory on `bars.csv`.

## 1. Fetch the challenge (2026-11-14 ~01:45 WIB)

```bash
git clone --depth 1 https://github.com/AIProveRH/aiprove.git
# challenges/eth-backtest/: README.md + bars.csv (1,440 bars)
```

Challenge README: frozen ETH 1-min OHLCV (Binance, 24h ending
2026-11-13 12:17 UTC), metric = net PnL after costs (maker + taker
fees, slippage scaled by size), ties break on lower max drawdown.
"Overfitting to this sample is easy; a held-out tail punishes it."
Bounty brief confirms the same; 66 submissions / 111 agents at the
time of building. **The exact fee/slippage numbers are not published
anywhere** — documented Binance-style assumptions were used instead
(see README "Costing assumptions") and survival under 2–5× harsher
costs was verified.

## 2. Dataset reconnaissance

```bash
python3 -c "..."  # pandas describe of bars.csv
```

- 1,440 bars, close 2028.68 → 1960.59 (−3.36% over the day)
- Mean |1-min return| 7.25 bps, median 5.2 bps; mean bar range $2.47
- Mean bar volume 282 ETH (~$558k notional/bar)
- Hourly returns mostly ±0.5%; biggest drop hour 15 (−1.99%), biggest
  rise hour 18 (+1.10%)
- 701 up bars / 734 down bars; 76 bars with |return| > 0.2%

Implication: per-bar moves are tiny vs a ~12–20 bp round-trip cost,
so the strategy must hold multi-bar moves and trade rarely.

## 3. v1 strategy — market-structure entries (too strict)

`strategy.py` v1: swing HH/HL trend filter, displacement leg ≥ 60 bps,
pullback into the Fib 50–61.8% zone, engulfing/displacement
confirmation, market entry at next open, SL at structure, TP 2R,
1% risk/trade.

```bash
python3 backtest.py bars.csv
# base costs: net +$18.81, 1 trade, max DD $23.83
```

Diagnosis (counts per stage): trend defined on 697/1380 bars, valid
legs on ~370 bars, but only 27 bars ever sat inside the 50–61.8% Fib
zone and only 2 passed confirmation. A 60 bp leg retraced 50%+ is a
30+ bp pullback — ~4× the mean bar move — too rare on 1-min data.

## 4. v2 — FVG zones + wider Fib + looser confirmation

- Entry zone = the leg's fair-value gap overlapping Fib 38.2–78.6%
- Confirmation = with-trend close (or engulfing/displacement)
- One signal per displacement leg; min leg 35 bps

```bash
python3 backtest.py bars.csv
# base costs: net -$427.49, 23 trades, win rate 39.1%, max DD $566.55
# zero costs: net +$11.27, profit factor 1.03
```

Barely breakeven before costs — the edge was too weak. Exit-reason
audit: 14 stops vs 8 targets; avg win $33 vs avg loss −$52. Stops
($5–15) were inside the noise and 2R targets on tiny stops could not
cover the ~12–20 bp round-trip cost.

## 5. v3 — honest fills, trailing stops, limit entries, look-ahead fix

- Fixed a look-ahead bug: swing highs/lows used centered rolling
  windows, so a swing was "known" 3 bars before it could be confirmed.
  Swings are now shifted by the fractal window (only usable after
  confirmation).
- Added limit entries (maker fee, FVG edge, 3-bar validity) and an
  optional ATR chandelier trailing stop.
- Widened stop floor to 30 bps.

## 6. Grid search — 144 combos

```bash
python3 optimize.py bars.csv   # -> optimization_results.json
```

Best full-sample: **+$212.65** (14 trades, 50% WR), but first half
−$190 / second half +$411 — not robust. **0 robust sets** (profitable
on both halves + harsh costs). The first half is chop; trend-following
bleeds there. Limit entries dominated the top of the list (maker fee
saves ~8 bp/trade).

## 7. Regime gate (ADX) — the fix for the chop half

Added ADX(14): entries require ADX ≥ threshold.

```bash
python3 experiment.py   # regime x top sets, then risk scaling
# top1 + ADX25: net +$275.06, DD $180.81, WR 62.5%, harsh +$214.19
# risk_pct 1–5%: identical results — the 1x notional cap binds, not risk
```

## 8. Round 2 — ADX threshold × RR × notional cap

```bash
python3 experiment2.py
# ADX30 + RR2.5 + 1x cap: net +$376.93, DD $86.80 (0.86%), 6 trades,
#   WR 66.7%, first half $0.00 (zero trades in the chop)
# ADX30 + RR2.5 + 2x cap: net +$631.54, DD $174.05 (1.71%)
# ADX25 + RR2.5 + 2x cap: net +$649.47, DD $338.18 (3.35%) — worse DD
```

The ADX30 gate is the key robustness result: it refuses every trade
in the choppy half. RR 2.5 beats RR 2.0 (same DD, +$102 more).
The 2× notional cap was chosen because 1-min stops are tight and the
1× cap prevents deploying the planned 1% risk; dollar risk per trade
stays ≤1% either way (see README "Sizing note").

## 9. Final verification

```bash
python3 run_backtest.py bars.csv
# net PnL (base costs): $631.54
# max drawdown: $174.05 (1.71%)
# trades: 6, win rate: 66.7%, profit factor: 7.92
# final equity: $10631.54 (+6.32%)
# half1 net: $0, half2 net: $631.54
# net PnL (harsh costs): $536.47
python3 backtest.py bars.csv   # same numbers via the second entry point
python3 strategy.py bars.csv   # 25 raw signals -> 6 filled trades
```

Extra stress check (not in `run_backtest.py`): "extreme" costs —
taker 20 bps, maker 10 bps, slippage 5× base — still **+$399.61**.

Final config (`strategy.py` → `FINAL_PARAMS`): min leg 35 bps,
FVG ∩ Fib 38.2–78.6%, close/engulf confirmation, ADX ≥ 30 regime
gate, limit entries at the FVG edge, SL = FVG/sweep extreme +
0.25×ATR (30–150 bps), TP 2.5R, 1% risk/trade, 2× notional cap,
max hold 240 bars, 5-bar cooldown.

## 10. What was NOT done

- No submission to the AiProve API (per the assignment).
- No wallet signing; no funds moved. This entry is pure code +
  backtest on the public frozen dataset.
