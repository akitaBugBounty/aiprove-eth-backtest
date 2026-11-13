"""Grid search over strategy parameters with robustness gates.

Ranks by full-sample net PnL (base costs) but flags a parameter set as
ROBUST only if it is also profitable on both halves of the sample and
under harsh costs — the bounty README warns a held-out tail punishes
overfitting, so the final pick is the best ROBUST set, not the best
full-sample set.
"""

from __future__ import annotations

import itertools
import json
import sys

import pandas as pd

import strategy as st
from backtest import COST_BASE, COST_HARSH, backtest

GRID = dict(
    min_leg_bps=[35.0, 50.0, 70.0],
    require_sweep=[False, True],
    confirm=["close_or_engulf", "engulf_or_displacement"],
    tp_mode=["fixed", "fixed_and_trail", "trail"],
    entry_mode=["market", "limit"],
    rr=[2.0, 3.0],
)

FIXED = dict(min_sl_bps=30.0, max_sl_bps=150.0, risk_pct=0.01,
             max_hold_bars=240, trend_filter=True, sl_buffer_atr=0.25,
             trail_after_r=1.0, trail_atr_mult=2.0)


def eval_params(raw: pd.DataFrame, params: dict) -> dict:
    sig = st.generate_signals(raw, params)
    full = backtest(sig, params, cost=COST_BASE)
    half = len(raw) // 2
    h1 = backtest(sig.iloc[:half], params, cost=COST_BASE)
    h2 = backtest(sig.iloc[half:], params, cost=COST_BASE)
    harsh = backtest(sig, params, cost=COST_HARSH)
    return dict(params={k: params[k] for k in GRID},
                full_net=full["net_pnl"], full_dd=full["max_drawdown"],
                full_dd_pct=full["max_drawdown_pct"], full_trades=full["n_trades"],
                full_wr=full["win_rate"],
                h1_net=h1["net_pnl"], h2_net=h2["net_pnl"],
                harsh_net=harsh["net_pnl"],
                robust=bool(h1["net_pnl"] > 0 and h2["net_pnl"] > 0
                            and harsh["net_pnl"] > 0 and full["n_trades"] >= 5))


def main():
    raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "bars.csv")
    results = []
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"{len(combos)} combos ...", flush=True)
    for ci, combo in enumerate(combos):
        params = dict(st.DEFAULT_PARAMS, **FIXED,
                      **dict(zip(keys, combo)))
        try:
            r = eval_params(raw, params)
        except Exception as exc:  # noqa: BLE001 - keep the sweep alive
            print(f"combo {combo} failed: {exc}", file=sys.stderr)
            continue
        results.append(r)
        if (ci + 1) % 24 == 0:
            print(f"  {ci + 1}/{len(combos)} done", flush=True)
    results.sort(key=lambda r: (-r["full_net"], r["full_dd"]))
    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== TOP 15 by full-sample net PnL ===")
    for r in results[:15]:
        print(f"net ${r['full_net']:8.2f} dd ${r['full_dd']:7.2f} "
              f"({r['full_dd_pct']}%) tr {r['full_trades']:3d} wr {r['full_wr']:5.1f}% "
              f"h1 ${r['h1_net']:8.2f} h2 ${r['h2_net']:8.2f} harsh ${r['harsh_net']:8.2f} "
              f"robust={r['robust']} {r['params']}")
    robust = [r for r in results if r["robust"]]
    print(f"\n=== {len(robust)} ROBUST sets (profitable both halves + harsh) ===")
    for r in robust[:10]:
        print(f"net ${r['full_net']:8.2f} dd ${r['full_dd']:7.2f} "
              f"({r['full_dd_pct']}%) tr {r['full_trades']:3d} wr {r['full_wr']:5.1f}% "
              f"h1 ${r['h1_net']:8.2f} h2 ${r['h2_net']:8.2f} harsh ${r['harsh_net']:8.2f} "
              f"{r['params']}")


if __name__ == "__main__":
    main()
