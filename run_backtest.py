"""Entry point: run the final strategy + backtest and write results.json.

Usage:
    python3 run_backtest.py [bars.csv]
"""

import json
import sys

import pandas as pd

import strategy as st
from backtest import COST_BASE, COST_HARSH, backtest


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bars.csv"
    raw = pd.read_csv(path)
    params = dict(st.FINAL_PARAMS)
    sig = st.generate_signals(raw, params)

    out = {"params": params, "capital": 10_000.0, "costs": {}}
    for name, cost in [("base", COST_BASE), ("harsh", COST_HARSH)]:
        out["costs"][name] = cost
        r = backtest(sig, params, cost=cost)
        out[name] = {k: v for k, v in r.items() if k != "equity_curve"}
        out[name]["equity_curve"] = r["equity_curve"]

    half = len(raw) // 2
    out["half1"] = {k: v for k, v in
                    backtest(sig.iloc[:half], params, cost=COST_BASE).items()
                    if k not in ("equity_curve", "trades")}
    out["half2"] = {k: v for k, v in
                    backtest(sig.iloc[half:], params, cost=COST_BASE).items()
                    if k not in ("equity_curve", "trades")}

    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

    b = out["base"]
    print(f"net PnL (base costs): ${b['net_pnl']}")
    print(f"max drawdown: ${b['max_drawdown']} ({b['max_drawdown_pct']}%)")
    print(f"trades: {b['n_trades']}  win rate: {b['win_rate']}%  "
          f"profit factor: {b['profit_factor']}")
    print(f"final equity: ${b['final_equity']} ({b['return_pct']}%)")
    print(f"half1 net: ${out['half1']['net_pnl']}  half2 net: ${out['half2']['net_pnl']}")
    print(f"net PnL (harsh costs): ${out['harsh']['net_pnl']}")
    print("results.json written")


if __name__ == "__main__":
    main()
