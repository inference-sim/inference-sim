"""Compact text tables from a perreq_diag.py / perreq_eval.py output (usage: perreq_table.py <json>)."""
import json
import sys

d = json.load(open(sys.argv[1]))["summary"]
print(f"points {d['points']} runs {d['runs']}")
for metric in ("r_ttft", "r_itl"):
    for key, bins in d[metric].items():
        cells = "  ".join(f"{b}: {v['median_pct']:+.1f} [{v['ci_pct'][0]:+.0f},{v['ci_pct'][1]:+.0f}] r{v['runs']} n{v['n']}"
                          for b, v in bins.items())
        print(f"{metric} by {key}: {cells}")
s = d["ttft_context_slope"]
print("ttft context slope (BLIS-meas ms per 1k prior ctx, new<4k):", {k: v for k, v in s.items() if k != "per_run"})
for scope, ms in d["distance"].items():
    print(scope, "  ".join(f"{k}: W1 {v['w1_norm']['mean']:.2f} KS {v['ks']['mean']:.2f} q50 {v['q50_err_pct']['mean']:.0f} "
                           f"q90 {v['q90_err_pct']['mean']:.0f} q99 {v['q99_err_pct']['mean']:.0f} (r{v['w1_norm']['runs']})"
                           for k, v in ms.items() if v["w1_norm"]["mean"] is not None))
