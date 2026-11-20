#!/usr/bin/env python3
"""Behavioural checks on the direct corpus, as properties rather than as field values.

Each check is something that, if it broke, would make the scored table wrong while every
file still parsed. That is the failure mode these guard: a corpus that looks valid and
measures a deployment nobody ran.

Run: python3 scripts/test_direct_corpus.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "testdata" / "measurements"
FAILURES = []


def ck(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def main() -> int:
    direct = json.loads((HERE / "inferencex_direct.json").read_text())
    corpus = json.loads((HERE / "inferencex_direct_corpus.json").read_text())
    absol = json.loads((HERE / "inferencex_direct_absolutes.json").read_text())
    sett = json.loads((HERE / "inferencex_direct_settings.json").read_text())
    hopper = json.loads((HERE / "inferencex_direct_hopper_corpus.json").read_text())

    # The four files describe one measurement. A sweep present in one and absent from
    # another would be scored against the wrong row, or silently dropped.
    files = (("direct", direct["sweeps"]), ("corpus", corpus["sweeps"]),
             ("absolutes", absol["anchors"]), ("settings", sett["settings"]))
    keys = {f: {(s["scenario"], s["label"]) for s in d} for f, d in files}
    ck("all four files cover the same sweeps",
       len({frozenset(v) for v in keys.values()}) == 1,
       " ".join(f"{f}={len(v)}" for f, v in keys.items()))

    # A relative is the point's absolute over the anchor's. If these drifted, shape
    # would be computed against one measurement and mape against another.
    worst = 0.0
    for s in corpus["sweeps"]:
        a = next(x for x in absol["anchors"]
                 if (x["scenario"], x["label"]) == (s["scenario"], s["label"]))
        anchor = a["absolute_tpot_us"][str(s["points"][0]["concurrency"])]
        for p in s["points"]:
            want = a["absolute_tpot_us"][str(p["concurrency"])] / anchor
            worst = max(worst, abs(p["measured_tpot_relative"] / want - 1))
    ck("every relative equals its own absolute over the anchor", worst < 1e-5,
       f"worst deviation {worst:.2e}")

    # The anchor's relative is 1.0 by construction. A sweep whose first point is not the
    # lowest concurrency would make shape error meaningless.
    bad = [s["scenario"] for s in corpus["sweeps"]
           if s["points"][0]["concurrency"]
           != min(p["concurrency"] for p in s["points"])]
    ck("each sweep's first point is its lowest concurrency", not bad, str(bad[:3]))

    # No published prediction is carried, which is the table's defining property.
    nonzero = sum(1 for s in corpus["sweeps"] for p in s["points"]
                  if p["aisimulate_tpot_relative"] or p["aic_tpot_relative"])
    ck("no AISimulate or AIC prediction is carried", nonzero == 0, f"{nonzero} nonzero")

    # Speculative arms are kept and marked. Dropping them would discard a quarter of the
    # corpus; carrying them unmarked would score a multi-token step as single-token.
    spec = sum(len(s["points"]) for s in direct["sweeps"] if s["spec_method"] != "none")
    ck("speculative arms are present and labelled", spec > 0, f"{spec} points")

    # Only parts the registry has coefficients for.
    chips = {s["gpu"] for s in corpus["sweeps"]}
    ck("every chip has registry coefficients",
       chips <= {"h100", "h200", "b200", "b300"}, str(sorted(chips)))

    # The Hopper subset must be a strict subset, not a re-derivation that could drift.
    hk = {(s["scenario"], s["label"]) for s in hopper["sweeps"]}
    ck("the Hopper subset is a subset of the full corpus", hk <= keys["corpus"],
       f"{len(hk)} sweeps")
    ck("the Hopper subset is exactly the Hopper sweeps",
       hk == {(s["scenario"], s["label"]) for s in corpus["sweeps"]
              if s["gpu"] in ("h100", "h200")})

    # The two sparse-attention models must appear on Hopper: they are the reason this
    # corpus exists, being deployments NVIDIA publishes on Blackwell only.
    hop_models = {s["model"] for s in hopper["sweeps"]}
    ck("both sparse-attention models appear on Hopper",
       {"minimax-m3", "deepseek-v4-pro"} <= hop_models, str(sorted(hop_models)))

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("PASS: 0 failure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
