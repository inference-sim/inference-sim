"""The rows of an InferenceX OperatorX MoE response that the Marlin NVFP4 ramp fit reads, and a
digest of them.

WHY NOT THE RESPONSE'S BYTES. The two entries once cited the sha256 of the response as served.
By 2026-11-05 the same URL served a different response, and the fit read the same rows from it
and gave the same values; the pin on bytes made the re-derivation unrunnable for a change that
moved no value. The entries now cite the digest of the rows
the fit reads, so the gate fails when a measurement changes and not when the response grows a
field or a case the fit ignores.

WHAT IS DIGESTED. For each chip, every case that the fit keeps (status ok, kernel MarlinExperts,
NVFP4 experts, tp 1, the cited run): its token count, whether it ran under a CUDA graph, and its
latency in microseconds. The rows are sorted and written as compact JSON, so the digest does not
depend on the order of cases. A case whose geometry is not the cited one is an error and not a
skipped row, so a response that changes shape fails the gate.
"""

from __future__ import annotations

import hashlib
import json

CHIPS = ("h100", "h200")
RUNS = {"h100": "36292796863", "h200": "36292840239"}
GEOM = (6144, 3072, 4, 128)          # hidden, inter, top_k, experts
SHARED_INTER = 3072


def rows(d: dict, chip: str) -> list[tuple[int, bool, float]]:
    """(tokens, cuda graph, latency in seconds) of the cases the fit keeps for `chip`."""
    m = d["measurements"][chip]
    out = []
    for i, c in enumerate(d["cases"]):
        k = json.loads(c["key"])
        w, e = k["experts"]["w1"], k["experts"]
        if m["status"][i] != "ok" or d["kernels"][m["kernel"][i]] != "MarlinExperts":
            continue
        if (w["dtype"], w["scale"]["dtype"], w["scale"]["group"][1]) != ("e2m1", "e4m3", 16):
            continue
        if k.get("parallel") or m["runId"][i] != RUNS[chip]:
            continue
        if (k["hidden"], e["inter"], e["top_k"], e["num"]) != GEOM:
            raise SystemExit(f"{chip}: geometry moved: {k['hidden'], e['inter'], e['top_k'], e['num']}")
        sh = k["shared"]
        if (sh["count"], sh["inter"], sh["w1"]["dtype"], sh["w1"]["scale"]["dtype"]) != (
                1, SHARED_INTER, "e4m3", "ue8m0"):
            raise SystemExit(f"{chip}: shared expert moved: {sh}")
        out.append((k["tokens"], bool(m["cudaGraph"][i]), m["latencyUs"][i] * 1e-6))
    return sorted(out)


def digest(d: dict) -> str:
    """The sha256 of the kept rows of both chips, as 64 lower-case hex digits."""
    rs = [[chip, t, g, lat] for chip in CHIPS for t, g, lat in rows(d, chip)]
    text = json.dumps(rs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(raw: bytes) -> str:
    return digest(json.loads(raw))
