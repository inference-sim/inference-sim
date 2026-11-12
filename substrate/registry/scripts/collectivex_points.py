"""The points of an InferenceX CollectiveX run body that the DeepEP fit reads, and a digest of
them.

WHY NOT THE BODY'S BYTES. Entries used to cite a run by the sha256 of the body as served. On
2026-11-07 the API began serving all fourteen cited bodies with different bytes, and the fit
read from them the same points. A pin on bytes then fails the re-derivation gate for a change
that moves no value. The entries now cite the digest of the points the fit reads, so the gate
fails when a measurement changes and not when the serialisation does.

WHAT IS DIGESTED. For every series of backend `deepep-v2`, precision `bf16`, the row
(sku, mode, ep_size, nodes, phase, library_version) and, for each point, the tokens per rank
and the roundtrip p50 latency in microseconds. The rows are sorted and written as compact JSON
with sorted keys, so the digest does not depend on the order the API lists series or points.
A series that lacks a field the fit reads is an error and not a skipped row, so a body that
changes shape fails the gate.
"""

from __future__ import annotations

import hashlib
import json

BACKEND, PRECISION = "deepep-v2", "bf16"


def projection(body: dict) -> list[list]:
    """The rows the digest covers, sorted."""
    rows = []
    for s in body.get("series") or []:
        if (s.get("backend"), s.get("precision")) != (BACKEND, PRECISION):
            continue
        sy = s["system"]
        head = [sy["sku"], s["mode"], sy["ep_size"], sy["nodes"], s["phase"],
                s["library_version"]]
        for p in s["points"]:
            rows.append(head + [p["tokens_per_rank"],
                                p["components"]["roundtrip"]["latency_us"]["p50"]])
    return sorted(rows)


def digest(body: dict) -> str:
    """The sha256 of the projection, as 64 lower-case hex digits."""
    text = json.dumps(projection(body), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(raw: bytes) -> str:
    return digest(json.loads(raw))
