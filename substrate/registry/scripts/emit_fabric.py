#!/usr/bin/env python3
"""Write coefficients/cost-model-fabric.yaml: each inter-node fabric's effective / nominal
bandwidth (inference-sim/blis-registry#60, gap 3).

    python scripts/emit_fabric.py            # rewrite the file
    python scripts/emit_fabric.py --check    # exit 1 if the file would change
    python scripts/emit_fabric.py --verify DIR   # also check the cited public bodies' sha256

THE QUANTITY. `fabric_bandwidth_efficiency`, dimensionless, scoped {fabric: [<networks/ name>]}:
the payload throughput a transfer over that fabric achieves, divided by the catalog's nominal
InterNodeBwGBps (line rate / 8, per GPU, per direction: roce-200g 25, ib-400g 50,
ethernet-100gbe 12.5 GB/s). blis-latency-kernel (fabric.go, fabricEfficiencyName) multiplies
the NIC rate a P/D handoff or pipeline boundary crosses by it, and scales the intra-to-inter
ratio of a cross-node collective, and refuses a value outside (0, 1]. The catalog itself
states only the nominal figure and says the correction belongs here (networks/*.yaml
_comment_interconnect, blis-catalog eb675ef).

ONE NUMBER PER FABRIC, AND WHAT IT IS NOT. A fabric reaches different fractions of line rate
for different traffic: one large RDMA write stream comes close to line rate, while a
multi-node NCCL collective at default settings gets less. The kernel reads one value per fabric
for all of it, so each entry states which evidence sets its value and gives the range the
other evidence spans. None of these is a measurement this project made, and none is `measured`.

THE ESTIMATOR FOR ib-400g, declared before the figures were read off. NVIDIA/nccl-tests#309
(https://github.com/NVIDIA/nccl-tests/issues/309, opened 2025-05-20): Dell XE9680 hosts,
8x H100 and 8x ConnectX-7 400 Gb/s InfiniBand per host, NCCL 2.26.2. An NVIDIA maintainer
states the multi-node all_reduce busbw ceiling for 8 such NICs as 400 GB/s and that results
from 2 nodes "tend to be an aberration" (the 2-node rows, 469 and 467 GB/s, exceed it).
Estimator: the median, over the issue's all_reduce rows from at least 4 hosts at default NCCL
settings, of busbw / 400 GB/s. The tuned row (NCCL_MIN_CTAS=24, NCCL_IB_QPS_PER_CONNECTION=2)
and the per-NIC ib_send_bw (396 Gb/s) bound the range from above. Method `literature`.

WHY NOT CollectiveX FOR ib-400g. InferenceX CollectiveX's h200-dgxc runs DeepEP EP16 over two
nodes on ConnectX-7 400G InfiniBand. But those nodes are KVM guests, where (its methodology
records) strict-ordered GPU-NIC writes cap near 10 GB/s per GPU; its normal-mode rows run a
relaxed-ordering patch, and its low-latency rows ride NVSHMEM IBGDA without stated gdrdrv. So
they measure that pool's path, not the fabric. The bare-metal b200-nscale EP16 rows cannot
isolate the fabric either: CollectiveX reports logical payload (one copy per token and
destination rank) over latency, while DeepEP v2's hybrid dispatch sends a token across nodes
once per destination node and fans it out over NVLink, so the bytes that cross InfiniBand are
not in the row.

roce-200g AND ethernet-100gbe ARE ASSUMED. No public multi-node measurement of an HGX
RoCE-200G rail or of NCCL over 100 GbE TCP was found. The one public RoCEv2 measurement used
is a two-node DGX Spark (GB10) pair, cabled directly with no switch, which bounds RoCE from
above but is not the catalog's fabric, so it is not `measured` or `literature`.

Every cited page is pinned by the sha256 of what was read: for a GitHub issue, the UTF-8
`body` of each issue or comment the GitHub API returns; for a web page, the bytes served.
`--verify DIR` recomputes them from DIR/nccl-tests-309-issue.json,
DIR/nccl-tests-309-comments.json (the GitHub API's JSON) and DIR/route179-dgx-spark.html.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET_PATH = HERE.parent / "coefficients" / "cost-model-fabric.yaml"
CATALOG = HERE.parents[1] / "catalog"

ISSUE = "https://github.com/NVIDIA/nccl-tests/issues/309"
# (what, GitHub id or "issue", sha256 of its body as the GitHub API returns it)
BODIES = {
    "issue": ("issue", "bc8137556e2ce25c651cd982f9958203b3ca96797543b4aeb34a997118b835ac"),
    "qoofyk-4host": (2904845294,
                     "e43aa547046cc3858c77f474f4c216aa671a85c6e803f44790aebdbfdb96ce89"),
    "gj-aion-8node": (2982333665,
                      "d40a242c3f613ca4199a60faec0ad1a8d719e3b297ff6d3609c356543bbc2cfb"),
    "gj-aion-4of8": (2983888014,
                     "99ed9844c145eda7047b6e712d29ffafb0cc8d3762f92c54bc3b52c6615aab9d"),
    "nvidia-peak": (2984447704,
                    "b3eb1ffbccb53470190bbb2927a5a7b2cdf0d8a5edaca4501deb6bd3fca9e8e0"),
    "gj-aion-tuned": (2984669604,
                      "41ab4e0975ba219142ac3d98d61d24cf280e3de2a03ea7a8e474059e53e1cdde"),
}
# The all_reduce rows the estimator reads, transcribed from those bodies: (body, hosts,
# NCCL settings, message bytes, out-of-place busbw GB/s).
ROWS = [
    ("qoofyk-4host", 4, "default", 17179869184, 365.84),
    ("gj-aion-4of8", 4, "default", 8589934592, 324.52),
    ("gj-aion-8node", 8, "default", 8589934592, 320.70),
    ("gj-aion-tuned", 8, "NCCL_MIN_CTAS=24 NCCL_IB_QPS_PER_CONNECTION=2", 8589934592, 391.37),
]
EXCLUDED_2HOST = [("issue", 2, 17179869184, 469.21), ("gj-aion-8node", 2, 8589934592, 466.73)]
PEAK_BUSBW = 400.0          # GB/s, 8 x 400 Gb/s NICs, as the NVIDIA maintainer states it
IB_SEND_BW_GBPS = 396.0     # Gb/s, ib_send_bw --use_cuda, one ConnectX-7 (issue body)
NIC_GBPS = 400.0

ROUTE179 = "https://route179.dev/2026/07/21/dgx-spark-nccl-roce-benchmarking"
ROUTE179_SHA = "07379fe826ab5049065f087054d6dab0778ca0526101c47b86c47ded04761baf"
SPARK_ALLGATHER_BUSBW = 24.20   # GB/s, all_gather in-place busbw at 16 GiB, 2 ranks
SPARK_WRITE_GBPS = 98.04 * 2    # Gb/s, ib_write_bw, two links concurrently
SPARK_NOMINAL_GBPS = 200.0
SPARK_TCP_FALLBACK_GBS = 3.0    # GB/s, "~3 GB/s instead of ~20" when NCCL fell back to TCP

NOMINAL = {"roce-200g": 25.0, "ib-400g": 50.0, "ethernet-100gbe": 12.5}

HEADER = """\
# Inter-node fabric bandwidth: the effective / nominal correction per fabric class.
#
# GENERATED by scripts/emit_fabric.py. Re-run it and commit the diff rather than editing
# values here; `emit_fabric.py --check` is the drift gate.
#
# One entry per blis-catalog networks/ fabric, scoped by the fabric's catalog identity
# (the `fabric` scope key, blis-schemas#112). blis-latency-kernel multiplies a fabric's
# nominal InterNodeBwGBps by it wherever bytes leave a node. An NVLink domain spanning a
# rack (gb200-nvl72) is the chip's, not a fabric's, and is not corrected here.
#
# None of these is a measurement this project made: ib-400g is `literature` (published
# nccl-tests results), roce-200g and ethernet-100gbe are `assumed`, each with its range.
kind: CoefficientSet
name: cost-model-fabric
coefficients:
"""


def ib_400g() -> dict:
    default = [r for r in ROWS if r[1] >= 4 and r[2] == "default"]
    fractions = [busbw / PEAK_BUSBW for *_, busbw in default]
    value = round(statistics.median(fractions), 3)
    tuned = max(busbw for *_, busbw in ROWS) / PEAK_BUSBW
    raw = IB_SEND_BW_GBPS / NIC_GBPS
    rows = "; ".join(f"{h} hosts {b:.2f} GB/s ({b / PEAK_BUSBW:.3f})"
                     for _, h, s, _, b in default)
    rat = (
        f"Published payload throughput over nominal for 400 Gb/s InfiniBand NDR, one ConnectX-7 "
        f"per GPU (catalog InterNodeBwGBps 50). From NVIDIA/nccl-tests#309 (2025-05-20; Dell "
        f"XE9680, 8x H100 and 8x ConnectX-7 400G IB per host, NCCL 2.26.2): the median of "
        f"all_reduce busbw over the 400 GB/s ceiling an NVIDIA maintainer states for 8 such "
        f"NICs, across the issue's rows from at least 4 hosts at default NCCL settings ({rows}) "
        f"= {value}. The 2-host rows ({', '.join(f'{b:.2f}' for *_, b in EXCLUDED_2HOST)} "
        f"GB/s) exceed the ceiling and are excluded, as the maintainer calls 2-node results an "
        f"aberration. The estimator was declared before the figures were read off "
        f"(scripts/emit_fabric.py). RANGE {min(fractions):.2f}-{raw:.2f}: the upper end is one "
        f"NIC's GPUDirect ib_send_bw (396 Gb/s), the regime of a single large RDMA write such "
        f"as a KV handoff; tuned NCCL (NCCL_MIN_CTAS=24, NCCL_IB_QPS_PER_CONNECTION=2) reached "
        f"{tuned:.3f} on 8 nodes. The value is collective-level, so it understates a large "
        f"point-to-point transfer by up to {raw / value - 1:.0%}. Not from InferenceX "
        f"CollectiveX: its h200-dgxc InfiniBand pool is KVM-virtualised (strict-ordered "
        f"GPU-NIC writes cap near 10 GB/s per GPU there) and its rows report logical, not "
        f"wire, payload, so they do not isolate the fabric. literature, not measured: "
        f"published results this project has not reproduced, on H100 hosts only.")
    cite = (f"{ISSUE}: issue body sha256 {BODIES['issue'][1]} (ib_send_bw 396 Gb/s, 2-host "
            f"469.21 GB/s); comments "
            + ", ".join(f"{BODIES[k][0]} sha256 {BODIES[k][1]}"
                        for k in ("qoofyk-4host", "gj-aion-8node", "gj-aion-4of8",
                                  "nvidia-peak", "gj-aion-tuned"))
            + " (GitHub API body text)")
    return {"fabric": "ib-400g", "value": value, "method": "literature", "rationale": rat,
            "sources": [("publication", cite, "primary")]}


def roce_200g(ib: float) -> dict:
    ag = SPARK_ALLGATHER_BUSBW / NOMINAL["roce-200g"]
    wr = SPARK_WRITE_GBPS / SPARK_NOMINAL_GBPS
    rat = (
        f"ASSUMED: no public multi-node measurement of a RoCE-200G rail on an HGX host was "
        f"found, and InferenceX CollectiveX runs no RoCE pool. Taken equal to the ib-400g "
        f"collective-level fraction, {ib}, on the reasoning that RoCEv2 over a switched "
        f"Ethernet fabric does not outperform InfiniBand at scale (DeepEP's README: fully "
        f"tested with InfiniBand, theoretically compatible with RoCE). RANGE 0.70-{ag:.2f}. "
        f"The upper end is the one public RoCEv2 measurement found: two DGX Spark (GB10) "
        f"nodes, cabled directly with no switch, two 100G links merged (NCCL 2.28.9): "
        f"all_gather busbw {SPARK_ALLGATHER_BUSBW} GB/s at 16 GiB over 25 GB/s nominal = "
        f"{ag:.3f}, and ib_write_bw 2 x 98.04 Gb/s over 200 = {wr:.3f}. That is a "
        f"best case with no switch, two ranks and no congestion, not this fabric, so it bounds "
        f"the value and does not set it. The lower end allows for congestion and PFC/ECN "
        f"tuning on a shared lossless Ethernet fabric, which no public figure pins. Replace "
        f"with a measurement when one exists.")
    cite = (f"{ROUTE179} (2026-07-21; page sha256 {ROUTE179_SHA}): 2x DGX Spark RoCEv2, "
            f"ib_write_bw 98.04 + 98.04 Gb/s, all_gather_perf busbw 24.20 GB/s at 16 GiB")
    return {"fabric": "roce-200g", "value": ib, "method": "assumed", "rationale": rat,
            "sources": [("discussion", cite, "supporting")]}


def ethernet_100gbe() -> dict:
    value = 0.5
    rat = (
        f"ASSUMED, with no measurement: plain 100 GbE with TCP (no RDMA), so the CPU stays in "
        f"the data path and NCCL runs its socket transport. No public, pinned measurement of "
        f"NCCL or a KV transfer over 100 GbE TCP between GPU hosts was found. {value} is the "
        f"midpoint of a wide RANGE 0.2-0.9, chosen to say that the rate is well below line "
        f"rate without claiming a figure. The one public datum, the RoCE page above, saw NCCL "
        f"silently fall back to TCP at about {SPARK_TCP_FALLBACK_GBS:.0f} GB/s on a 25 GB/s "
        f"link ({SPARK_TCP_FALLBACK_GBS / 25:.2f}). That was a misconfigured fallback on a "
        f"different link, so it shows how low TCP can go and does not set the value. Replace "
        f"with a measurement when one exists.")
    cite = (f"{ROUTE179} (2026-07-21; page sha256 {ROUTE179_SHA}): NCCL's silent TCP "
            f"fallback, about 3 GB/s against about 20 GB/s over RDMA on the same link")
    return {"fabric": "ethernet-100gbe", "value": value, "method": "assumed", "rationale": rat,
            "sources": [("discussion", cite, "supporting")]}


def entry_lines(e: dict) -> list[str]:
    out = [
        "  - fabric_bandwidth_efficiency:",
        f"      value: {e['value']}",
        "      units: dimensionless",
        f"      method: {e['method']}",
        "      fitted: false",
        f"      scope: {{fabric: [{e['fabric']}]}}",
    ]
    if e["sources"]:
        out.append("      sources:")
        for kind, cite, role in e["sources"]:
            out.append(f'        - {{kind: {kind}, cite: "{cite}", role: {role}}}')
    out += ["      rationale: >", f"        {e['rationale']} Reproduce with "
            f"scripts/emit_fabric.py."]
    return out


def render() -> str:
    fabrics = sorted(p.stem for p in (CATALOG / "networks").glob("*.yaml"))
    ib = ib_400g()
    entries = {"ib-400g": ib, "roce-200g": roce_200g(ib["value"]),
               "ethernet-100gbe": ethernet_100gbe()}
    if sorted(entries) != fabrics:
        raise SystemExit(f"the vendored catalog's fabrics {fabrics} are not the ones this "
                         f"writer prices {sorted(entries)}; add the new fabric here")
    import yaml
    for f in fabrics:
        nominal = yaml.safe_load((CATALOG / "networks" / f"{f}.yaml").read_text())
        if float(nominal["InterNodeBwGBps"]) != NOMINAL[f]:
            raise SystemExit(f"{f}: catalog InterNodeBwGBps {nominal['InterNodeBwGBps']} is "
                             f"not the {NOMINAL[f]} this writer's rationale divides by")
    lines = []
    for f in fabrics:
        lines += entry_lines(entries[f])
    return HEADER + "\n".join(lines) + "\n"


def verify(d: Path) -> list[str]:
    bad = []
    issue = json.loads((d / "nccl-tests-309-issue.json").read_text())
    comments = {c["id"]: c for c in json.loads((d / "nccl-tests-309-comments.json").read_text())}
    for k, (cid, sha) in BODIES.items():
        body = issue["body"] if cid == "issue" else comments[cid]["body"]
        if hashlib.sha256(body.encode()).hexdigest() != sha:
            bad.append(f"{k} ({cid})")
    page = (d / "route179-dgx-spark.html").read_bytes()
    if hashlib.sha256(page).hexdigest() != ROUTE179_SHA:
        bad.append("route179 page")
    return bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the file would change; write nothing")
    ap.add_argument("--verify", metavar="DIR",
                    help="recompute the cited bodies' sha256 from saved copies in DIR")
    args = ap.parse_args(argv[1:])
    if args.verify:
        bad = verify(Path(args.verify))
        if bad:
            print(f"cited bodies differ from their recorded sha256: {', '.join(bad)}",
                  file=sys.stderr)
            return 1
        print("every cited body matches its recorded sha256")
    after = render()
    for line in after.splitlines():
        if line.startswith("      value:"):
            print(line.strip())
    before = SET_PATH.read_text(encoding="utf-8") if SET_PATH.exists() else ""
    if args.check:
        if after != before:
            print(f"{SET_PATH.name} would change", file=sys.stderr)
            return 1
        print(f"{SET_PATH.name} already matches")
        return 0
    SET_PATH.write_text(after, encoding="utf-8")
    print(f"{SET_PATH.name} written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
