#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash CUDA-graph capture memory: the v2.3 floor and per-token entries,
fitted to vLLM's own start-up logs.

    python scripts/fit_deepseek_v41_capture.py             # print, rewrite the memory set
    python scripts/fit_deepseek_v41_capture.py --check     # exit 1 if the set would change
    python scripts/fit_deepseek_v41_capture.py --extract-anchor <ix_dsv41 dir>
        # re-derive the vendored start-ups from local copies of the InferenceX logs

Owns, in cost-model-memory.yaml, scope {hardware: [h100, h200], model: [deepseek-v4.1-flash]}:

  cudagraph_capture_floor_bytes_full_and_piecewise
  cudagraph_capture_bytes_per_mixed_token_full_and_piecewise
  cudagraph_capture_bytes_per_decode_token_full_and_piecewise

The kernel reads the three all or none (blis-latency-kernel captureMemory) as

  floor + per_mixed_token x C_mixed + per_decode_token x C_decode

with C_mixed the capture ceiling and C_decode the largest uniform-decode FULL graph, each
capture size rounded up to 1 + K and kept within max_num_seqs requests and the ceiling.

THE DATA (validator/testdata/anchors/v41_h200_tp8_capture_startups.json, vendored derived
figures; the logs are not redistributed). Five InferenceX H200 TP8 start-ups of
DeepSeek-V4.1-Flash, K = 5, max_num_seqs 1024 (default), mnbt 8192, at capture ceilings 64,
128, 256, 512 and 1024. From each, rank 0's

  E  "CUDA graph pool memory: A GiB (actual), E GiB (estimated)", what vLLM subtracts
  M  the profiling pass's "Graph capturing finished ... took M GiB"

THE SPLIT, from vLLM v0.31.0 source, not fitted. profile_cudagraph_memory captures every
PIECEWISE and drafter graph and only the two largest FULL graphs, then returns
measured - sum(samples) + first + (n_full - 1) x second
(vllm/v1/worker/gpu/cudagraph_utils.py:928-937, :959-967), so E - M = (n_full - 2) x second
is the uniform-decode FULL family alone. The decode slope is E - M against C_decode through
the origin; the floor and the mixed slope are a least-squares line of M against C_mixed. The
fit arm holds max_num_seqs at 1024, so it cannot separate the slopes; M also holds the drafter
and the two sampled FULL graphs, which follow the decode ceiling, so this split understates
the max_num_seqs sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coefficient_blocks as cb  # noqa: E402

REPO = HERE.parent
MEMORY = REPO / "coefficients" / "cost-model-memory.yaml"
ANCHOR = REPO / "validator" / "testdata" / "anchors" / "v41_h200_tp8_capture_startups.json"
GIB = 1 << 30
MODE = "full_and_piecewise"
NAMES = (f"cudagraph_capture_floor_bytes_{MODE}",
         f"cudagraph_capture_bytes_per_mixed_token_{MODE}",
         f"cudagraph_capture_bytes_per_decode_token_{MODE}")
SCOPE = "{hardware: [h100, h200], model: [deepseek-v4.1-flash]}"
SOURCE = ("https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35494445931 "
          "(head bc2b235fc26a), server.log of each tp8 artifact")
BUILD = "vLLM 0.29.1rc1.dev128+gcd10ed6f9"


def capture_sizes(ceiling: int, mnbt: int) -> list[int]:
    """vLLM v0.31.0's default capture list (vllm/config/vllm.py:2563-2595)."""
    c = min(ceiling, mnbt)
    s = [i for i in (1, 2, 4) if i <= c]
    if c >= 8:
        s += list(range(8, min(c + 1, 256), 8))
    if c >= 256:
        s += list(range(256, c + 1, 16))
    if mnbt <= c:
        s.append(mnbt)
    return sorted(set(s))


def decode_graphs(ceiling: int, mnbt: int, seqs: int, q: int) -> list[int]:
    """Uniform-decode FULL graphs (_init_candidates, cudagraph_utils.py:313-327)."""
    c = min(ceiling, mnbt)
    return sorted({-(-s // q) * q for s in capture_sizes(c, mnbt)
                   if -(-s // q) * q <= min(c, seqs * q)})


def extract_anchor(ix: Path) -> int:
    binding = json.loads((ix / "artifact_binding.json").read_text())
    pool = re.compile(r"TP0 .*CUDA graph pool memory: ([\d.]+) GiB \(actual\), "
                      r"([\d.]+) GiB \(estimated\)")
    took = re.compile(r"TP0 .*Graph capturing finished in \d+ secs?, took ([\d.]+) GiB")
    rows, seen = [], set()
    for c, v in sorted(binding["h200-tp8-0921"].items(), key=lambda x: int(x[0])):
        p = ix / v["bound"]["dir"] / "server.log"
        raw = p.read_bytes()
        t = raw.decode(errors="replace")
        ceiling = int(re.search(r"'max_cudagraph_capture_size': (\d+)", t).group(1))
        if ceiling in seen:
            continue
        seen.add(ceiling)
        assert "max_num_seqs" not in t.split("non-default args:")[1].split("\n")[0]
        assert "max_num_batched_tokens=8192" in t and "num_spec_tokens=5" in t
        assert BUILD.split()[1] in t
        m = [float(x) for x in took.findall(t)]
        actual, est = map(float, pool.search(t).groups())
        assert len(m) == 2 and m[1] == actual, (p, m, actual)
        rows.append({"artifact": v["bound"]["dir"].split("/")[-1],
                     "server_log_sha256": hashlib.sha256(raw).hexdigest(),
                     "max_cudagraph_capture_size": ceiling, "max_num_seqs": 1024,
                     "max_num_batched_tokens": 8192, "num_spec_tokens": 5,
                     "estimated_gib": est, "profiling_took_gib": m[0], "actual_gib": actual})
    ANCHOR.write_text(json.dumps({
        "source": SOURCE, "vllm_build": BUILD, "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "hardware": "h200", "tensor_parallel_size": 8, "rank": 0,
        "units": "GiB per rank, as logged (two decimals)", "startups": rows}, indent=1) + "\n")
    print(f"wrote {ANCHOR.relative_to(REPO)} ({len(rows)} start-ups)")
    return 0


def fit(anchor: dict) -> dict:
    rows = []
    for s in anchor["startups"]:
        dec = decode_graphs(s["max_cudagraph_capture_size"], s["max_num_batched_tokens"],
                            s["max_num_seqs"], 1 + s["num_spec_tokens"])
        rows.append((s["max_cudagraph_capture_size"], dec[-1], s["estimated_gib"],
                     s["profiling_took_gib"]))
    b = (sum(d * (e - m) for _, d, e, m in rows) / sum(d * d for _, d, _, _ in rows))
    xs, ys = [c for c, *_ in rows], [m for *_, m in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    f = my - a * mx
    resid = [f + a * c + b * d - e for c, d, e, _ in rows]
    return {"floor": round(f * GIB), "mixed": round(a * GIB), "decode": round(b * GIB),
            "rows": rows, "resid": resid}


def entry(name: str, value: int, rationale: str) -> list[str]:
    return [f"  - {name}:",
            f"      value: {value}",
            "      units: bytes_per_rank",
            "      method: measured",
            "      fitted: true",
            f"      scope: {SCOPE}",
            "      sources:",
            f'        - {{kind: discussion, cite: "{SOURCE}; derived figures vendored in '
            f'validator/testdata/anchors/{ANCHOR.name}", role: primary}}',
            '        - {kind: discussion, cite: "vLLM v0.31.0 vllm/v1/worker/gpu/'
            'cudagraph_utils.py:854-967 (profile_cudagraph_memory) and vllm/v1/worker/'
            'gpu_worker.py:628-676, the estimate and its subtraction", role: supporting}',
            "      rationale: >",
            f"        {rationale}"]


def entries(r: dict) -> list[list[str]]:
    table = "; ".join(f"C {c}: decode {d}, E {e:.2f}, M {m:.2f}, fit {e + x:+.2f}"
                      for (c, d, e, m), x in zip(r["rows"], r["resid"]))
    common = (
        f"THE FORM the kernel reads all or none: floor + per_mixed_token x C_mixed + "
        f"per_decode_token x C_decode, C_mixed the capture ceiling and C_decode the largest "
        f"uniform-decode FULL graph (each capture size rounded up to 1 + K, kept within "
        f"max_num_seqs requests and the ceiling). It is what vLLM v0.31.0 takes from the KV "
        f"budget for graphs: the profile_cudagraph_memory estimate, subtracted by "
        f"determine_available_memory. FIT on five InferenceX H200 TP8 start-ups of this model "
        f"({BUILD}; K 5, max_num_seqs 1024, mnbt 8192; per rank, GiB, E the logged estimate, "
        f"M the profiling pass's capture): {table}. THE SPLIT is the source's, not fitted: the "
        f"estimate is M - samples + first + (n_full - 1) x second, so E - M is the uniform-"
        f"decode FULL family alone and gives the decode slope (through the origin); M gives the "
        f"floor and the mixed slope. The fit arm holds max_num_seqs at 1024, so it cannot "
        f"separate the slopes, and M also holds the drafter and two sampled FULL graphs, which "
        f"follow the decode ceiling: the max_num_seqs sensitivity is UNDERSTATED. VERSION: the "
        f"logs are a nightly; the estimate's code path (profile_cudagraph_memory, "
        f"_extrapolate_full_graph_memory, CudaGraphManager._init_candidates, capture_model, the "
        f"subtraction) is unchanged between cd10ed6f9 and v0.31.0 except prepare_inputs_to_"
        f"capture's max_query_len, which for a uniform-decode graph is 1 + K either way; the "
        f"model's own forward did change (vllm/models/deepseek_v41), which this cannot rule out "
        f"moving the magnitude. SCOPE h100 and h200: one per-rank figure, since both run the "
        f"same SM90 kernels (FLASHMLA_SPARSE_DSV41, Marlin MXFP4 experts, Marlin MXFP8 linears "
        f"in the logs); TP-invariant per rank (TP4 and TP8 log 8.59 / 8.61 GiB at C 512). "
        f"HELD OUT, not used: InferenceX H100 TP8 and H200 TP4, and the H200 calibration runs' "
        f"boot-time pools. UNITS: bytes_per_rank, with the per-token meaning in the name, "
        f"because the closed units vocabulary has no bytes_per_token. Reproduce with "
        f"scripts/fit_deepseek_v41_capture.py.")
    return [
        entry(NAMES[0], r["floor"],
              f"The floor of DeepSeek-V4.1-Flash's FULL_AND_PIECEWISE capture estimate, "
              f"{r['floor'] / GIB:.3f} GiB per rank: the intercept of M against the capture "
              f"ceiling. {common}"),
        entry(NAMES[1], r["mixed"],
              f"Capture-estimate bytes per rank per token of the capture ceiling (the "
              f"PIECEWISE, drafter and sampled FULL graphs), {r['mixed'] / 2**20:.2f} MiB. "
              f"{common}"),
        entry(NAMES[2], r["decode"],
              f"Capture-estimate bytes per rank per token of the uniform-decode FULL ceiling "
              f"(the extrapolated FULL family), {r['decode'] / 2**20:.2f} MiB. {common}"),
    ]


def render(text: str, new: list[list[str]]) -> str:
    """After the flat FULL_AND_PIECEWISE figure the entries supersede for this model."""
    blocks = cb.split(text)
    for e in new:
        cb.upsert(blocks, e, after=lambda b: cb.name_of(b) in
                  {f"cudagraph_capture_bytes_{MODE}", *NAMES})
    return cb.join(blocks)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--extract-anchor", metavar="IX_DSV41_DIR")
    args = ap.parse_args(argv[1:])
    if args.extract_anchor:
        return extract_anchor(Path(args.extract_anchor))
    r = fit(json.loads(ANCHOR.read_text()))
    print(f"floor {r['floor']} B, mixed {r['mixed']} B/token, decode {r['decode']} B/token; "
          f"residuals {', '.join(f'{x:+.3f}' for x in r['resid'])} GiB")
    t0 = MEMORY.read_text(encoding="utf-8")
    t1 = render(t0, entries(r))
    if args.check:
        if t0 != t1:
            print(f"would change: {MEMORY.name}", file=sys.stderr)
            return 1
        print("the memory set already matches")
        return 0
    MEMORY.write_text(t1, encoding="utf-8")
    print("the memory set written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
