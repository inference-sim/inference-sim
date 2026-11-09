#!/usr/bin/env python3
"""The price-change report (release plan section 3.3, design X13): what moving the cost-model
fixtures to vLLM 0.32.0, stating only what each run stated (X19), did to the prices.

    scripts/kernel/price_change_report.py BEFORE.jsonl AFTER.jsonl OLD_FIXTURES_DIR [NEW_FIXTURES_DIR]

BEFORE and AFTER are `go run ./scripts/kernel/priceprobe` output (one JSON line per fixture) for
the old fixtures priced by the build that ran them (the v0.15.0 release of inference-sim) and the new fixtures
priced by this build, on the same catalog and registry. OLD_FIXTURES_DIR is the old
testdata/kernel (to read which settings each old fixture wrote); NEW_FIXTURES_DIR defaults to the
repository's own. Prints markdown.

Each generated fixture's move is attributed to the settings the old generator wrote and the new
one does not (the old files wrote cache_dtype, max_num_seqs, max_num_batched_tokens,
cudagraph_mode, block_size and gpu_memory_utilization where the run stated none). A move is
attributed to a cause by a probe, not by guesswork: see `attribute` below, which prices a few
fixtures with each old setting stated back, one at a time.

    scripts/kernel/price_change_report.py --attribute PROBE_DIR_PREFIX
        prints the one-at-a-time probe's table from PROBE_DIR_PREFIX.<variant>.jsonl files; the
        files are priceprobe output for copies of new fixtures with one old setting added
        (the variants are listed in ATTRIBUTION_VARIANTS).
"""
import json
import os
import re
import statistics as st
import sys

ENGINE_KEYS = ("cache_dtype", "block_size", "max_num_seqs", "max_num_batched_tokens",
               "cudagraph_mode", "gpu_memory_utilization")

# What each unstated setting does to a price, from the engine's own resolution
# (blis-vllm Model.EngineSources) and the kernel:
CAUSES = {
    "cache_dtype": "the old files wrote fp8; the new ones leave it to the engine, which resolves a wider "
                   "dtype (`auto`): KV bytes per sequence x2, and a larger KV read per decode step at "
                   "batch 256 (see the attribution table)",
    "max_num_seqs": "the old 256 became vLLM's resolved cap; moves fixed bytes only, by the activation "
                    "workspace the cap sizes (at most a few percent)",
    "max_num_batched_tokens": "the old 8192 became vLLM's resolved budget; moves fixed bytes only (the "
                              "activation workspace), by up to about 3%",
    "cudagraph_mode": "the old PIECEWISE became the release's resolved mode (decode under a FULL "
                      "graph): decode at batch 1 is faster by 3 to 10%, batch 256 by about 1%",
    "block_size": "the old 16 became vLLM's resolved size, or the run's stated one; moves no price",
    "gpu_memory_utilization": "the old 0.92 is also vLLM's default; moves no price",
}


# The one-at-a-time probe: each variant states one setting the old generator wrote, on top of the
# new fixture. A price that moves under a variant is moved by that setting.
ATTRIBUTION_VARIANTS = ("cache_dtype_fp8", "max_num_seqs_256", "max_num_batched_tokens_8192",
                        "cudagraph_mode_PIECEWISE", "block_size_16")


def attribute(prefix):
    base = load(prefix + ".new.jsonl")
    print("### Attribution: one old setting stated back at a time\n")
    print("Each row prices the same new fixtures with one setting the old generator wrote added; "
          "a figure moves only if that setting moves it. Percentages are against the new fixture.\n")
    print("| setting stated | fixture | decode_1 | decode_256 | KV bytes | fixed bytes |\n|---|---|---|---|---|---|")
    for v in ATTRIBUTION_VARIANTS:
        for k, j in sorted(load(f"{prefix}.{v}.jsonl").items()):
            b = base[k]
            if "error" in j or "error" in b:
                continue
            row = [pct(j["step_ns"][m] / b["step_ns"][m] - 1) for m in ("decode_1", "decode_256")]
            row.append(f"x{j['kv_bytes_1024'] / b['kv_bytes_1024']:.2f}")
            row.append(pct(j["fixed_bytes"] / b["fixed_bytes"] - 1))
            print(f"| `{v}` | {k} | " + " | ".join(row) + " |")


def load(path):
    out = {}
    for line in open(path):
        j = json.loads(line)
        out[j["fixture"]] = j
    return out


def engine_block(path):
    """The keys of the first pool's engine block in a fixture, as {key: value-string}."""
    text = open(path).read()
    m = re.search(r"^    engine:\n((?:      .*\n)+)", text, re.M)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.strip().partition(":")
            out[k] = v.strip()
    return out


def pct(x):
    return f"{x * 100:+.1f}%"


def main():
    before, after, old_dir = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    new_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "kernel")
    keys = sorted(k for k in before if k in after and "error" not in before[k] and "error" not in after[k])
    errs = [k for k in set(before) | set(after) if k not in keys]
    # A fixture is hand-written when the generator did not write it: its engine block is unchanged.
    gen = []
    hw = []
    for k in keys:
        a, b = engine_block(os.path.join(old_dir, k)), engine_block(os.path.join(new_dir, k))
        (hw if a == b else gen).append(k)
    print("## Price-change report\n")
    print("Every cost-model fixture (testdata/kernel) priced in two builds on the same catalog and registry: "
          "before, the v0.15.0 release of inference-sim on the fixtures as v0.5.0 wrote them; after, this branch on the "
          "regenerated fixtures (`engine: {kind: vllm, version: \"0.32.0\"}`, only what each run stated, X19). "
          "Steps: decode batches of 1, 32 and 256 sequences at context 4096, one prefill of 2048 tokens; "
          "KV bytes per sequence at 1024 tokens; fixed bytes.\n")
    print(f"{len(keys)} fixtures priced in both builds: {len(gen)} regenerated, {len(hw)} hand-written (engine block unchanged). "
          f"Not priced in both: {len(errs)}.\n")
    print("### Regenerated fixtures\n")
    print("| price | fixtures | median | min | max | unchanged |\n|---|---|---|---|---|---|")
    for m in ("decode_1", "decode_32", "decode_256", "prefill_2048"):
        r = [after[k]["step_ns"][m] / before[k]["step_ns"][m] - 1 for k in gen]
        print(f"| {m} step time | {len(r)} | {pct(st.median(r))} | {pct(min(r))} | {pct(max(r))} | {sum(abs(x) < 1e-12 for x in r)} |")
    for m, lab in (("kv_bytes_1024", "KV bytes per sequence"), ("fixed_bytes", "fixed bytes")):
        r = [after[k][m] / before[k][m] for k in gen]
        print(f"| {lab} | {len(r)} | x{st.median(r):.3f} | x{min(r):.3f} | x{max(r):.3f} | {sum(abs(x - 1) < 1e-12 for x in r)} |")
    same = sum(all(after[k]["step_ns"][m] == before[k]["step_ns"][m] for m in before[k]["step_ns"]) and
               after[k]["kv_bytes_1024"] == before[k]["kv_bytes_1024"] and after[k]["fixed_bytes"] == before[k]["fixed_bytes"]
               for k in gen)
    print(f"\nRegenerated fixtures with every price unchanged: {same}.\n")
    # Which old settings differ from the new
    print("### What changed in the fixtures\n")
    print("| setting the old file wrote and the new one does not | fixtures |\n|---|---|")
    gone = {key: 0 for key in ENGINE_KEYS}
    for k in gen:
        a, b = engine_block(os.path.join(old_dir, k)), engine_block(os.path.join(new_dir, k))
        for key in ENGINE_KEYS:
            if key in a and (key not in b or a[key] != b[key]):
                gone[key] += 1
    for key in ENGINE_KEYS:
        print(f"| `{key}` | {gone[key]} |")
    print("\n| setting | effect on prices |\n|---|---|")
    for key in ENGINE_KEYS:
        print(f"| `{key}` | {CAUSES[key]} |")
    # KV x2: all and only fixtures that wrote cache_dtype fp8
    kv2 = [k for k in gen if abs(after[k]["kv_bytes_1024"] / before[k]["kv_bytes_1024"] - 2) < 1e-9]
    fp8 = [k for k in gen if engine_block(os.path.join(old_dir, k)).get("cache_dtype") == "fp8"]
    print(f"\nKV bytes per sequence double (x2.000) in {len(kv2)} fixtures; {len(fp8)} old files wrote `cache_dtype: fp8`; "
          f"{len(set(kv2) & set(fp8))} are in both, and {len(set(kv2) - set(fp8))} doubled without having written it.\n")
    print("### Largest decode_256 moves\n")
    print("| fixture | decode_256 | KV bytes | cache_dtype in the old file |\n|---|---|---|---|")
    for k in sorted(gen, key=lambda k: -(after[k]["step_ns"]["decode_256"] / before[k]["step_ns"]["decode_256"]))[:6]:
        print(f"| {k} | {pct(after[k]['step_ns']['decode_256'] / before[k]['step_ns']['decode_256'] - 1)} | "
              f"x{after[k]['kv_bytes_1024'] / before[k]['kv_bytes_1024']:.2f} | {engine_block(os.path.join(old_dir, k)).get('cache_dtype', '-')} |")
    print("\n### Hand-written fixtures\n")
    moved = [k for k in hw if before[k]["step_ns"] != after[k]["step_ns"] or before[k]["kv_bytes_1024"] != after[k]["kv_bytes_1024"]
             or before[k]["fixed_bytes"] != after[k]["fixed_bytes"]]
    print(f"{len(hw)} hand-written fixtures keep their stated settings and change only their engine statement; "
          f"{len(hw) - len(moved)} price identically.\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--attribute":
        attribute(sys.argv[2])
        sys.exit(0)
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main()
