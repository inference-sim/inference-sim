#!/usr/bin/env python3
"""Reproduce InferenceX's random-prompt construction offline and measure, per scored point, how
far the prompt length the benchmark client SENT differs from its numpy TARGET.

The client (SemiAnalysisAI/InferenceX utils/bench_serving/benchmark_serving.py) builds request
i's prompt from token ids (offsets[i] + i + j) % vocab_size, decodes it to text, and re-encodes
it up to ten times -- padding with random ids when short, truncating when long -- to land on the
target; it then records len(encode(prompt, add_special_tokens=False)) and the difference
("mismatch"). Two client versions are live over the measured runs' dates:

  before 1780296b (2026-04-21)  serial loop; padding from the GLOBAL np.random stream after
                                `offsets` (1780296b^ lines 118-150)
  from 1780296b                 local_rng = RandomState(np.random.get_state()[1][:4]) after
                                `offsets`; with a tokenizer id and num_workers = min(cpu_count,
                                8) > 1, prompts are built in ceil(n/W) chunks, chunk w padding
                                from RandomState(local_rng.randint(0, 2**31)) in chunk order
                                (08d7a9cd lines 162-196, 238-290)

The server then tokenizes the prompt TEXT itself; with add_special_tokens=True (the completions
default) it may add a BOS. Both differences are reported per point.

Requirements: numpy, transformers, tokenizers (and tiktoken for Kimi). Tokenizer files are
downloaded from Hugging Face into $HF_HOME, which must be set and must lie OUTSIDE this
repository. Run with `python -I`. Kimi-K2.5's tokenizer is custom code (tokenization_kimi.py)
and needs trust_remote_code; read it before running. Gated models (Llama) are skipped.

Usage:
    export HF_HOME=~/.cache/blis-check/tokenizers/hf
    python -I scripts/inferencex_prompt_lengths.py \\
        --corpus "$BLIS_MEASUREMENTS"/aisimulate_e2e.json \\
        --engine-settings "$BLIS_MEASUREMENTS"/inferencex_engine_settings.json [--model minimax-m2.5]
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_PARALLEL_FROM = "2026-04-21"  # 1780296b, "Multiprocess Parallel Random Data Generation"
RANGE_RATIO = 0.8
LABELS = {"1k1k": (1024, 1024), "1k8k": (1024, 8192), "8k1k": (8192, 1024)}

# The checkpoint each scored model was served from (engine-args `model`), and the public HF repo
# whose tokenizer files are that checkpoint's. The NVFP4 repos' tokenizer files are byte-identical
# to their base models' (checked by sha256 when this table was written).
TOKENIZERS = {
    "gpt-oss-120b": ("openai/gpt-oss-120b", False),
    "minimax-m2.5": ("MiniMaxAI/MiniMax-M2.5", False),
    "minimax-m3": ("nvidia/MiniMax-M3-NVFP4", False),
    "kimi-k2.5": ("nvidia/Kimi-K2.5-NVFP4", True),  # tokenization_kimi.TikTokenTokenizer
    "deepseek-v4-pro": ("deepseek-ai/DeepSeek-V4-Pro", False),
    "llama-3.1-70b-instruct": ("meta-llama/Llama-3.1-70B-Instruct", False),  # gated
}


def prompts(tok, isl, osl, conc, path, workers):
    """The client's prompts for one point: per request (target, sent, server) lengths."""
    n = 10 * conc
    vocab = tok.vocab_size
    np.random.seed(0)
    prefix = np.random.randint(0, vocab, size=0).tolist()
    input_lens = np.random.randint(int(isl * RANGE_RATIO), isl + 1, size=n).tolist()
    np.random.randint(int(osl * RANGE_RATIO), osl + 1, size=n)  # output_lens: drawn, not needed
    offsets = np.random.randint(0, vocab, size=n)

    def build(i, rng):
        tgt = len(prefix) + input_lens[i]
        ids = prefix + [int((offsets[i] + i + j) % vocab) for j in range(input_lens[i])]
        prompt = tok.decode(ids)
        for _ in range(10):
            ids = tok.encode(prompt, add_special_tokens=False)
            if len(ids) < tgt:
                ids.extend(rng.randint(0, vocab, size=tgt - len(ids)).tolist())
            elif len(ids) > tgt:
                ids = ids[:tgt]
            else:
                break
            prompt = tok.decode(ids)
        return (tgt, len(tok.encode(prompt, add_special_tokens=False)),
                len(tok.encode(prompt, add_special_tokens=True)))

    if path == "serial-global":
        return [build(i, np.random) for i in range(n)]
    local_rng = np.random.RandomState(np.random.get_state()[1][:4].tolist())
    if workers <= 1:
        return [build(i, local_rng) for i in range(n)]
    rows, cs = [], (n + workers - 1) // workers
    for w in range(workers):
        start, end = w * cs, min(w * cs + cs, n)
        if start >= n:
            break
        rng = np.random.RandomState(int(local_rng.randint(0, 2**31)))
        rows.extend(build(i, rng) for i in range(start, end))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--engine-settings", required=True, help="supplies each sweep's run date")
    ap.add_argument("--model", action="append", help="restrict to these models (repeatable)")
    ap.add_argument("--workers", type=int, default=8, help="the client's min(cpu_count, 8); not recorded")
    ap.add_argument("--out", help="write the per-point report as JSON here")
    args = ap.parse_args()

    hf = os.environ.get("HF_HOME")
    if not hf or os.path.abspath(hf).startswith(REPO + os.sep):
        sys.exit("set HF_HOME to a directory outside this repository; tokenizer files are downloaded there")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Kimi's tokenizer logs a warning on every encode() that passes add_special_tokens.
    logging.disable(logging.WARNING)

    corpus = json.load(open(args.corpus))
    dates = {(s["scenario"], s["label"]): s["source_run_date"]
             for s in json.load(open(args.engine_settings))["settings"]}
    points, undated = {}, set()
    for sw in corpus["sweeps"]:
        if sw["framework"] != "vllm" or (args.model and sw["model"] not in args.model):
            continue
        date = dates.get((sw["scenario"], sw["label"]))
        if date is None:
            undated.add(sw["model"])
            continue
        isl, osl = LABELS[sw["label"]]
        path = "serial-global" if date < CLIENT_PARALLEL_FROM else "parallel"
        for p in sw["points"]:
            points.setdefault((sw["model"], isl, osl, p["concurrency"], path), []).append(
                f'{sw["scenario"]} {sw["label"]}')

    for model in sorted(undated - {m for m, *_ in points}):
        print(f"{model}: no engine-args log, so no run date and no known client version; skipped")
    toks, report = {}, []
    for (model, isl, osl, conc, path), sweeps in sorted(points.items()):
        if model not in toks:
            repo, trust = TOKENIZERS[model]
            try:
                d = snapshot_download(repo, allow_patterns=["*.json", "*.model", "*.tiktoken", "tokenization_*.py",
                                                            "tool_declaration_ts.py", "merges.txt", "vocab.*"])
                toks[model] = (AutoTokenizer.from_pretrained(d, trust_remote_code=trust), repo, os.path.basename(d))
            except Exception as e:  # gated or unavailable: reported, never guessed
                toks[model] = (None, repo, f"{type(e).__name__}")
        tok, repo, rev = toks[model]
        if tok is None:
            print(f"{model}: {repo} unavailable ({rev}); skipped")
            continue
        rows = prompts(tok, isl, osl, conc, path, args.workers)
        mm = [s - t for t, s, _ in rows]
        bos = sorted({v - s for _, s, v in rows})
        rel = [abs(s - t) / t for t, s, _ in rows]
        r = {"model": model, "tokenizer": f"{repo}@{rev}", "isl": isl, "osl": osl, "concurrency": conc,
             "client_path": path, "workers": args.workers, "requests": len(rows),
             "max_abs_mismatch": max(map(abs, mm)), "mean_mismatch": sum(mm) / len(mm),
             "mean_abs_rel": sum(rel) / len(rel), "max_abs_rel": max(rel),
             "requests_mismatched": sum(1 for x in mm if x), "server_added_tokens": bos, "sweeps": sweeps}
        report.append(r)
        print(f'{model:24s} {isl}:{osl} c={conc:<4d} {path:13s} n={len(rows):<5d} max|mismatch|={r["max_abs_mismatch"]} '
              f'mean={r["mean_mismatch"]:+.3f} mean|mismatch|/target={100 * r["mean_abs_rel"]:.4f}% '
              f'server+{bos}', flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
