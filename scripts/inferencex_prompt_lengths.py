#!/usr/bin/env python3
"""Reproduce InferenceX's random-prompt construction offline, at pinned tokenizer revisions, and
record per scored point how far the prompt length the benchmark client SENT lies from its numpy
TARGET -- the evidence testdata/inferencex-workloads/prompt-lengths.json carries.

The client (SemiAnalysisAI/InferenceX utils/bench_serving/benchmark_serving.py) builds request
i's prompt from token ids (offsets[i] + i + j) % vocab_size, decodes it to text, and re-encodes
it up to ten times -- padding with random ids when short, truncating when long -- to land on the
target; it then records len(encode(prompt, add_special_tokens=False)). Two client versions are
live over the measured runs' dates:

  before 1780296b (2026-04-21)  serial loop; padding from the GLOBAL np.random stream after
                                `offsets` (1780296b^ lines 118-150)
  from 1780296b                 local_rng = RandomState(np.random.get_state()[1][:4]) after
                                `offsets`; with a tokenizer id and num_workers = min(cpu_count,
                                8) > 1, prompts are built in ceil(n/W) chunks, chunk w padding
                                from RandomState(local_rng.randint(0, 2**31)) in chunk order
                                (08d7a9cd lines 162-196, 238-290)

The server then tokenizes the prompt TEXT itself; the tokens add_special_tokens=True adds (the
completions default) are recorded too.

Tokenizers are pinned: for every point, the Hugging Face revision of its tokenizer repo current
at the point's run date, and the repo's revision today. Every tokenizer file of every revision
used is recorded by sha256. Revisions whose tokenizer files are byte-identical produce identical
prompts, so the construction runs once per distinct file set.

Requirements: numpy, transformers, tokenizers, huggingface_hub (and tiktoken for Kimi). Tokenizer
files are downloaded into $HF_HOME, which must be set and must lie OUTSIDE this repository. Run
with `python -I`. Kimi-K2.5's tokenizer is custom code (tokenization_kimi.py) and needs
trust_remote_code; read it before running. Points with no engine log (Llama-3.1-70B, whose
tokenizer is also gated) have no run date or known client version and are not recorded.

Usage:
    export HF_HOME=~/.cache/blis-check/tokenizers/hf
    python -I scripts/inferencex_prompt_lengths.py \\
        --corpus "$BLIS_MEASUREMENTS"/aisimulate_e2e.json \\
        --engine-settings "$BLIS_MEASUREMENTS"/inferencex_engine_settings.json \\
        --out testdata/inferencex-workloads/prompt-lengths.json
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_PARALLEL_FROM = "2026-04-21"  # 1780296b, "Multiprocess Parallel Random Data Generation"
RANGE_RATIO = 0.8
LABELS = {"1k1k": (1024, 1024), "1k8k": (1024, 8192), "8k1k": (8192, 1024)}
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tiktoken.model",
                   "tokenization_kimi.py", "tool_declaration_ts.py", "special_tokens_map.json", "added_tokens.json"]
LOADER_FILES = ["config.json"]  # read by AutoTokenizer to resolve the class; hashed, not compared
TRUST_REMOTE_CODE = {"nvidia/Kimi-K2.5-NVFP4"}


def repo_for(served_model):
    """The public tokenizer repo of the checkpoint a run served (engine-args `model`). A local
    path is taken to be a copy of the repo of the same name."""
    name = served_model.rstrip("/").split("/")[-1]
    return {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
        "MiniMax-M2.5-NVFP4": "nvidia/MiniMax-M2.5-NVFP4",
        "MiniMax-M3-NVFP4": "nvidia/MiniMax-M3-NVFP4",
        "Kimi-K2.5-NVFP4": "nvidia/Kimi-K2.5-NVFP4",
        "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    }.get(name)


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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--engine-settings", required=True, help="supplies each point's run date and served model")
    ap.add_argument("--workers", type=int, default=8, help="the client's min(cpu_count, 8); not recorded")
    ap.add_argument("--out", required=True, help="the evidence file to write")
    args = ap.parse_args()

    hf = os.environ.get("HF_HOME")
    if not hf or os.path.abspath(hf).startswith(REPO + os.sep):
        sys.exit("set HF_HOME to a directory outside this repository; tokenizer files are downloaded there")
    import huggingface_hub
    import tokenizers
    import transformers
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    logging.disable(logging.WARNING)  # Kimi's tokenizer warns on every encode() with kwargs
    api = HfApi()

    corpus = json.load(open(args.corpus))
    settings = {(s["scenario"], s["label"]): s for s in json.load(open(args.engine_settings))["settings"]}
    points = []
    for sw in corpus["sweeps"]:
        rec = settings.get((sw["scenario"], sw["label"]))
        if sw["framework"] != "vllm" or rec is None:
            continue
        isl, osl = LABELS[sw["label"]]
        date = rec["source_run_date"]
        for p in sw["points"]:
            ps = rec["by_concurrency"].get(str(p["concurrency"]))
            if ps is None:
                continue
            repo = repo_for(ps["passed"]["model"])
            if repo is None:
                sys.exit(f'no tokenizer repo known for served model {ps["passed"]["model"]}')
            points.append({"scenario": sw["scenario"], "label": sw["label"], "concurrency": p["concurrency"],
                           "served_model": ps["passed"]["model"], "repo": repo, "run_date": date, "isl": isl,
                           "osl": osl, "client_path": "serial-global" if date < CLIENT_PARALLEL_FROM else "parallel"})

    # Pin the revisions: the one current at each run date, and today's.
    commits = {r: api.list_repo_commits(r) for r in sorted({p["repo"] for p in points})}
    revisions = {}  # (repo, sha) -> record

    def revision(repo, sha, created, role):
        key = (repo, sha)
        if key not in revisions:
            d = snapshot_download(repo, revision=sha, allow_patterns=TOKENIZER_FILES + LOADER_FILES)
            files = {f: sha256(os.path.join(d, f)) for f in TOKENIZER_FILES + LOADER_FILES
                     if os.path.exists(os.path.join(d, f))}
            revisions[key] = {"repo": repo, "revision": sha, "committed": created, "roles": [], "dir": d,
                              "files": files}
        if role not in revisions[key]["roles"]:
            revisions[key]["roles"].append(role)
        return key

    for p in points:
        cs = commits[p["repo"]]
        today = cs[0]
        end = datetime.datetime.fromisoformat(p["run_date"] + "T23:59:59+00:00")
        at = next((c for c in cs if c.created_at <= end), None)
        p["revisions"] = []
        if at is None:
            p["revisions"].append(None)  # the repo did not exist at the run date
        else:
            p["revisions"].append(revision(p["repo"], at.commit_id, at.created_at.isoformat(), "run-date " + p["run_date"]))
        p["revisions"].append(revision(p["repo"], today.commit_id, today.created_at.isoformat(), "today"))

    # One construction per distinct tokenizer file set and point shape.
    loaded, results = {}, {}
    for p in points:
        out = []
        for key in p["revisions"]:
            if key is None:
                out.append(None)
                continue
            rev = revisions[key]
            content = tuple(sorted((f, h) for f, h in rev["files"].items() if f in TOKENIZER_FILES))
            if content not in loaded:
                loaded[content] = AutoTokenizer.from_pretrained(rev["dir"], trust_remote_code=rev["repo"] in TRUST_REMOTE_CODE)
            rk = (content, p["isl"], p["osl"], p["concurrency"], p["client_path"])
            if rk not in results:
                rows = prompts(loaded[content], p["isl"], p["osl"], p["concurrency"], p["client_path"], args.workers)
                mm = [s - t for t, s, _ in rows]
                results[rk] = {"requests": len(rows), "max_abs_mismatch": max(map(abs, mm)),
                               "mean_mismatch": sum(mm) / len(mm), "requests_mismatched": sum(1 for x in mm if x),
                               "server_added_tokens": sorted({v - s for _, s, v in rows})}
                print(f'{p["repo"]:30s} {key[1][:8]} {p["isl"]}:{p["osl"]} c={p["concurrency"]:<4d} {p["client_path"]:13s} '
                      f'max|mismatch|={results[rk]["max_abs_mismatch"]} server+{results[rk]["server_added_tokens"]}', flush=True)
            out.append({"revision": key[1], **results[rk]})
        p["evidence"] = out

    doc = {
        "what": "per scored point, the InferenceX client's sent prompt length against its numpy target, from "
                "reproducing the client's prompt construction with the point's tokenizer at pinned revisions",
        "generator": "scripts/inferencex_prompt_lengths.py",
        "client": {"source": "SemiAnalysisAI/InferenceX utils/bench_serving/benchmark_serving.py",
                   "serial-global": "before 1780296b (2026-04-21)", "parallel": "from 1780296b",
                   "workers": args.workers},
        "environment": {"numpy": np.__version__, "transformers": transformers.__version__,
                        "tokenizers": tokenizers.__version__, "huggingface_hub": huggingface_hub.__version__},
        "revisions": [{k: v for k, v in r.items() if k != "dir"} for _, r in sorted(revisions.items())],
        "points": [{**{k: v for k, v in p.items() if k not in ("revisions", "evidence")},
                    "evidence": p["evidence"],
                    "max_abs_mismatch": max((e["max_abs_mismatch"] for e in p["evidence"] if e), default=None),
                    "server_added_tokens": sorted({t for e in p["evidence"] if e for t in e["server_added_tokens"]})}
                   for p in points],
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}: {len(points)} points, {len(revisions)} revisions, {len(results)} constructions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
