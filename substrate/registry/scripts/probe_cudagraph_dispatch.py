"""Test whether a vLLM decode step can miss its CUDA graph.

Two candidate mechanisms motivated this probe. Both are refuted here, so no
coefficient follows from either.

  Capture-size miss. CudagraphDispatcher.dispatch (vllm/v1/cudagraph_dispatcher.py:235)
  returns CUDAGraphMode.NONE when num_tokens exceeds max_cudagraph_capture_size,
  which launches every layer eagerly. That would be a step function in host cost.

  Batch padding. _create_padded_batch_descriptor (vllm/v1/cudagraph_dispatcher.py:132)
  rounds the token count up to the next captured size via _bs_to_padded_graph_size,
  so a step just above a captured size pays for the one above it. That would be a
  sawtooth against batch width.

Neither can fire on a non-speculative vLLM decode step, for a reason that is
structural rather than a property of this corpus:

  the ceiling is   min(2 * max_num_seqs, platform_default, max_num_batched_tokens)
                   -- vllm/config/vllm.py:1971-2067
  the batch is     min(concurrency, max_num_seqs)
                   -- scheduler.py:784 breaks on num_running >= max_num_running_reqs,
                      and scheduler.py:1210 asserts len(running) <= max_num_running_reqs

so batch <= max_num_seqs, which is at most half of 2 * max_num_seqs. A miss needs
the platform default to bind below max_num_seqs, i.e. max_num_seqs > 256 on Hopper
or > 512 on Blackwell.

Padding cannot fire either: the grid is {1,2,4} union range(8,256,8) union
range(256,max,16), and the corpus sweeps concurrency in powers of two. Every power
of two at or above 8 is a multiple of 8 or of 16, so it is itself a captured size.

Reads InferenceX settings only to characterize a model error, never to fit. The
decision this probe reports is derived from vLLM source, not from InferenceX
measurements, so it would hold for any corpus with the same engine settings.
"""

import argparse
import collections
import json
from pathlib import Path

BLACKWELL = {"b200", "b300", "gb200", "gb200-nvl72"}

# vllm/config/scheduler.py:44
DEFAULT_MAX_NUM_SEQS = 128


def capture_sizes(max_num_seqs: int, max_num_batched_tokens: int, blackwell: bool) -> list[int]:
    """vLLM's default capture grid for uniform_decode_query_len == 1.

    Mirrors vllm/config/vllm.py:1971-2120. The speculative branch, which appends
    uniform_decode_sizes, is deliberately not reproduced: this probe reports
    coverage for the non-speculative arms and counts the rest as untested.
    """
    default_max = 1024 if blackwell else 512
    ceiling = min(max_num_seqs * 2, default_max)
    ceiling = min(max_num_batched_tokens, ceiling)
    sizes = [i for i in (1, 2, 4) if i <= ceiling]
    if ceiling >= 8:
        sizes += list(range(8, min(ceiling + 1, 256), 8))
    if ceiling >= 256:
        sizes += list(range(256, ceiling + 1, 16))
    if max_num_batched_tokens <= ceiling and max_num_batched_tokens not in sizes:
        sizes.append(max_num_batched_tokens)
    return sorted(set(sizes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    args = ap.parse_args()

    settings = json.loads(args.settings.read_text())
    corpus = json.loads(args.corpus.read_text())
    spec_method = {s["scenario"]: s.get("spec_method", "none") for s in corpus["sweeps"]}

    total = stated_both = exceeded = padded = 0
    binding: collections.Counter[str] = collections.Counter()
    spec_cells = 0

    for scenario in settings["settings"]:
        blackwell = scenario["gpu"] in BLACKWELL
        default_max = 1024 if blackwell else 512
        is_spec = spec_method.get(scenario["scenario"], "none") != "none"
        for concurrency, record in scenario["by_concurrency"].items():
            passed = record.get("passed") or {}
            total += 1
            if is_spec:
                spec_cells += 1
                continue
            nbt = passed.get("max_num_batched_tokens")
            if nbt is None:
                continue
            ns = passed.get("max_num_seqs")
            if ns is not None:
                stated_both += 1
            ns_eff = ns if ns is not None else DEFAULT_MAX_NUM_SEQS

            batch = min(int(concurrency), ns_eff)
            sizes = capture_sizes(ns_eff, nbt, blackwell)
            if not sizes:
                continue
            if batch > sizes[-1]:
                exceeded += 1
            else:
                nxt = next(z for z in sizes if z >= batch)
                if nxt != batch:
                    padded += 1

            terms = {
                "2*max_num_seqs": 2 * ns_eff,
                "platform_default": default_max,
                "max_num_batched_tokens": nbt,
            }
            binding[min(terms, key=terms.get)] += 1

    evaluated = sum(binding.values())
    print(f"scenario x concurrency cells   : {total}")
    print(f"  speculative (not tested here): {spec_cells}")
    print(f"  evaluated                    : {evaluated}")
    print(f"    of which max_num_seqs stated (not defaulted): {stated_both}")
    print(f"  decode batch exceeds ceiling : {exceeded}")
    print(f"  decode batch needs padding   : {padded}")
    print("  binding term:", dict(binding))
    verdict = "REFUTED" if exceeded == 0 and padded == 0 else "NOT REFUTED"
    print(f"\ncapture-miss and padding mechanisms: {verdict} on the evaluated cells")


if __name__ == "__main__":
    main()
