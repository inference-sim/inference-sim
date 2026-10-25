# Original to blis-vllm.
"""Mutation sweep for blis-vllm's tier 5 (core/metrics): apply one mutant at a time to a
scratch copy of the module, run the metrics unit, diff and fuzz tests, and report whether each
is caught.

Each mutant breaks one rule vLLM's PrometheusStatLogger follows (vllm/v1/metrics/loggers.py,
stats.py@5df63a72), so the supported "values of the scheduler-decided exact families" row
(docs/feature-matrix.md) names mutants that exist and run in CI.

Usage: python3 oracle/mutate_metrics.py <blis-vllm checkout> <fuzz dir> <scratch dir> [X1,...]
(<fuzz dir> holds gen_fuzz.py scenarios with their run_metrics.py .metrics.json traces)
Exits non-zero if a mutant survives or breaks the build without a divergence.
"""
import os
import shutil
import subprocess
import sys

# (id, the rule it breaks, file, exact old text, new text)
MUTANTS = [
    ("X1", "a resumed (preempted) request's prefix lookup is not counted again", "core/metrics/metrics.go",
     "\t\tif a.Preempted {\n\t\t\tcontinue\n\t\t}", "\t\tif false {\n\t\t\tcontinue\n\t\t}"),
    ("X2", "a preemption counts when the request next emits", "core/metrics/metrics.go",
     "\tfor _, id := range s.Preempted {\n\t\tm.preemptions[id]++\n\t}",
     "\tfor range s.Preempted {\n\t\tm.counters[\"vllm:num_preemptions_total\"]++\n\t}"),
    ("X3", "prompt stats arrive with the first token", "core/metrics/metrics.go",
     "\t\tm.pending[a.ID] = prefill{prompt: a.PromptTokens, localHit: a.LocalHit}\n",
     "\t\tm.pending[a.ID] = prefill{prompt: a.PromptTokens, localHit: a.LocalHit}\n"
     "\t\tm.counters[\"vllm:prompt_tokens_total\"] += float64(a.PromptTokens)\n"),
    ("X4", "cached prompt tokens are the local hit", "core/metrics/metrics.go",
     "\t\tm.counters[\"vllm:prompt_tokens_cached_total\"] += float64(p.localHit)\n",
     "\t\tm.counters[\"vllm:prompt_tokens_cached_total\"] += 0\n"),
    ("X5", "iteration tokens count the prefill tokens computed", "core/metrics/metrics.go",
     "m.observe(\"vllm:iteration_tokens_total\", float64(computed+generated))",
     "m.observe(\"vllm:iteration_tokens_total\", float64(generated))"),
    ("X6", "a histogram bucket is cumulative (v <= bound)", "core/metrics/metrics.go",
     "\t\tif v <= b {\n\t\t\th.counts[i]++", "\t\tif v < b {\n\t\t\th.counts[i]++"),
    ("X7", "the prefill KV computed is the prompt less its cached tokens", "core/metrics/metrics.go",
     "float64(f.PromptTokens-max(f.CachedTokens, 0))", "float64(f.PromptTokens)"),
]

# Not a mutant here: the deferred split of vllm:num_requests_waiting_by_reason. No committed
# scenario or fuzz seed defers a waiting request (deferred_waiting is set only by a structured
# output, a remote-KV wait or a LoRA or encoder budget skip, scheduler.py:230), so vLLM's traces
# hold 0 there and the diff test feeds Deferred=0; dropping the subtraction is equivalent on every
# trace. docs/feature-matrix.md states the gauge as checked at 0 only.

TESTS = "TestCountersFollowTheSteps|TestAResumedRequestIsNotCountedAgain|TestNoLookupNoPrefixCacheQueries|" \
        "TestPromptStatsArriveWithTheFirstToken|TestMetricsAgainstVLLM|TestMetricsFuzzCorpusAgainstVLLM"


def main() -> int:
    src, scratch = sys.argv[1], sys.argv[3]
    # go test runs each package in its own directory, so the corpus path must be absolute.
    fuzz = os.path.abspath(sys.argv[2])
    if not any(f.endswith(".metrics.json") for f in os.listdir(fuzz)):
        print(f"no metrics traces in {fuzz}")
        return 1
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    env = dict(os.environ, BLIS_VLLM_METRICS_FUZZ_DIR=fuzz)
    bad = 0
    for mid, row, path, old, new in MUTANTS:
        if only and mid not in only:
            continue
        if os.path.exists(scratch):
            shutil.rmtree(scratch)
        shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(".git", "vllm-source", "oracle-model", "out"))
        target = os.path.join(scratch, path)
        text = open(target).read()
        if text.count(old) != 1:
            print(f"BROKEN   {mid:3} {row}: pattern found {text.count(old)} times in {path}")
            bad += 1
            continue
        open(target, "w").write(text.replace(old, new))
        try:
            r = subprocess.run(["go", "test", "-count=1", "-run", TESTS, "./core/metrics/"],
                               cwd=scratch, env=env, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            # A timeout is not a divergence: a mutant that hangs the tests proves nothing.
            print(f"BROKEN   {mid:3} {row}: the tests timed out")
            bad += 1
            continue
        out = r.stdout + r.stderr
        if r.returncode == 0:
            print(f"SURVIVED {mid:3} {row}")
            bad += 1
        elif "build failed" in out or "[setup failed]" in out:
            print(f"BROKEN   {mid:3} {row}: the mutant does not build")
            bad += 1
        elif "--- FAIL" in out:
            print(f"CAUGHT   {mid:3} {row}")
        else:
            tail = out.strip().splitlines()[-3:]
            print(f"BROKEN   {mid:3} {row}: failed without a test failure: {' | '.join(tail)[:200]}")
            bad += 1
    shutil.rmtree(scratch, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
