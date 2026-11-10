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
    # SpecDecodingStats.observe_draft and SpecDecodingProm (vllm/v1/spec_decode/metrics.py).
    ("X8", "a draft is counted for each verifying request", "core/metrics/metrics.go",
     "m.counters[\"vllm:spec_decode_num_drafts_total\"]++", "m.counters[\"vllm:spec_decode_num_drafts_total\"] += 2"),
    ("X9", "the proposed draft tokens are the ones scheduled", "core/metrics/metrics.go",
     "m.counters[\"vllm:spec_decode_num_draft_tokens_total\"] += float64(d.Proposed)",
     "m.counters[\"vllm:spec_decode_num_draft_tokens_total\"] += float64(d.Proposed + 1)"),
    ("X10", "the accepted tokens are the drafts the target accepted", "core/metrics/metrics.go",
     "m.counters[\"vllm:spec_decode_num_accepted_tokens_total\"] += float64(d.Accepted)",
     "m.counters[\"vllm:spec_decode_num_accepted_tokens_total\"] += float64(d.Proposed)"),
    ("X11", "position i counts the requests that accepted more than i drafts", "core/metrics/metrics.go",
     "for i := 0; i < d.Accepted && i < m.specTokens; i++ {", "for i := 0; i < d.Accepted-1 && i < m.specTokens; i++ {"),
    ("X12", "one accepted-tokens counter is registered for every draft position", "core/metrics/metrics.go",
     "for i := range numSpecTokens {", "for i := range numSpecTokens - 1 {"),
    # The deferred side of vllm:num_requests_waiting_by_reason (loggers.py:547-567), and aborts
    # (output_processor.py:552-613, core.py:2132-2140@5df63a72).
    ("X13", "capacity is the waiting requests less the deferred ones", "core/metrics/metrics.go",
     "float64(s.Waiting - s.Deferred)", "float64(s.Waiting)"),
    ("X14", "the deferred gauge is the deferred requests", "core/metrics/metrics.go",
     "m.gauges[\"vllm:num_requests_waiting_by_reason{reason=deferred}\"] = float64(s.Deferred)",
     "m.gauges[\"vllm:num_requests_waiting_by_reason{reason=deferred}\"] = 0"),
    ("X15", "a client-aborted request is in no finish-reason sample", "core/metrics/metrics.go",
     "\tfor _, id := range s.Forgotten {\n\t\tdelete(m.pending, id)",
     "\tfor _, id := range s.Forgotten {\n\t\tm.counters[\"vllm:request_success_total{finished_reason=abort}\"]++\n\t\tdelete(m.pending, id)"),
    ("X16", "a request the engine aborts is counted with reason abort", "core/metrics/metrics.go",
     "m.counters[\"vllm:request_success_total{finished_reason=\"+f.Reason+\"}\"]++",
     "if f.Reason != \"abort\" {\n\t\t\tm.counters[\"vllm:request_success_total{finished_reason=\"+f.Reason+\"}\"]++\n\t\t}"),
    ("X17", "an engine-sent abort output is an iteration of the logger", "core/metrics/metrics.go",
     "\tif s.EngineAborts > 0 {\n\t\tm.observe(\"vllm:iteration_tokens_total\", 0)\n\t}", "\t_ = s.EngineAborts"),
]

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
