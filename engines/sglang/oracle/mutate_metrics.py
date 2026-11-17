# Original to blis-sglang.
"""Mutation check of core/metrics and the reports of core/sched (S15): each mutant changes one decision
of the port; the replay of SGLang's registry and KV-event bytes (TestSeriesEqualSGLangs) must fail on
every one. Usage: python3 -I oracle/mutate_metrics.py [index...]"""
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GOVER = "go" + re.search(r"^go (\S+)", open(os.path.join(ROOT, "go.mod")).read(), re.M).group(1)  # the toolchain go.mod names
M = [
    ("core/metrics/metrics.go", "usage := float64(used) / float64(maxTok)", "usage := float64(used) / float64(maxTok+1)", "usage denominator"),
    ("core/metrics/metrics.go", "gauge(\"sglang:token_usage\", roundHalfEven2(usage))", "gauge(\"sglang:token_usage\", usage)", "token_usage is not rounded"),
    ("core/metrics/metrics.go", "used := maxTok - (c.avail + c.evict)", "used := maxTok - c.avail", "evictable counted as used"),
    ("core/metrics/metrics.go", "effIn := p.LogInputTokens - p.ReprocessedInput", "effIn := p.LogInputTokens", "reprocessed input counted as effective"),
    ("core/metrics/metrics.go", "effHit := p.LogHitTokens - p.ReprocessedHit", "effHit := p.LogHitTokens", "reprocessed hit counted as effective"),
    ("core/metrics/metrics.go", "c.addRealtime(\"prefill_cache\", p.LogHitTokens)", "", "prefill cache tokens"),
    ("core/metrics/metrics.go", "c.addRealtime(\"decode\", d.Generated)", "c.addRealtime(\"decode\", d.Generated+1)", "decode token count"),
    ("core/metrics/metrics.go", "\tif !r.Logged {\n\t\treturn\n\t}\n\tc.running, c.queue, c.sumSeq", "\tc.running, c.queue, c.sumSeq", "decode gauges every batch"),
    ("core/metrics/metrics.go", "c.running, c.queue, c.newRatio = p.Running, r.Waiting, p.NewTokenRatio", "c.running, c.queue, c.newRatio = p.Running, r.Waiting, 0", "new_token_ratio"),
    ("core/metrics/metrics.go", "m[\"priority\"] = \"\"", "_ = m", "priority label on every series"),
    ("core/metrics/metrics.go", "add(name, float64(q.ByPriority[p]), \"priority\", strconv.Itoa(p))", "add(name, float64(q.Total), \"priority\", strconv.Itoa(p))", "per-priority count"),
    ("core/metrics/metrics.go", "add(\"sglang:context_len\", float64(c.cfg.ContextLen))", "add(\"sglang:context_len\", float64(c.cfg.MaxContextLen))", "context_len gauge"),
    ("core/sched/report.go", "r.Retracted = s.numRetracted\n\ts.numRetracted = 0\n\ts.publishKVEvents(&r)\n\ts.reports = append(s.reports, r)\n}\n\n// reportDecode", "r.Retracted = s.numRetracted\n\ts.publishKVEvents(&r)\n\ts.reports = append(s.reports, r)\n}\n\n// reportDecode", "retracted count not reset by a prefill report"),
    ("core/sched/report.go", "interval := max(s.cfg.DecodeLogInterval, 1)", "interval := 1", "decode log interval"),
    ("core/sched/report.go", "sum += l", "sum += l - 1", "decode_sum_seq_lens"),
    ("core/sched/report.go", "Running: s.queueOf(b.Reqs)", "Running: s.queueOf(nil)", "running count of a decode report"),
    ("core/sched/report.go", "if r == nil {\n\t\ts.reports = append(s.reports, Report{Idle: true, Events: ev})\n\t\treturn\n\t}\n\tr.Events = ev", "if r == nil {\n\t\treturn\n\t}\n\tr.Events = ev", "events published at idle"),
    ("core/sched/scheduler.go", "s.numRetracted = len(retracted)", "s.numRetracted = 0", "retracted request count"),
    ("core/sched/adder.go", "a.logHitTokens += prefixLen", "a.logHitTokens += 0", "hit tokens of the adder"),
    ("core/sched/adder.go", "a.logInputTokens += extendInputLen", "a.logInputTokens += extendInputLen - 1", "input tokens of the adder"),
    ("core/sched/adder.go", "if !r.RetractedStain {\n\t\ta.logDeviceHitTokens += adm.prefixLen\n\t}", "a.logDeviceHitTokens += adm.prefixLen", "device hit of a retracted request"),
    ("core/sched/adder.go", "a.updatePrefillBudget(0, r.ExtendLen(), maxNew, r.RetractedStain)", "a.updatePrefillBudget(r.C.PrefixLen, r.ExtendLen(), maxNew, r.RetractedStain)", "a chunk continuation counts as a hit"),
    ("core/metrics/kvevents.go", "b = appendString(b, \"lora_id\")\n\t\tb = append(b, 0xc0)", "", "lora_id omitted"),
    ("core/metrics/kvevents.go", "case v >= 0 && v < 1<<8:\n\t\treturn append(b, 0xcc, byte(v))", "case v >= 0 && v < 1<<8:\n\t\treturn append(b, 0xcd, 0, byte(v))", "integer width"),
    ("core/metrics/kvevents.go", "b = appendString(b, \"parent_block_hash\")", "b = appendString(b, \"parent\")", "event field name"),
    ("core/metrics/kvevents.go", "return appendInt(b, int64(attnDPRank))", "return append(b, 0xc0)", "attn_dp_rank omitted"),
]
only = sys.argv[1:]
res = []
for i, (f, a, b, name) in enumerate(M):
    if only and str(i) not in only:
        continue
    p = os.path.join(ROOT, f)
    src = open(p).read()
    if a not in src:
        res.append((i, name, "NOT APPLIED"))
        continue
    open(p, "w").write(src.replace(a, b, 1))
    try:
        r = subprocess.run(["go", "test", "./core/metrics/", "-run", "TestSeriesEqual"], cwd=ROOT,
                           env={**os.environ, "GOTOOLCHAIN": GOVER}, capture_output=True, text=True, timeout=300)
        out = "caught" if r.returncode != 0 else "SURVIVED"
        if "build failed" in r.stdout:
            out = "BUILD FAILED"
    finally:
        open(p, "w").write(src)
    res.append((i, name, out))
for r in res:
    print(*r)
