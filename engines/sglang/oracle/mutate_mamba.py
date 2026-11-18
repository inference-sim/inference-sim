# Original to blis-sglang.
"""Mutation check of the hybrid mamba pools (S48, #94): each mutant changes one decision of the port
in core/cache/mamba.go or core/cache/mamba_budget.go; the replay of SGLang's traces
(TestLinearStateIsSGLangs, TestHybridPoolBytesAreSGLangs, TestHybridRequestPoolAnswersAsSGLang,
TestMambaBudgetIsSGLangs) must fail on every one.
Usage: python3 -I oracle/mutate_mamba.py [index...]"""
import os
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core", "cache")
GOVER = "go" + re.search(r"^go (\S+)", open(os.path.join(ROOT, "..", "..", "go.mod")).read(), re.M).group(1)
M = [
    ("mamba.go", "Conv:           [][2]int{{convDim / tp, c.ConvKernel - 1}},", "Conv:           [][2]int{{convDim / tp, c.ConvKernel}},", "conv window length"),
    ("mamba.go", "nGroups += tp - nGroups // extra_groups_for_head_shards", "nGroups += 0 // extra_groups_for_head_shards", "key heads not replicated"),
    ("mamba.go", "kHeads = (nGroups + tp - 1) / tp", "kHeads = nGroups / tp", "key heads per rank rounds down"),
    ("mamba.go", "convDim := intermediate + 2*nGroups*c.KeyHeadDim", "convDim := intermediate + nGroups*c.KeyHeadDim", "conv dim omits value of k"),
    ("mamba.go", "slots := int64(p.Size) + 1", "slots := int64(p.Size)", "padding slot not charged"),
    ("mamba.go", "func DedupConvWindow(eagleTopK int) bool { return eagleTopK <= 1 }", "func DedupConvWindow(eagleTopK int) bool { return false }", "window never deduplicated"),
    ("mamba.go", "func DedupConvWindow(eagleTopK int) bool { return eagleTopK <= 1 }", "func DedupConvWindow(eagleTopK int) bool { return true }", "tree verify deduplicated"),
    ("mamba.go", "total += layers * specSlots * d * st.temporalNumel() * ssmB // intermediate_ssm", "total += layers * slots * d * st.temporalNumel() * ssmB // intermediate_ssm", "intermediate state sized by size"),
    ("mamba.go", "if !p.ReplaySpec {\n\t\t\ttotal += layers * specSlots * d", "if true {\n\t\t\ttotal += layers * specSlots * d", "replay spec keeps intermediate state"),
    ("mamba.go", "num, ring = specSlots, convB", "num, ring = slots, convB", "spec ring sized by size"),
    ("mamba.go", "if p.ReplaySpec && ring != 4 {", "if false {", "ring low parts omitted"),
    ("mamba.go", "case lazy:\n\t\treturn 1\n\tcase overlap:\n\t\treturn 2", "case lazy:\n\t\treturn 2\n\tcase overlap:\n\t\treturn 2", "lazy tracks two"),
    ("mamba.go", "case overlap:\n\t\treturn 2\n\t}\n\treturn 1", "case overlap:\n\t\treturn 1\n\t}\n\treturn 1", "overlap tracks one"),
    ("mamba.go", "a.free = append(a.free, slots...)", "a.free = append(slots, a.free...)", "freed slots go to the head"),
    ("mamba.go", "got := slices.Clone(a.free[:need])\n\ta.free = a.free[need:]", "got := slices.Clone(a.free[len(a.free)-need:])\n\ta.free = a.free[:len(a.free)-need]", "slots taken from the tail"),
    ("mamba.go", "last := p.PingPongOtherIdx(0)\n\tif p.lazy {\n\t\tlast = 0\n\t}", "last := p.PingPongOtherIdx(0)", "lazy last track index"),
    ("mamba.go", "if p.pingPongSize == 2 {\n\t\treturn 1 - next\n\t}", "if p.pingPongSize == 3 {\n\t\treturn 1 - next\n\t}", "other ping-pong index"),
    ("mamba.go", "p.SetPingPongSlot(row, m, idx, newSlot)\n\treturn donated", "return donated", "donation keeps the old slot"),
    ("mamba.go", "if (keep == nil || i != *keep) && !m.PingPongEmpty[i] {", "if !m.PingPongEmpty[i] {", "kept tracking slot freed"),
    ("mamba.go", "m.NeedsClear = true", "m.NeedsClear = false", "new slot not marked for clearing"),
    ("mamba.go", "if p.extraBuffer && m.PingPongBuffer == nil {", "if p.extraBuffer {", "tracking slots allocated again"),
    ("mamba.go", "m.PingPongBuffer, m.PingPongEmpty, m.NextTrackIdx, m.LastTrackIdx = nil, nil, nil, nil", "m.PingPongBuffer, m.PingPongEmpty = nil, nil", "free keeps the track indices"),
    ("mamba_budget.go", "mambaBase = 3", "mambaBase = 2", "base slots per request"),
    ("mamba_budget.go", "extra = MambaTrackingSlots(true, !disableOverlap, lazy)", "extra = 1", "tracking slots in the ratio"),
    ("mamba_budget.go", "if disableRadixCache {\n\t\treturn 1, nil", "if false {\n\t\treturn 1, nil", "no-radix ratio"),
    ("mamba_budget.go", "budgetBytes := rest * b.FullMemoryRatio / (1 + b.FullMemoryRatio) * gib", "budgetBytes := rest * b.FullMemoryRatio * gib", "budget share"),
    ("mamba_budget.go", "size = int(pyFloorDiv(budgetBytes-float64(fixed)-perSlot, perSlot))", "size = int(pyFloorDiv(budgetBytes-float64(fixed), perSlot))", "padding slot in the solve"),
    ("mamba_budget.go", "num := budgetBytes - float64(perReq)*(1+d) - float64(extraPerSlot)", "num := budgetBytes - float64(extraPerSlot)", "spec solve omits the fixed term"),
    ("mamba_budget.go", "den := float64(perReq)*(1+d/float64(ratio)) + float64(extraPerSlot)", "den := float64(perReq)*(1+d) + float64(extraPerSlot)", "spec solve charges every slot a draft"),
    ("mamba_budget.go", "reservesSpec := hasSpec && !b.ReplaySpec", "reservesSpec := hasSpec", "replay still reserves intermediate state"),
    ("mamba_budget.go", "size = *b.MaxMambaCacheSize / b.AttnDPSize", "size = *b.MaxMambaCacheSize", "explicit size per dp worker"),
    ("mamba_budget.go", "case b.DisableRadixCache && b.MaxRunningRequests != nil:", "case b.MaxRunningRequests != nil:", "running requests set the size under radix"),
    ("mamba_budget.go", "state := (float64(int64(size+1)*(perReq+extraPerSlot)) + float64(fixed)) / gib", "state := (float64(int64(size)*(perReq+extraPerSlot)) + float64(fixed)) / gib", "state memory omits the padding slot"),
    ("mamba_budget.go", "n = min(n, mambaSize/ratio)", "n = min(n, mambaSize)", "request cap ignores the ratio"),
    ("mamba_budget.go", "if size <= 0 {", "if size < 0 {", "empty state cache accepted"),
    ("mamba_budget.go", "fixed = ring * int64(max(1, *b.MaxRunningRequests/b.AttnDPSize)+1)", "fixed = ring * int64(max(1, *b.MaxRunningRequests/b.AttnDPSize))", "replay scratch slots"),
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
        r = subprocess.run(["go", "test", ".", "-run", "TestLinearState|TestHybrid|TestMambaBudget", "-count=1"], cwd=ROOT,
                           env={**os.environ, "GOTOOLCHAIN": GOVER}, capture_output=True, text=True, timeout=300)
        out = "caught" if r.returncode != 0 else "SURVIVED"
        if "build failed" in r.stdout or "[build failed]" in r.stdout:
            out = "BUILD FAILED"
    finally:
        open(p, "w").write(src)
    res.append((i, name, out))
for r in res:
    print(*r)
