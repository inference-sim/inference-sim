# Review Agent Dispatch Prompts

Reference file for the hypothesis-experiment skill. Contains exact prompts for all 20 reviewer perspectives across the three review gates.

**Canonical source:** `docs/process/hypothesis.md` (v2.0). If prompts here diverge from hypothesis.md, the process doc is authoritative.

**Usage**: Launch each perspective as a parallel Task agent:
```
Task(subagent_type="general-purpose", model="haiku", run_in_background=True,
     prompt="<prompt from below>")
```

**After all agents complete**: Read each output file independently. Count CRITICAL and IMPORTANT findings yourself. Do NOT trust agent-reported totals.

---

## Section A: Design Review (5 perspectives) — Step 2

### DR-1: Hypothesis Quality

```
You are reviewing a hypothesis experiment design for the BLIS discrete-event simulator.

HYPOTHESIS AND DESIGN:
<paste hypothesis sentence, classification, experiment design>

YOUR FOCUS: Hypothesis Quality
- Is the hypothesis behavioral, testable, and diagnostic?
- Does it follow the family-specific sentence pattern from docs/standards/experiments.md?
- Is the diagnostic clause present ("If this fails, it would indicate...")?
- Is it correctly classified (family, VV&UQ category, type)?
- Is the hypothesis conceptual (not code-grounded)? It should NOT reference specific files/lines.

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DR-2: Experiment Design Rigor (ED-1–ED-6)

```
You are reviewing a hypothesis experiment design for the BLIS discrete-event simulator.

HYPOTHESIS AND DESIGN:
<paste hypothesis sentence, classification, experiment design>

YOUR FOCUS: Experiment Design Rigor
Check compliance with each standard:
- ED-1: Is exactly ONE dimension varied between configurations? Everything else constant?
- ED-2: Is there a rate where the effect should vanish, to confirm mechanism dependence?
- ED-3: Are preconditions verified in the script (not just prose)?
- ED-4: Is seed handling correct? (3+ seeds for statistical: 42, 123, 456)
- ED-5: Is the experiment reproducible from run.sh alone?
- ED-6: If reusing calibration from a prior experiment, is the config diff documented?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DR-3: Parameter Calibration

```
You are reviewing a hypothesis experiment design for the BLIS discrete-event simulator.

HYPOTHESIS AND DESIGN:
<paste hypothesis sentence, classification, experiment design>

YOUR FOCUS: Parameter Calibration
- Are parameters computed analytically from known coefficients, not guessed?
  - Beta coefficients (llama-3.1-8b, H100, TP=2): [6910.42, 17.67, 2.84]
    stepTime = 6910.42 + 17.67*cacheMissTokens + 2.84*decodeTokens (microseconds)
  - Alpha coefficients: [1601.35, 3.51, 1805.54]
    queueDelay = 1601.35 + 3.51*inputLen; outputProcessing = 1805.54 (microseconds)
- Are capacity estimates matched to the actual workload mode?
  - CLI mode (--rate): uses defaults prompt=512, output=512 -> step ~17.4ms, capacity ~57.4 req/s
  - Workload-spec YAML: uses distributions defined in the YAML (compute step time from YAML params)
- Is the operating point correct? (near saturation for queueing effects, sub-saturation for baseline)
- Is --total-kv-blocks appropriate? (CLI default is overridden by defaults.yaml to 132139 for llama/H100/TP=2)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DR-4: Control Completeness

```
You are reviewing a hypothesis experiment design for the BLIS discrete-event simulator.

HYPOTHESIS AND DESIGN:
<paste hypothesis sentence, classification, experiment design>

YOUR FOCUS: Control Completeness
- Does every proposed mechanism have a planned control experiment? (RCV-4)
- Does each control isolate exactly one variable?
- Is the baseline configuration clearly defined?
- Are there confounding variables that could explain results without the proposed mechanism?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DR-5: DES and Domain Fit

```
You are reviewing a hypothesis experiment design for the BLIS discrete-event simulator.

HYPOTHESIS AND DESIGN:
<paste hypothesis sentence, classification, experiment design>

YOUR FOCUS: DES and Domain Fit
- Will the experiment create the conditions needed for the hypothesis to be testable?
- Are there DES-specific subtleties that could confound results?
  - Alpha overhead (~4.3ms/req) is non-blocking but inflates E2E metrics
  - Step quantization: step time is per-batch, not per-request
  - Clock granularity: microsecond ticks
  - Event ordering: (timestamp, priority, seqID) at cluster level
- Is the experiment duration sufficient? Request count adequate for the intended effect?
- Is the warmup period adequate (or does the experiment need cold-start behavior)?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

---

## Section B: Code Review (5 perspectives) — Step 5

### CR-1: Parser–Output Format Agreement

```
You are code-reviewing a BLIS hypothesis experiment BEFORE it runs.

FILES TO REVIEW:
<paste or reference run.sh and analyze.py paths>

YOUR FOCUS: Parser–Output Format Agreement
For every regex or field extraction in analyze.py, verify the pattern matches actual output:
- Read cmd/root.go — what text does the CLI print? (format strings for "Preemption Rate: %.4f", etc.)
- Read sim/metrics_utils.go — what JSON fields exist in MetricsOutput?
- Match every regex in analyze.py against the format string in the producer code
- SILENT DEFAULTS: verify that when a regex matches nothing, analyze.py warns to stderr rather than silently defaulting to 0
- Check parse_blis_output() from hypotheses/lib/analyze_helpers.py — does it extract what this experiment needs? (NOTE: it does NOT extract dropped_unservable)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### CR-2: CLI Flag Correctness

```
You are code-reviewing a BLIS hypothesis experiment BEFORE it runs.

FILES TO REVIEW:
<paste or reference run.sh path>

YOUR FOCUS: CLI Flag Correctness
For every flag in run.sh:
- Verify the flag name exists in cmd/root.go
- Verify the value type matches (string, int, float)
- Check for typos that strict YAML parsing would reject
- Verify --stderr comes BEFORE other flags in blis_run calls (harness checks position 3 only)
- Cross-reference blis_run timeout tiers: TIMEOUT_QUICK (<100 req), TIMEOUT_STANDARD (100-500), TIMEOUT_EXTENDED (>500)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### CR-3: YAML Field Validation

```
You are code-reviewing a BLIS hypothesis experiment BEFORE it runs.

FILES TO REVIEW:
<paste or reference run.sh and any .yaml files>

YOUR FOCUS: YAML Field Validation
If the experiment uses workload-spec YAML files:
- Verify field names against sim/workload/spec.go struct tags
- Key gotchas: `id:` not `client_id:`, `process:` not `type:` in arrival, `aggregate_rate:` top-level, `rate_fraction:` per client, distribution params under `params:` with `std_dev` not `stdev`, `prefix_group`+`prefix_length` not `prefix_tokens`
- KnownFields(true) will reject typos at runtime — catching them now saves a failed run
- If YAML is generated inline in run.sh (heredoc), verify the heredoc syntax is correct

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### CR-4: Config Diff (ED-6)

```
You are code-reviewing a BLIS hypothesis experiment BEFORE it runs.

FILES TO REVIEW:
<paste or reference run.sh path>

YOUR FOCUS: Config Diff (ED-6)
- If the experiment reuses calibration from a prior experiment, diff every CLI flag and YAML field between the two
- Explicitly list all differences
- Verify the # Reference: comment in run.sh points to the correct file
- Check: does any changed flag (routing policy, seed, rate, blocks) invalidate the calibration?
- Evidence: H10 used --routing-policy least-loaded while H8 used round-robin, shifting the preemption cliff. Caught only in post-publication review.

If no prior experiment is referenced, report "N/A — no referenced experiment" and move on.

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### CR-5: Seed and Determinism

```
You are code-reviewing a BLIS hypothesis experiment BEFORE it runs.

FILES TO REVIEW:
<paste or reference run.sh path>

YOUR FOCUS: Seed and Determinism
- Verify --seed is passed correctly in every blis_run call
- Verify workload YAML seed: field doesn't conflict with --seed (CLI overrides YAML when explicit)
- Verify seeds vary across runs as intended (ED-4)
- Check that run.sh builds the binary (setup_experiment) and is fully self-contained (ED-5)
- Verify run.sh sources hypotheses/lib/harness.sh
- Verify every blis_run call has a timeout tier (no bare $BINARY run calls)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

---

## Section C: FINDINGS Review (10 perspectives) — Step 8

### FR-1: Code Verifier

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>

YOUR FOCUS: Code Verification
- READ the actual source files cited in FINDINGS.md. Verify every file:line citation.
- Does the code at the cited location actually produce the claimed behavior?
- Are there off-by-one errors in line citations? (Acceptable: +/-2 lines. Flag: >2 lines off.)
- Does the mechanism explanation match what the code does, not just what it's named?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-2: Experiment Designer

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>
RUN SCRIPT: <path to run.sh>

YOUR FOCUS: Experiment Design Compliance
- ED-1 through ED-6 compliance (controlled comparison, rate awareness, preconditions, seeds, reproducibility, config diff)
- Are there missing control experiments or confound matrix cells?
- Are parameters properly calibrated?
- Cross-reference every CLI flag in run.sh against cmd/root.go flag definitions
- Cross-reference every YAML field name against sim/workload/spec.go struct tags

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-3: Statistical Rigor

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>

YOUR FOCUS: Statistical Rigor
- Are "surprises" computed from first principles? (RCV-2)
- Is the sample size adequate (seeds, operating points)?
- Are claims properly scoped (not over-generalized)?
- Is the evidence quality table complete and honest?
- Effect size thresholds: >20% for dominance (ALL seeds), <5% for equivalence (ALL seeds), <10% in any seed = inconclusive
- Is the status classification consistent with the data?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-4: Control Experiment Auditor

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>
RUN SCRIPT: <path to run.sh>

YOUR FOCUS: Control Experiment Audit
- Does every proposed mechanism (RCV-3) have a control experiment (RCV-4)?
- Were controls EXECUTED, not just proposed? Look for past tense with data vs conditional language ("could be confirmed by")
- Does each control isolate exactly one variable? Diff CLI flags between treatment and control in run.sh
- Do control results confirm or refute the proposed mechanism?
- Do Evidence Quality table entries reflect the current round (not stale from prior rounds)?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-5: Standards Compliance

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>

YOUR FOCUS: Standards Compliance
- Are ALL FINDINGS.md sections present and non-empty? (per docs/templates/hypothesis.md)
- Is the hypothesis correctly classified (family, VV&UQ, type)?
- Does the Devil's Advocate section argue both directions convincingly? (RCV-5)
- Are Scope and Limitations complete? (RCV-6) Operating point, dependencies, what was NOT tested, generalizability, UQ
- Does the Standards Audit check against docs/standards/rules.md and docs/standards/invariants.md?
- Are any new rules or invariants warranted?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-6: Substance and Logic

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>

YOUR FOCUS: Substance and Logic
- Are there logical errors in the conclusions?
- Are there mathematical mistakes in effect size calculations or statistical claims?
- Does the evidence actually support the claims? (Not just "the numbers are close enough")
- Are alternative explanations adequately considered?
- Does the analyzer verdict in analyze.py match the FINDINGS.md status? If not, is the discrepancy acknowledged?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-7: DES Mechanism Expert

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>

YOUR FOCUS: DES Mechanism Analysis
- Are there event-ordering subtleties that could explain results differently?
- Are assumptions about DES timing correct?
  - Alpha overhead (~4.3ms/req) is non-blocking (delays enqueue, not step advance)
  - Step quantization: step time is per-batch
  - Clock: microsecond ticks
- Could the result be a simulation artifact rather than modeled system behavior?
- Are routing snapshot freshness assumptions correct? (INV-7: PendingRequests = synchronous, KVUtilization = stale)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-8: Reproducibility and Robustness

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>
RUN SCRIPT: <path to run.sh>

YOUR FOCUS: Reproducibility and Robustness
- Can run.sh reproduce results from scratch on a clean checkout?
- Are results fragile to small parameter variations? (Would +/-10% on key params change the conclusion?)
- Are all intermediate files generated by the script, not checked in as stale artifacts?
- Does run.sh build the binary (setup_experiment call)?
- Are there any non-deterministic dependencies (timestamps, system load, file ordering)?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-9: Cross-Experiment Consistency

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>

YOUR FOCUS: Cross-Experiment Consistency
- Do findings contradict any prior experiment? If so, is the contradiction acknowledged and explained?
- Check references to prior experiments — are specific claims accurate? (Read the referenced FINDINGS.md)
- Are there stale references to prior review rounds that should have been updated?
- Key prior findings to check against:
  - H4: RR and LL mean-equivalent at low load; LL p99 worse from tie-breaking bias
  - H7: Horizontal scaling is super-linear (7.4x TTFT p99 for 4->8 instances)
  - H20: ParetoLogNormal produces FEWER preemptions than Gaussian (median drives KV pressure)
  - H23: Uniform workloads eliminate policy differentiation even under overload

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### FR-10: User Guidance and Actionability

```
You are reviewing a FINDINGS.md for a BLIS hypothesis experiment.

FINDINGS FILE: <path to FINDINGS.md>

YOUR FOCUS: User Guidance and Actionability
- Are "Implications for Users" practical and specific enough to act on?
- Are proposed issues (bugs, enhancements, follow-up hypotheses) well-scoped?
- Would a BLIS user reading this understand what to do differently?
- Are findings classified correctly in the Findings Classification table?
- Is the promotion assessment complete? (Should any confirmed findings become Go tests or formal invariants?)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```
