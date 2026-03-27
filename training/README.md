# Agentic Latency Training

**Purpose**: Iteratively improve BLIS latency prediction through hypothesis-driven evolution.

---

## 📚 Documentation Structure

```
training/docs/
├── agent-workflow-discipline.md  ⭐ START HERE - Mandatory validation gates
├── iteration-workflow-checklist.md  - Detailed phase-by-phase guide  
├── agentic-latency-training-problem-statement.md  - Problem definition
├── outer-loop-specs.md  - 5-file deliverables specification
├── outer-inner-loop-contract.md  - Interface contract
├── generalization-validation-protocol.md  - CV testing strategy
└── cv-test-implementation.md  - CV test technical details
```

---

##  Validation Scripts (Mandatory Gates)

```
training/scripts/
├── validate_hypothesis.py  → GATE 1: Before implementation
├── validate_backend.py     → GATE 2: After Go code written
├── validate_iteration.py   → GATE 3: After optimization
├── analyze_errors.py       → Extract error patterns
├── run_cv_tests.py         → GATE 4: Cross-validation (iter1+)
└── monitor_optimization.py → Live progress dashboard (optional)
```

**All validation scripts MUST pass before proceeding to next phase.**

**Note on hypothesis validation**: `validate_hypothesis.py` only requires H-main section (with quantitative prediction). Other hypotheses (H-prefill-regime, H-tp-invariance, H-moe-specific, etc.) are iteration-specific and flexible — the script validates their structure when present but doesn't enforce which specific hypotheses must exist.

---

## ⚡ Quick Start for Iter0

```bash
# Phase 1: Hypothesis validation
python scripts/validate_hypothesis.py --hypothesis iter0-HYPOTHESIS.md

# Phase 2: Backend validation (after generating evolved_model.go)
python scripts/validate_backend.py evolved

# Phase 3: Optimization
cd training
python inner_loop_optimize.py --n-trials 50

# Phase 4: Analysis
python scripts/validate_iteration.py --iteration 0
python scripts/analyze_errors.py --results inner_loop_results.json --output iter0-error-analysis.md

# Phase 5: Document findings
# (Manually write iter0-FINDINGS.md based on validation reports)
```

---

## 🎯 Agent Discipline Rules

### 5 Mandatory Rules:

1. **NEVER skip validation** - Scripts are gates, not suggestions
2. **NEVER write Go code before hypothesis validated** - Gate 1 must pass
3. **NEVER start optimization with stale binary** - Recompile after every Go change
4. **NEVER skip error analysis** - Patterns reveal next iteration's design
5. **NEVER add >2 basis functions per iteration** - Incremental changes only

### Go Code Translation Requirements:

Every basis function MUST have:
- **Physics justification** (comment explaining hardware/software cost)
- **Expected coefficient range** (based on specs or profiling)
- **Functional form rationale** (why this math: linear, log, etc.)
- **Units consistency** (all times in microseconds)
- **Defensive bounds** (clamp to prevent overflow/negative)

Example:
```go
// β₃ × log₂(TP) × num_layers × all_reduce_latency
//
// Physics: Ring all-reduce scales O(log₂ N) with N ranks
// Expected β₃: 0.8-1.2 (near-ideal ring performance)
// Range: [0.5, 2.0] (allow for inefficiency)
tpCommOverhead := m.Beta[3] * math.Log2(float64(m.tp)) *
                  float64(m.modelConfig.NumLayers) * 50e-6  // 50μs per layer
```

---

## 📊 Decision Tree: Add vs Tune?

```
Is loss > 50%?
├─ YES: Add 1-2 new basis functions (structural change)
└─ NO: Model structure reasonable
    ├─ Random errors (no pattern)? → Tune coefficients
    ├─ Systematic errors (TP/model-dependent)? → Add targeted basis function
    ├─ Training low but CV high? → Remove basis functions (overfitting)
    └─ Bimodal distribution? → Add categorical basis (e.g., MoE flag)
```

---

## 🔬 Iteration Lifecycle

```
Iter{N-1}-FINDINGS.md
    ↓ (extract principles)
Iter{N}-HYPOTHESIS.md
    ↓ (validate hypothesis)
[GATE 1] validate_hypothesis.py
    ↓ (translate to Go)
evolved_model.go + coefficient_bounds.yaml
    ↓ (validate backend)
[GATE 2] validate_backend.py
    ↓ (optimize)
inner_loop_optimize.py (50 trials)
    ↓ (analyze)
[GATE 3] validate_iteration.py + analyze_errors.py
    ↓ (cross-validate, iter1+)
[GATE 4] run_cv_tests.py
    ↓ (document)
Iter{N}-FINDINGS.md
    ↓ (decide)
Converged? YES → STOP / NO → Iter{N+1}
```

---

## 📁 Files Per Iteration

```
training/
├── iter{N}-HYPOTHESIS.md           # Before implementation
├── iteration_manifest.yaml          # Generated (outer loop)
├── coefficient_bounds.yaml          # Generated (outer loop)
├── inner_loop_results.json         # Generated (optimization)
├── iter{N}-VALIDATION-REPORT.md    # Generated (validate_iteration.py)
├── iter{N}-error-analysis.md       # Generated (analyze_errors.py)
└── iter{N}-FINDINGS.md             # Manual (required!)
```

---

## ✅ Success Criteria

| Metric | Iter0 Target | Iter1 Target | Final Target |
|--------|--------------|--------------|--------------|
| Overall loss | < 35% | < 20% | < 10% |
| CV-1 MAPE | N/A | < 20% | < 15% |
| CV-2 MAPE | N/A | < 20% | < 15% |
| CV-3 MAPE | N/A | < 20% | < 15% |

**Converged when:**
- Overall loss < 10%
- All CV tests pass
- Error pattern is white noise (no systematic clustering)
- Analytical consistency checks pass

---

## 🚨 Common Mistakes

| Mistake | Fix |
|---------|-----|
| Skipping validation scripts | Run all gates - they catch real issues |
| Adding too many basis functions | Max 1-2 per iteration |
| No physics justification in comments | Every term needs hardware/software explanation |
| Mixed units (ms vs μs) | Always use microseconds |
| Forgot to recompile BLIS | `go build -o blis main.go` after every change |

---

## 📞 Support

For questions about:
- **Methodology**: See `docs/agent-workflow-discipline.md`
- **Scripts**: See script reference table in discipline doc
- **CV tests**: See `docs/generalization-validation-protocol.md`
- **Strategy Evolution**: See `../../docs/methodology/strategy-evolution.md`

---

**Remember**: Scripts are **validation gates**. If a gate fails, **FIX THE ISSUE** before proceeding.
