# Iteration 13: Pre-Design Analysis

## Step 1: Baseline Simulator Analysis

### From baseline_errors.json:

**Universal Pattern: TTFT Overestimation**
| Simulator | TTFT Range | Pattern |
|-----------|-----------|---------|
| blis-roofline | +330% to +3031% | Massive overestimation on ALL codegen workloads |
| llm-optimizer | +95% to +1971% | Same failure mode as roofline |
| aiconfigurator | +36% to +2081% | Same failure mode |
| **vidur** | **-14% to -32%** | **ONLY simulator avoiding overestimation!** |

**Critical Insight**: All analytical models (roofline, llm-optimizer, aiconfigurator) systematically overestimate TTFT, often by 10-30×. Vidur is the ONLY simulator with reasonable TTFT errors.

**Codegen is Universal Killer**:
- Mistral codegen: roofline +1031%, llm-optimizer +1971%
- Llama-2 codegen: roofline +587%, llm-optimizer +619%
- Llama-3.1 codegen: roofline +912%
- **All analytical models catastrophically fail on codegen**

**Scout/MoE Wildly Inconsistent**:
- Scout general: roofline -99.9% TTFT (predicts near-zero!)
- Scout reasoning: roofline -92% TTFT
- Scout codegen: roofline -50% TTFT
- Scout roleplay: roofline +66% TTFT
- **No consistent pattern - errors range from -99% to +66%**

**ITL Universally Underestimated**:
- ALL simulators: -20% to -70% ITL MPE
- **Decode modeling systematically wrong across ALL simulators**

### Why Vidur Succeeds (Need to investigate):
- What does vidur model that roofline doesn't?
- Why does vidur avoid catastrophic TTFT overestimation?
- Key hypothesis: vidur likely models framework overhead, scheduler delays, or queuing effects that pure roofline misses

---

## Step 2: Training Journey Summary (Iter0-12)

### Stable Iterations (0-7):
| Iter | Loss | Architecture | Key Innovation | Status |
|------|------|-------------|----------------|--------|
| 0 | 200% | Basic roofline | Baseline | ❌ Insufficient |
| 1 | 135% | +5 overhead terms | TP, KV, chunk, MoE | ⚠️ Progress |
| 2 | N/A | Sigmoid transition | Smooth decode transition | ✅ Design pattern |
| 3-5 | Failed | Long-context terms | Wrong assumption | ❌ Wasted |
| 6 | 162% | β₆ → QueueingTime | Decoupling breakthrough | ⚠️ Stable baseline |
| **7** | **155%** | **+ β₇ decode overhead** | **Stabilized β₁/β₄** | **✅ Best stable** |

**Key Finding**: **Iter7 (155%) > Iter6 (162%)** - Loss improved AND β₇ stabilized other coefficients

### Scout Discovery (Iter8):
- Added β₈ (MoE routing overhead)
- Captured real 39ms Scout mechanism
- Overall: 0pp improvement (loss 155% → 155%)
- **Verdict**: β₈ captures REAL mechanism but Scout problems remain

### Critical Discovery (Iter9):
**Scout bottleneck is sequence-length-dependent, NOT architecture-dependent!**

Evidence:
- Short-sequence Scout improved massively: roleplay -53pp, codegen -34pp
- Long-sequence Scout failed: general-lite 0pp, reasoning-lite -8pp
- β₉ (FP8 hypothesis) rejected: converged to 0.14μs (essentially zero)

**Inverse correlation**: Longer sequences → worse performance

Coefficient explosions after β₉ rejection:
- β₆ (scheduler): +654% (13ms → 99ms)
- β₂ (TP comm): +343% (0.18 → 0.82)
- β₈ (MoE routing): +143% (30μs → 73μs)
- **These absorbed the error β₉ was supposed to capture**

### Catastrophic Failures (Iter10-12):
| Iter | Loss | What Added | What Failed |
|------|------|-----------|-------------|
| 10 | 4267% | β₁₀ (batching inefficiency) | Catastrophic explosion |
| 11 | 4084% | Audited β₁₀ (proven correct) | Still catastrophic |
| 12 | 2590% | Widened β₃' bounds | β₃' collapsed (moved AWAY from target) |

**Root Cause Analysis**:
1. **Warm-start paradox**: Iter10-12 started from iter9's inflated coefficients
2. **Collinearity**: Multiple terms (β₁₀, β₃', β₆, β₂) fighting for same error
3. **Coefficient cascade**: One inflation triggers others

---

## Step 3: What Actually Worked

### Proven Mechanisms (Keep These):
1. **β₆ in QueueingTime** (iter6): Decouples scheduler from compute ✅
2. **β₇ decode overhead** (iter7): Stabilized β₁/β₄ ✅
3. **β₈ MoE routing** (iter8): Captures real 39ms Scout mechanism ✅
4. **β₁₀ batching inefficiency** (iter10): Proven correct in unit tests, addresses sequence-length discovery ✅

### Why Iter7 > Iter6:
- Lower loss: 155% vs 162%
- Has β₇ which stabilized β₁ (1.851 → 1.108) and β₄ (1.451 → 0.713)
- All non-Scout experiments performed better

### Critical Constraint:
**Scout experiments dominate error budget**: 4 Scout = 49% of total loss despite being 27% of experiments

---

## Step 4: Root Cause of Iter9-12 Failures

### Primary Issue: Warm-Start from Inflated Coefficients
- Iter9 had β₆=99ms, β₂=0.82, β₈=73μs (all inflated after β₉ rejection)
- Iter10-12 started from these inflated values
- Optimizer struggled to escape bad local optimum

### Secondary Issue: Collinearity
When multiple terms target same phenomenon:
- β₃' (KV seq-len scaling)
- β₁₀ (batching inefficiency)
- Both scale with sequence length → optimizer confused

### Tertiary Issue: Missing Mechanisms
All analytical simulators (roofline, llm-optimizer, aiconfigurator) fail on:
1. Framework overhead (vidur likely models this)
2. Scheduler queuing delays
3. Batching dynamics
4. Cold-start effects

---

## Step 5: Design Strategy for Iter13

### Option A: Start from Iter7 + Minimal Changes
**Architecture**: 3 alpha + 9 beta (iter7's β₀-β₇ + iter8's β₈ + iter10's β₁₀)
**Rationale**:
- Iter7 most stable (155%)
- Keep proven β₇ and β₈
- Add β₁₀ for sequence-length discovery
- **NO β₃'** (causes collinearity with β₁₀)

**Expected**: Loss 155% → 120-140% (modest improvement, stable foundation)

### Option B: Start from Iter9 with Reversion
**Architecture**: 3 alpha + 10 beta (keep iter9 terms but revert coefficients)
**Rationale**:
- Iter9 had right architecture (β₀-β₈)
- Problem was inflated coefficients, not structure
- Warm-start from **iter7** coefficients (not iter9)
- Add β₁₀ carefully

**Expected**: Loss 161% → 110-130% (larger improvement, higher risk)

### Option C: Architectural Simplification (What I Tried Before - WRONG)
**Architecture**: 3 alpha + 7 beta (revert to iter6)
**Rationale**: ❌ REJECTED
- Loses proven β₇ and β₈
- Ignores sequence-length discovery
- Iter6 worse than iter7

---

## Step 6: Recommendation

**I recommend Option A: Start from Iter7 + β₈ + β₁₀**

**Architecture**: 3 alpha + 10 beta
- β₀-β₇: From iter7 (proven stable, 155% loss)
- β₈: From iter8 (captures Scout MoE routing, 39ms mechanism)
- β₉: Remove (FP8 hypothesis rejected)
- β₁₀: Add (addresses sequence-length discovery from iter9)
- β₃': Do NOT add (causes collinearity with β₁₀)

**Warm-start from**: Iter7 optimal coefficients (NOT iter9-12 inflated values)

**Expected outcome**:
- Loss: 155% → 120-140% (15-35pp improvement)
- Address sequence-length bottleneck via β₁₀
- Maintain stability (proven architecture + clean warm-start)
- If successful, iter14 can incrementally add β₃' with wider bounds

**Risk mitigation**:
- If loss doesn't improve, we have stable iter7 baseline
- If β₁₀ causes problems, remove it for iter14
- Incremental approach reduces catastrophic failure risk

---

## Step 7: Key Questions to Validate

1. **Are coefficients indices correct?** Iter7 has β₀-β₇, adding β₈ shifts indices
2. **What are iter7's exact coefficient values?** Need to extract from results
3. **How to handle β₆ index shift?** β₆ in QueueingTime at index 7 (iter7) → index 8 (iter13)?
4. **Do we need unit tests for β₁₀?** Iter11 already validated (0% error)

---

## Next Step: Extract Iter7 Coefficients and Design

