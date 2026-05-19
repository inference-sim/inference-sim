# Saturation Point Discovery - Experiment Report

**Model**: Llama-3.1-8B-Instruct
**Hardware**: H100 GPU (single card)
**Date**: May 18-19, 2026

---

## What We Did

We used BLIS's **automated saturation search** to find the maximum safe load this system can handle. The search tested different request rates and used a **composite saturation detector** to identify when the system starts to get overloaded.

**Result**: The saturation point is **12.41 requests/second**

### Experiment Settings

**Workload**: Multi-cohort afternoon traffic pattern with 5 SLO classes
- 5 cohorts: critical, standard, batch, background, sheddable
- Each cohort has **population = 1** (one independent client per cohort)
- All cohorts share the **same arrival rate** (equal split of total load)
- Example: At 12.41 req/s total → each cohort sends 2.48 req/s

**Configuration**:
- Model: Llama-3.1-8B-Instruct (BF16 precision)
- Hardware: Single H100 GPU (TP=1)
- Settings: mbt=2048, max_model_len=8192, priority scheduling enabled
- Duration: 600 seconds (10 minutes) per observation

---

## What is "Saturation"?

Think of it like a highway:
- **Below saturation**: Cars flow smoothly, travel time is predictable
- **At saturation**: Highway is at capacity but still moving
- **Above saturation**: Traffic jams form, delays grow over time

For an LLM serving system:
- **Below saturation**: Response time depends mainly on how long your prompt/response is
- **Above saturation**: Response time depends mainly on how many people are waiting in line

---

## How the Search Worked

### Binary Search with 7 Test Runs

| Rate | Multiplier | What Happened | Verdict |
|------|------------|---------------|---------|
| 4.51 req/s | 1× (baseline) | Everything fine | ✓ STABLE |
| 9.02 req/s | 2× | Still good | ✓ STABLE |
| 18.05 req/s | 4× | Overloaded! | ❌ OVERLOADED |
| 13.53 req/s | 3× | Still overloaded | ❌ OVERLOADED |
| 11.28 req/s | 2.5× | Back to stable | ✓ STABLE |
| 12.41 req/s | 2.75× | Still stable | ✓ STABLE |
| 12.97 req/s | 2.875× | Just crossed threshold | ❌ OVERLOADED |

**Found**: Maximum stable rate is between 12.41 and 12.97 req/s

---

## How the Detector Works

The **composite saturation detector** looks at three signals:

### 1. Latency Trend (Most Important)
- Measures if response times are increasing over time
- **Stable system**: Response times stay flat
- **Overloaded system**: Response times keep growing

### 2. Quartile Pattern
- Looks at p25, p50, p75, p90 latency percentiles
- **Stable system**: Percentiles vary based on request size
- **Overloaded system**: All percentiles increase in lockstep (queue effects)

### 3. Rate Deficit
- Checks if system is actually handling the target load
- **Note**: This signal was NOT reliable in our tests (see below)

---

## Validation: Two Real Observations

We ran two actual 10-minute observations to validate the detector:

### Observation 1: exp1-saturation (12.41 req/s)
**Purpose**: Capture the system at maximum safe capacity

**Results**:
- Total requests: 37,171
- Saturation score: **0.0014** (near zero = healthy)
- Latency trend: Flat over 10 minutes
- Queue behavior: No backlog growth

**Verdict**: ✓ **STABLE** - System is running at full capacity without degrading

### Observation 2: exp1-overloaded (12.97 req/s)
**Purpose**: Test just beyond the threshold (+4.5% higher rate)

**Results**:
- Total requests: 38,860
- Saturation score: **0.885** (high = unstable)
- Latency trend: Rising over time
- Queue behavior: Backlog accumulating

**Verdict**: ❌ **OVERLOADED** - System is struggling, latency is growing

---

## Key Finding: Small Increase, Big Impact

**Going from 12.41 to 12.97 req/s (just 4.5% more load)**:
- Saturation score jumped **630×** (from 0.001 to 0.885)
- Latency trend changed from flat to rising
- Queue pattern changed from stable to monotonic

**This shows the saturation boundary is sharp** - there's a clear tipping point.

---

## What We Learned

### ✓ What Worked

1. **Latency trend is the key signal**
   - Caught overload early at 12.97 req/s
   - Clear difference between stable and overloaded cases

2. **Quartile pattern confirms saturation**
   - Stable: Varied percentiles (workload diversity shows)
   - Overloaded: Strict monotonic pattern (queue effects dominate)

3. **Binary search is efficient**
   - Found saturation point in just 7 runs
   - Final precision: ±0.56 req/s

### ⚠️ What Didn't Work

**Rate deficit is NOT a good signal**:
- At 12.41 req/s (stable): 0.14% deficit
- At 12.97 req/s (overloaded): 0.09% deficit (lower!)
- System can keep up with arrival rate even when queues are building

**Lesson**: Don't rely on throughput measurements alone - you need to look at latency behavior

---

## Comparison Table

| Metric | Saturation (12.41 req/s) | Overloaded (12.97 req/s) | Change |
|--------|-------------------------|--------------------------|--------|
| **Rate** | 2.75× baseline | 2.875× baseline | +4.5% |
| **Saturation Score** | 0.0014 | 0.885 | **+630×** |
| **Latency Trend** | Flat (0.056) | Rising (0.885) | **+16×** |
| **Queue Pattern** | Varied | Monotonic | Dominated |
| **Rate Deficit** | 0.14% | 0.09% | Actually lower! |

---

## Practical Takeaways

### For Capacity Planning
- **Safe operating point**: 12.41 req/s
- **Headroom**: 2.75× the baseline rate
- **Warning zone**: Anything above 12.5 req/s risks degradation

### For Monitoring
- **Watch latency trends**, not just throughput
- Look for increasing p90/p99 over time
- Check if percentiles become strictly ordered (sign of queuing)

### For BLIS Calibration
- **Use exp1-saturation (12.41 req/s) for training models**
  - Clean execution-physics data
  - Latency predictable from prompt/response sizes

- **Use exp1-overloaded (12.97 req/s) for validation only**
  - Tests if simulator can reproduce queue effects
  - Don't train on this data (mixing physics + queue dynamics)

---

## Bottom Line

We successfully identified that:
1. This system maxes out at **12.41 req/s** (2.75× baseline)
2. The transition to overload is **sharp** (4.5% more load causes problems)
3. **Latency trends matter more than throughput** for detecting saturation
4. We have two clean observations: one at capacity, one just beyond

These observations provide ground truth for calibrating and validating the BLIS simulator.
