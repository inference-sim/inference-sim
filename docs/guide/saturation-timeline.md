# Saturation Timeline

A saturation detector answers one question: **was the system overloaded?** By default BLIS answers it once, for the whole run. The *saturation timeline* answers it repeatedly — at fixed intervals — so you can see **when** a system tipped over instead of just whether it did.

```bash
# Emit a saturation label every 5 seconds of simulated time into a report file
./blis run --model qwen/qwen3-14b --num-requests 200 \
  --post-hoc-detector composite \
  --saturation-interval 5s \
  --saturation-report saturation.json
```

The timeline lands in the `--saturation-report` file as a `saturation_timeline` array. It is **off by default**: without `--saturation-interval`, output is unchanged.

## What a label means

Each interval gets one of three labels:

| Label | Meaning |
|-------|---------|
| **UNSATURATED** | The system was keeping up as of this point in time. |
| **SATURATED** | The system was overloaded (sustained). |
| **UNSURE** | Not enough information to decide yet. This is **not** a middle ground between the other two — it means the detector can't call it. |

Labels are **cumulative**: the label at t=30s reflects everything from the start of the run up to t=30s.

### How the label is decided

At each interval the detector produces a raw severity (`STABLE` / `BACKLOGGED` / `OVERLOADED`) plus a confidence. That is collapsed into a label in two steps, **in this order**:

1. **UNSURE gate (checked first).** If too few requests have arrived **or** the detector's confidence is below the cutoff → **UNSURE**, regardless of severity.
2. **Severity mapping.** `OVERLOADED` → **SATURATED**; `STABLE` and `BACKLOGGED` → **UNSATURATED**.

!!! note "Why BACKLOGGED is not SATURATED"
    Only *sustained* overload counts as saturation. A transient queue build-up (BACKLOGGED) is treated as healthy, so a brief spike doesn't get flagged as saturation.

## Choosing a detector

Pass the detector with `--post-hoc-detector`. Two run live:

| Detector | How it decides | Best for |
|----------|----------------|----------|
| **composite** | Combines rate deficit (arrivals vs completions) and a smoothed latency trend. Confidence grows with request count (`arrivals / 20`). | General use. Steady, low-noise verdict. **No tuning needed.** |
| **backlog-drift** | Regression over time-windowed backlog. Confidence grows with the number of complete windows. | Detecting *transient* backlog spikes. More reactive. **Needs window tuning** (see below). |

(`threshold` — a simple mean-latency cutoff — and `none` also exist; `none` skips the timeline.)

Under the hood both satisfy one `LiveDetector` interface and run through the same event-loop wiring, so new detectors can be added without touching the CLI.

## Tuning

### The UNSURE thresholds (both detectors)

| Flag | Default | Effect |
|------|---------|--------|
| `--saturation-unsure-min-requests` | 20 | Below this many cumulative arrivals, a point is UNSURE. |
| `--saturation-unsure-min-confidence` | 0.5 | Below this detector confidence, a point is UNSURE. |

### backlog-drift window size (backlog-drift only)

backlog-drift bins data into fixed-width windows and needs several complete windows before it is confident. The default window is **60 seconds** — so on a run shorter than a few minutes it never accumulates enough windows and stays permanently UNSURE.

For short runs, shrink the window:

```bash
./blis run --model qwen/qwen3-14b --num-requests 200 \
  --post-hoc-detector backlog-drift --saturation-interval 5s \
  --saturation-window 3 --saturation-min-windows 3 \
  --saturation-report saturation.json
```

!!! tip
    `composite` needs no such tuning — prefer it unless you specifically want backlog-drift's transient sensitivity.

## Reading the output

Each entry in `saturation_timeline`:

```json
{
  "clock_us": 10000000,     // simulation time of this point (µs)
  "label": "SATURATED",     // UNSATURATED / SATURATED / UNSURE
  "level": "OVERLOADED",    // raw detector severity (diagnostic)
  "score": 0.87,            // raw detector score (diagnostic)
  "confidence": 1.0,        // detector confidence at this point
  "arrivals": 400,          // cumulative arrivals by this point
  "completions": 50         // cumulative completions by this point
}
```

`arrivals` and `completions` only ever increase. A widening gap between them is the physical picture of a backlog building. `level`/`score` expose the raw detector reading beneath the collapsed label.

## Where it works

`--saturation-interval` is supported on `blis run`, `blis replay`, and `blis observe`, and writes to the same `--saturation-report` file in each. `--saturation-interval` requires `--post-hoc-detector` (not `none`) and `--saturation-report`.

!!! note "Determinism & parity"
    The timeline is collected live during the run but is written only to the report file — never to stdout — so enabling it never changes the deterministic stdout metrics (INV-6). A trace exported with `--trace-output` and replayed with identical flags produces an identical timeline (INV-13).

## Live vs. real-time control

This timeline is an **analysis** tool: it observes and records, it does not change the simulation. It is distinct from BLIS's real-time saturation detector used for admission/flow control (`--flow-control --saturation-detector`), which acts on the run to shed or gate requests. See [Admission Control](admission.md).
