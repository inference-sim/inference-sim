# scripts/

Reproducible analysis scripts for BLIS. Each script runs `blis` end-to-end and
emits a single CSV summary alongside per-run raw outputs, so anyone can
re-validate a published claim without manually reconstructing the command set.

## archon-build.sh — clone and build the pinned archon-go binary

Clones `archon-go` at a pinned version and builds it, so the `archon.yml` and
`deliver-verify.yml` workflows can obtain the binary the same way (`ARCHON_BIN=$(scripts/archon-build.sh)`).

```bash
# Use the version pinned in .archon-version at the repo root
ARCHON_BIN=$(scripts/archon-build.sh)

# Or override with an explicit version (git branch/tag)
ARCHON_BIN=$(scripts/archon-build.sh v1.2.3)
```

| Argument | Meaning |
|---|---|
| `[version]` | optional git branch/tag to build; when omitted, falls back to `.archon-version` at the repo root |

The built binary path is printed on **stdout**; every diagnostic (`Building…`, the `git clone`
output, `Built: …`) goes to **stderr**. That split is deliberate: it is what lets a caller do
`ARCHON_BIN=$(scripts/archon-build.sh)` and capture only the path. Mixing a diagnostic into
stdout would silently corrupt `ARCHON_BIN`, so keep new output on the correct stream.

Exits non-zero when no version can be resolved — no argument given **and** no `.archon-version`
present — rather than building an arbitrary default. Requires `git` and `go` (1.26+).

## archon-review.sh — `/archon-pr-review` review step

Runs `archon-go pr-review` and composes the PR comment. Called by
`.github/workflows/archon.yml`; it lives here rather than inline in the workflow so its
behaviour can be tested (`scripts/archon_review_test.go` drives it against a stub
`archon-go` and throwaway git repositories).

```bash
ARCHON_BIN=... BASE_SHA=... HEAD_SHA=... DECL_FILE=... OUTPUT_FILE=... RUN_URL=... \
  scripts/archon-review.sh
```

| Variable | Meaning |
|---|---|
| `ARCHON_BIN` | archon-go binary, from `archon-build.sh` |
| `BASE_SHA` | the PR's base branch tip (where the plan is looked for first) |
| `HEAD_SHA` | the PR's head commit, already fetched into the object store |
| `DECL_FILE` | file holding plan-declaration candidate text (PR body, then closing-issue bodies) |
| `OUTPUT_FILE` | where the comment body is written; truncated up front |
| `RUN_URL` | workflow run URL, used in the truncation notice |
| `GITHUB_STEP_SUMMARY` | optional; the untruncated body is appended when set |

Exits 0 for every reviewed outcome, including archon failure, so the caller always has a
body to post. Exit 2 only on a usage or environment error. Adds `--plan` to the single
`archon-go` invocation when a plan resolves; retries once without it if that invocation
fails, so a bad plan file never costs the three views.

## deliver-gate.sh — the L1 delivery loop's decision

Decides what happens next in a delivery round: label the PR ready, run another correction, or
stop for a human. Called by `.github/workflows/deliver-verify.yml`; it lives here rather than
inline in the workflow because it is the one place in the loop where a bug could mark broken
code ready to merge, so it is the one place that needs tests
(`scripts/deliver_gate_test.go`). See
[docs/contributing/automated-delivery.md](../docs/contributing/automated-delivery.md).

```bash
CI_STATUS=success PLAN_GATE=pass AGENT_VERDICT=GREEN ROUND=0 MAX_ROUNDS=3 \
  scripts/deliver-gate.sh
# decision=ready
# reason=CI passed, plan signal 'pass', and the review returned GREEN
```

| Variable | Domain | Meaning |
|---|---|---|
| `CI_STATUS` | `success` \| `failure` \| `unknown` | verify runs `go build`, `go test` and `golangci-lint` against the PR tree; anything but all three passing is `failure`, and an unread signal is `unknown` |
| `PLAN_GATE` | `pass` \| `regression` \| `conflicts` \| `unverified` \| `absent` | from `.archon/review.json`; `absent` (no plan claimed) delivers exactly as `pass`, while `unverified` (a plan declared but never checked) blocks — `archon-review.sh` exits 0 after falling back to a plan-less review, so the two must not be conflated |
| `AGENT_VERDICT` | `GREEN` \| `NOT-GREEN` \| `MISSING` | the review comment's `DELIVER-VERDICT:` marker |
| `ROUND` | integer | correction rounds already spent, read from the `deliver:round-N` label |
| `MAX_ROUNDS` | integer | cap before stopping for a human |

Prints `decision=ready|correct|needs-human` and a one-line `reason`, exiting 0. Exit 2 only on a
wiring error — an unset input or a non-integer counter — so a misconfigured workflow fails
loudly instead of receiving a verdict. A value outside a declared domain is different: it takes
a catch-all and returns `needs-human`, because GitHub has eight check conclusions rather than
two and an unmapped one must not fall through with no decision at all.

`ready` requires the checks passing, a non-regressing (or absent) plan signal, **and** an explicit GREEN. A GREEN that contradicts an objective signal returns `needs-human` naming the
disagreement — never `ready`, at any round.

## archon-plan-resolve.sh — find and extract a declared archon plan

Finds the first `archon-plan: <path>` line in the declaration text and extracts that file
out of git. Tested by `scripts/archon_plan_resolve_test.go`.

```bash
scripts/archon-plan-resolve.sh <base-ref> <head-ref> <decl-file> <out-file>
```

Prints one `key=value` per line and exits 0 in all three cases (exit 2 only on a usage error):

| Output | Meaning |
|---|---|
| `status=none` | no declaration found — the normal case for a bug fix |
| `status=resolved` + `plan_path`, `plan_source` (`base`/`head`), `plan_commit` | `<out-file>` written |
| `status=error` + `plan_path`, `message` | declared but unusable |

The declared path comes from a PR or issue body, so it is untrusted. It must be
repository-relative, end in `.json`, be at most 256 characters, and match
`[A-Za-z0-9._/-]+` with no `..`; anything echoed back has out-of-allowlist characters
replaced, so a rejected path cannot inject markdown or a workflow command into the comment.

Extraction uses git plumbing against an explicit commit — never the filesystem. The
workflow is `issue_comment`-triggered, so the working tree holds the default branch rather
than the PR: a filesystem read would find the wrong file and would let a traversal escape
the repository. The object's type, mode, and size are gated (regular-file blob, non-empty,
at most 1 MiB) before any bytes are written.

Base is tried before head so a hole PR cannot be graded against a plan it rewrote. A base
copy that exists but is unusable, and a commit that is not reachable locally, are both hard
errors rather than a fall-through to the head copy — `git ls-tree` is silent for both, and
either one would quietly hand grading back to the PR.

## find-saturation.sh — Rate-sweep saturation finder

Drives `blis run` across a configurable rate sweep against a chosen
`(model, hardware, TP, workload)` configuration. For each rate it:

1. Runs `blis run` once with the **detector bank** (`--detectors all`), which
   fans one deterministic replay out to every post-hoc detector (composite,
   threshold, backlog-drift, peak-rate) in a single pass (#1519, #1614).
2. Reads each detector's final verdict from the run's `--saturation-report`
   (the `"final"` detector→label map, #1517).
3. Extracts throughput, latency, and all detector verdicts into a single CSV row.

The output reproduces the validation table against the bundled
`model_configs/llama-3.1-70b-instruct/`. Pointing it at any other configuration
should produce a comparable table with the same column shape.

### Quick start

```bash
# Default: Llama-3.1-70B / TP=8 / H100 / chatbot, sweeps 0.5..100 req/s
./scripts/find-saturation.sh

# Llama-2-7B / TP=1, narrower sweep
MODEL=meta-llama/Llama-2-7b-hf \
  MODEL_CONFIG_FOLDER=model_configs/llama-2-7b-hf \
  TP=1 RATES="2 4 6 8 10 12 16 20" \
  ./scripts/find-saturation.sh

# Custom workload, slower coarse sweep
WORKLOAD=summarization NUM_REQUESTS=2000 RATES="4 6 8 10 12" \
  ./scripts/find-saturation.sh

# Only one detector (skip the bank)
DETECTORS=composite ./scripts/find-saturation.sh

# No bundled config — let blis fetch from HuggingFace
MODEL=qwen/qwen3-14b MODEL_CONFIG_FOLDER="" TP=1 \
  ./scripts/find-saturation.sh
```

### Inputs (all environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `meta-llama/Llama-3.1-70B-Instruct` | HuggingFace-style model name |
| `MODEL_CONFIG_FOLDER` | `model_configs/llama-3.1-70b-instruct` | Path to bundled `config.json`; set to `""` to force HF auto-fetch |
| `HARDWARE` | `H100` | GPU type passed to `--hardware` |
| `TP` | `8` | Tensor parallelism degree |
| `WORKLOAD` | `chatbot` | Built-in preset (chatbot/summarization/contentgen/multidoc) |
| `LATENCY_MODEL` | `trained-physics` | `--latency-model` backend |
| `NUM_REQUESTS` | `6000` | `--num-requests` per rate |
| `HORIZON_US` | `600000000` (600s) | `--horizon` per rate |
| `DETECTORS` | `all` | `--detectors` selection (`all`, or a comma-list like `composite,threshold`) |
| `FINAL_WINDOW` | `10s` | `--saturation-final-window` (trailing window for the plurality vote) |
| `RATES` | `0.5 1 2 4 6 8 10 12 14 16 20 30 40 50 60 80 100` | Space-separated rate sweep |
| `SEED` | `42` | RNG seed |
| `OUT_DIR` | `results/saturation-<ts>-<pid>` | Output directory |

### Outputs

```
$OUT_DIR/
├── summary.csv                          # one row per rate (12 columns)
├── rate-{R}.json                        # blis run stdout (metrics)
├── rate-{R}.stderr                      # blis run stderr (progress logs)
└── rate-{R}.saturation.json             # {"final":{...},"trace":[...]} report
```

`summary.csv` columns:

| Column | Source | Meaning |
|---|---|---|
| `intended_rate` | input flag | What `--rate` was set to |
| `sustained_throughput` | `injected_requests / vllm_estimated_duration_s` | Actual req/s injected over total sim time |
| `goodput_rps` | `responses_per_sec` | Completed req/s |
| `goodput_vs_intended` | `goodput_rps / intended_rate` | Ratio; <100% indicates the engine couldn't sustain intended load |
| `timeout_frac` | `timed_out_requests / injected_requests` | Fraction culled by client timeout |
| `e2e_p99_ms` / `ttft_p99_ms` | metrics | Tail latencies |
| `still_queued` / `still_running` | metrics | End-state residue |
| `composite_verdict` | report `.final.composite` | STABLE / BACKLOGGED / OVERLOADED |
| `threshold_verdict` | report `.final.threshold` | STABLE / OVERLOADED (binary) |
| `backlog_drift_verdict` | report `.final["backlog-drift"]` | STABLE / BACKLOGGED / OVERLOADED |
| `peak_rate_verdict` | report `.final["peak-rate"]` | STABLE / BACKLOGGED / OVERLOADED |

A detector that is not in `DETECTORS` shows `n/a` in its column.

### Reading the output

A clean read of "where does this configuration saturate?" looks like:

```
intended_rate  goodput_rps  ratio  composite     threshold   backlog-drift  peak-rate
0.5            0.50         100%   STABLE        STABLE      STABLE
…
60             56.29        94%    BACKLOGGED    STABLE      BACKLOGGED
80             64.49        81%    OVERLOADED    OVERLOADED  OVERLOADED   ← knee
100            64.76        65%    OVERLOADED    OVERLOADED  OVERLOADED
```

The saturation knee is the first rate where `ratio` falls below ~100% OR a
detector's final verdict flips to `OVERLOADED`. The detectors measure
different things — composite blends rate deficit with a latency trend,
threshold is a pure mean-E2E cutoff, and backlog-drift tracks the slope of
in-flight — so a rate where they disagree (e.g. backlog-drift flags
`BACKLOGGED` while threshold is still `STABLE`) is itself informative: the queue
is growing before mean latency crosses the cutoff.

### Tips

- **Build once.** The script auto-builds `./blis` if absent. Subsequent runs
  reuse it; remove the binary to force a rebuild.
- **Pin the seed.** Two seeds at the same rate land in different parts of
  Poisson variance and look noisy. Default `SEED=42` keeps every step of the
  sweep on the same noise realization.
- **Don't trust a single rate.** Saturation curves are smoother than they look;
  the knee is a transition zone (~3-5 rate steps wide). Look at three rates
  before and after the suspected knee.
- **Fine sweep after coarse.** First pass with the default 17 rates spanning
  200×; identify the knee zone (e.g., between 60 and 80); then re-run with
  `RATES="62 64 66 68 70 72 74 76 78"` to pin the exact transition.
- **Tune the final window.** `FINAL_WINDOW` controls how much of the run's tail
  the plurality vote considers. Use `--horizon` to end observation while load is
  still active — if you let all requests drain, the tail looks STABLE.

### Running on a custom configuration

The script's defaults match the reference validation experiment so anyone can
reproduce that exact table. To validate any other configuration, override the
relevant variables:

```bash
# Example: probe Mixtral-8x7B FP8 on 4×H100 TP=4 with summarization workload
MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 \
  MODEL_CONFIG_FOLDER=model_configs/mixtral-8x7b-instruct \
  TP=4 WORKLOAD=summarization \
  RATES="2 4 8 16 24 32 40 48" \
  ./scripts/find-saturation.sh
```

If your model isn't in `model_configs/`, either:
- Drop a `config.json` into `model_configs/<your-model-slug>/` and point
  `MODEL_CONFIG_FOLDER` at it, or
- Set `MODEL_CONFIG_FOLDER=""` to let `blis` fetch from HuggingFace at startup.

### Dependencies

- `bash` 4+
- `jq` (for JSON parsing)
- `bc` (for ratio arithmetic)
- `column` (for the final pretty-print; falls back gracefully if missing)
- `go` (auto-builds `./blis` on first run)
