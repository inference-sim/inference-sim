# Measure how far the router policy moves a figure

<p class="lede">A router-table entry stands an llm-d policy in for a router BLIS does not host.
This guide shows how far BLIS's own prediction moves when the policy changes, which is the
number an entry's <code>bound</code> states. It gives the method, the inputs by commit and
hash, the commands, the expected output, the cost, and what the sweep does not show.</p>

## Contents

- [What the sweep answers](#what-the-sweep-answers)
- [The definition of a ratio](#the-definition-of-a-ratio)
- [Inputs](#inputs)
- [Reproduce it](#reproduce-it)
- [What it produced](#what-it-produced)
- [Where the bound is large](#where-the-bound-is-large)
- [The prefill-pool check](#the-prefill-pool-check)
- [Cost](#cost)
- [What it does not show](#what-it-does-not-show)
- [Where the data came from](#where-the-data-came-from)

## What the sweep answers

Design §11.3 scores a point measured behind a router BLIS does not host (Dynamo's, for the
entries here) as G2 when the [router table](../reference/router-table.md) maps the router to the
nearest llm-d policy. The mapping is a claim of nearness, and to a reader who ranks
configurations nearness is a number: if the choice of policy moves time to first token by 50 %
and time per output token by 1 %, the substitution is harmless for a ranking by the second and
not for the first.

The sweep holds the engine, the model, the hardware and the workload fixed, changes only the
router's policy, and records how far each figure moves. That is a **sensitivity of BLIS's
prediction to the policy**. It is not an error against Dynamo: no run behind Dynamo was compared
with a run of BLIS. It bounds what a wrong guess of the router's mode costs inside BLIS. It says
nothing about a difference between BLIS's rotation and Dynamo's that BLIS does not model; the
entry's `difference` lists those.

It is measured, not reasoned, because reasoning from the code was wrong once. The first version of
the table said, from llm-d's code, that round-robin would give a prefill pool of even size every
other endpoint, and so mapped no policy. [The prefill-pool check](#the-prefill-pool-check)
ran it and the prediction did not hold.

## The definition of a ratio

One run is one simulation of one **cell** (a pool shape, a workload kind and a number of
closed-loop clients), one policy and one seed. Three seeds (41, 42 and 43) are run for every
cell and policy. The *cell mean* of a figure is the mean of the three runs.

The **ratio** between two policies A and B in a cell is `max(mean_A / mean_B, mean_B / mean_A)`:
at least 1, and the same whichever policy is above. The **bound** of a regime is the largest ratio
over every cell of the regime and every alternative it names, rounded **up** to two decimals so
that it never understates. A **regime** is a workload kind, a range of concurrency, the model and
hardware, and the pool shapes; outside it the number says nothing. The catalog's tests recompute
every committed bound from the evidence file with code of their own (`contracts/spec/routertable`,
`TestTheCommittedBoundsAreTheirEvidences`), so an edited ratio or a regenerated file that the entry
does not follow fails.

Seed noise is part of the number. At 2P3D, balanced, 256 clients, median time to first token under
round-robin was 141.9, 141.5 and 163.0 ms (a coefficient of variation of 8 %), under the random
picker 150.3, 145.2 and 145.2 ms (2 %), under least-active-requests 101.5, 101.3 and 101.3 ms (0.1 %).
A ratio under about 1.10 between two load-blind policies is inside that noise.

## Inputs

- **The simulator.** [inference-sim](https://github.com/inference-sim/inference-sim) at commit
  `780f661ef9c18a2a9928acf8a63a67667cd597c0` (branch `m2/5-closeness-data-in-the-result`), built with the Go toolchain
  that inference-sim's `go.mod` names. Its `go.mod` pins blis-substrate at `v0.13.1-0.20261111044143-f70c6c7503a4` and replaces the
  router with `inference-sim/llm-d-router` `v1.0.0-blis.3`, commit `b1ab0ff7c0b84090e40b874f8873e78f2ae6897c`.
  The model, hardware and coefficients are the ones **embedded in that build** (Result
  `data.catalog.hash` `394a38a536b9a0802551fe614e3a5dec43516640dc5b8f403f4a86a0503ce2c6`,
  `data.registry.hash` `ac593a3d3e46c144861a47739343adb3dcfbcbf16fd1c1d87347768fdfd5b4f5`), not
  this working tree's.
- **The scenario** (`registry/docs/router-sensitivity/scenario.yaml`): MiniMax-M2.5 (`minimax-m2.5`),
  NVFP4 weights, FP8 KV cache, vLLM 0.32.0 on `gb200-nvl72` over `ib-400g`, block size 16,
  8,192 batched tokens, 256 sequences. A prefill pool of one-GPU instances (TP 1) and a decode pool
  of four-GPU instances (TP 4 with expert parallelism), joined by NIXL. Pool shapes `1P3D`, `2P3D`,
  `3P2D` and `4P1D`: n prefill and m decode instances.
- **The routers** (`registry/docs/router-sensitivity/router/*.yaml`): the llm-d profile pair that sends
  every request through a prefill pick and a decode pick, with the picker and scorers swapped.
  `round-robin`: the max-score picker, no scorer. `random-picker`: llm-d's random picker.
  `least-active-requests`: the active-request scorer. `queue-scorer`: the queue scorer. `default`:
  queue 2, KV-cache utilisation 2, prefix cache 3.
- **The workload kinds**, closed loop, fixed lengths, no shared prefix: `balanced` (1,024 prompt and
  1,024 output tokens), `prefill-bound` (8,192 and 16), `decode-bound` (256 and 2,048). Concurrency 16,
  64, 256 and 1,024; requests `min(4000, max(500, 5 x concurrency))`.

## Reproduce it

```sh
# 1. Build the simulator at the pinned commit.
git clone https://github.com/inference-sim/inference-sim && cd inference-sim
git checkout 780f661ef9c18a2a9928acf8a63a67667cd597c0
GOTOOLCHAIN=local go build -o /tmp/blis .

# 2. From a blis-substrate checkout, run the sweep (720 runs).
python3 -I registry/docs/router-sensitivity/sweep.py --blis /tmp/blis --work /tmp/router-sweep --out /tmp/runs.csv --jobs 3

# 3. Check it is the committed file, and that every cell has its three seeds.
shasum -a 256 /tmp/runs.csv        # f54dad99e89c657fb03e9484c2699303754bb2e2750dbffcaf9c670ae8cd1ec2
python3 -I registry/docs/router-sensitivity/summarise.py /tmp/runs.csv --check

# 4. Print the tables below, the spread, and the ratios block a catalog entry carries.
python3 -I registry/docs/router-sensitivity/summarise.py /tmp/runs.csv
python3 -I registry/docs/router-sensitivity/summarise.py /tmp/runs.csv --spread
python3 -I registry/docs/router-sensitivity/summarise.py /tmp/runs.csv --emit
```

The prefill-pool check is the same script with a narrower grid:

```sh
python3 -I registry/docs/router-sensitivity/sweep.py --blis /tmp/blis --work /tmp/router-alt --out /tmp/alternation.csv \
  --shapes 2P3D,4P1D --workloads prefill-bound --concurrency 1,2,4,8 --policies round-robin,least-active-requests
```

Expected: `runs.csv` has 720 rows (4 shapes x 3 workloads x 4 concurrencies x 5 policies x 3 seeds), each
with `completed` equal to `requests`, and the sha256 above. `alternation.csv` has 48 rows, two of which
(least-active-requests, 2P3D, 4 clients, seeds 42 and 43) have `completed` 498 of 500; see [the check](#the-prefill-pool-check).
The simulator is deterministic: rerunning the two round-robin and random-picker cells of 2P3D, balanced, 64
clients, seed 41 reproduced their rows exactly. The numbers were produced on an arm64 Mac; another platform may differ in the last digits.

A worked example. 2P3D, balanced, 256 clients, median time to first token: round-robin 141.9, 141.5 and 163.0 ms
(mean 148.8), least-active-requests 101.5, 101.3 and 101.3 ms (mean 101.4). The ratio is 148.8 / 101.4 = 1.47. Over
all four shapes the largest such ratio in the regime is 1.55, the figure in the entry's `balanced` row against
`default`, `least-active-requests` and `queue-scorer`.

## What it produced

Each table is one comparison: the ratio between `round-robin` and one alternative, the largest over the four pool
shapes, at each concurrency. The catalog entries state the largest over the four concurrencies, per workload kind, for the
four figures shown; the sweep also records `e2e_p50_ms` and `request_throughput`.

**Against the random picker** (the other policy that ignores load):

| Figure | Workload | 16 | 64 | 256 | 1024 |
|---|---|---:|---:|---:|---:|
| `ttft_p50_ms` | balanced | 1.01 | 1.07 | 1.06 | 1.01 |
| `ttft_p50_ms` | prefill-bound | 1.04 | 1.13 | 1.04 | 1.01 |
| `ttft_p50_ms` | decode-bound | 1.01 | 1.09 | 1.08 | 1.02 |
| `ttft_p99_ms` | balanced | 1.11 | 1.07 | 1.04 | 1.01 |
| `ttft_p99_ms` | prefill-bound | 1.01 | 1.19 | 1.10 | 1.01 |
| `ttft_p99_ms` | decode-bound | 1.06 | 1.07 | 1.04 | 1.10 |
| `tpot_p50_ms` | balanced | 1.01 | 1.01 | 1.01 | 1.01 |
| `tpot_p50_ms` | prefill-bound | 1.01 | 1.01 | 1.01 | 1.01 |
| `tpot_p50_ms` | decode-bound | 1.01 | 1.01 | 1.01 | 1.01 |
| `output_tok_s` | balanced | 1.01 | 1.01 | 1.01 | 1.01 |
| `output_tok_s` | prefill-bound | 1.02 | 1.04 | 1.04 | 1.01 |
| `output_tok_s` | decode-bound | 1.01 | 1.01 | 1.01 | 1.01 |

**Against `least-active-requests`** (the router's own count of in-flight requests):

| Figure | Workload | 16 | 64 | 256 | 1024 |
|---|---|---:|---:|---:|---:|
| `ttft_p50_ms` | balanced | 1.02 | 1.13 | 1.55 | 1.07 |
| `ttft_p50_ms` | prefill-bound | 1.18 | 1.18 | 1.07 | 1.03 |
| `ttft_p50_ms` | decode-bound | 1.03 | 1.12 | 1.41 | 1.08 |
| `ttft_p99_ms` | balanced | 1.01 | 1.05 | 1.04 | 1.01 |
| `ttft_p99_ms` | prefill-bound | 1.48 | 1.65 | 1.32 | 1.12 |
| `ttft_p99_ms` | decode-bound | 1.03 | 1.05 | 1.04 | 1.26 |
| `tpot_p50_ms` | balanced | 1.02 | 1.01 | 1.01 | 1.01 |
| `tpot_p50_ms` | prefill-bound | 1.03 | 1.04 | 1.04 | 1.04 |
| `tpot_p50_ms` | decode-bound | 1.02 | 1.01 | 1.01 | 1.01 |
| `output_tok_s` | balanced | 1.01 | 1.01 | 1.02 | 1.01 |
| `output_tok_s` | prefill-bound | 1.16 | 1.07 | 1.06 | 1.03 |
| `output_tok_s` | decode-bound | 1.01 | 1.02 | 1.01 | 1.02 |

**Against `default`** (queue depth, KV-cache utilisation, prefix cache):

| Figure | Workload | 16 | 64 | 256 | 1024 |
|---|---|---:|---:|---:|---:|
| `ttft_p50_ms` | balanced | 1.06 | 1.47 | 1.21 | 1.19 |
| `ttft_p50_ms` | prefill-bound | 1.13 | 1.18 | 1.07 | 1.02 |
| `ttft_p50_ms` | decode-bound | 1.01 | 1.18 | 1.28 | 1.15 |
| `ttft_p99_ms` | balanced | 1.05 | 1.00 | 1.00 | 1.05 |
| `ttft_p99_ms` | prefill-bound | 1.03 | 1.34 | 1.13 | 1.05 |
| `ttft_p99_ms` | decode-bound | 1.02 | 1.00 | 1.00 | 1.12 |
| `tpot_p50_ms` | balanced | 1.01 | 1.02 | 1.01 | 1.09 |
| `tpot_p50_ms` | prefill-bound | 1.03 | 1.04 | 1.02 | 1.01 |
| `tpot_p50_ms` | decode-bound | 1.05 | 1.06 | 1.01 | 1.01 |
| `output_tok_s` | balanced | 1.01 | 1.01 | 1.01 | 1.12 |
| `output_tok_s` | prefill-bound | 1.03 | 1.07 | 1.05 | 1.03 |
| `output_tok_s` | decode-bound | 1.02 | 1.03 | 1.01 | 1.07 |

**Against `queue-scorer`:**

| Figure | Workload | 16 | 64 | 256 | 1024 |
|---|---|---:|---:|---:|---:|
| `ttft_p50_ms` | balanced | 1.01 | 1.04 | 1.04 | 1.01 |
| `ttft_p50_ms` | prefill-bound | 1.14 | 1.18 | 1.07 | 1.02 |
| `ttft_p50_ms` | decode-bound | 1.01 | 1.05 | 1.03 | 1.05 |
| `ttft_p99_ms` | balanced | 1.00 | 1.00 | 1.00 | 1.01 |
| `ttft_p99_ms` | prefill-bound | 1.10 | 1.34 | 1.13 | 1.05 |
| `ttft_p99_ms` | decode-bound | 1.00 | 1.00 | 1.00 | 1.25 |
| `tpot_p50_ms` | balanced | 1.01 | 1.01 | 1.01 | 1.01 |
| `tpot_p50_ms` | prefill-bound | 1.01 | 1.01 | 1.01 | 1.01 |
| `tpot_p50_ms` | decode-bound | 1.01 | 1.01 | 1.01 | 1.01 |
| `output_tok_s` | balanced | 1.01 | 1.01 | 1.01 | 1.01 |
| `output_tok_s` | prefill-bound | 1.06 | 1.07 | 1.05 | 1.03 |
| `output_tok_s` | decode-bound | 1.01 | 1.01 | 1.01 | 1.01 |

The spread across all five policies in each cell (largest over smallest; the smallest and largest over
the four pool shapes), the measure the first scoping run used. For `balanced` it is the same
shape of result that run found (median time to first token 1.10 to 1.46 at 64 clients and 1.53 to 1.70 at 256, against its 1.11 to 1.74 and 1.50 to 1.70;
time per output token at most 1.09 against 1.08; throughput at most 1.12 against 1.07, at 1,024 clients):

| Figure | Workload | 16 | 64 | 256 | 1024 |
|---|---|---:|---:|---:|---:|
| `ttft_p50_ms` | balanced | 1.01 to 1.06 | 1.10 to 1.46 | 1.53 to 1.70 | 1.00 to 1.27 |
| `ttft_p50_ms` | prefill-bound | 1.00 to 1.21 | 1.00 to 1.18 | 1.00 to 1.08 | 1.00 to 1.03 |
| `ttft_p50_ms` | decode-bound | 1.00 to 1.03 | 1.01 to 1.31 | 1.46 to 1.78 | 1.00 to 1.14 |
| `ttft_p99_ms` | balanced | 1.00 to 1.10 | 1.00 to 1.08 | 1.00 to 1.07 | 1.00 to 1.05 |
| `ttft_p99_ms` | prefill-bound | 1.00 to 1.52 | 1.00 to 1.69 | 1.00 to 1.34 | 1.00 to 1.12 |
| `ttft_p99_ms` | decode-bound | 1.00 to 1.09 | 1.00 to 1.08 | 1.00 to 1.07 | 1.00 to 1.41 |
| `tpot_p50_ms` | balanced | 1.00 to 1.02 | 1.00 to 1.01 | 1.00 to 1.01 | 1.00 to 1.09 |
| `tpot_p50_ms` | prefill-bound | 1.01 to 1.03 | 1.00 to 1.04 | 1.00 to 1.04 | 1.00 to 1.04 |
| `tpot_p50_ms` | decode-bound | 1.00 to 1.05 | 1.00 to 1.06 | 1.00 to 1.01 | 1.00 to 1.01 |
| `output_tok_s` | balanced | 1.00 to 1.01 | 1.00 to 1.02 | 1.01 to 1.01 | 1.00 to 1.12 |
| `output_tok_s` | prefill-bound | 1.00 to 1.17 | 1.00 to 1.07 | 1.00 to 1.05 | 1.00 to 1.03 |
| `output_tok_s` | decode-bound | 1.00 to 1.02 | 1.00 to 1.04 | 1.00 to 1.01 | 1.00 to 1.07 |

## Where the bound is large

The bound is **not** small for every figure. It holds tightly for time per output token (at most 1.09 against any
alternative in any workload swept) and, away from the extremes of concurrency, for throughput. It does not hold for:

- **Median time to first token at 64 to 256 clients, against a policy that reads load.** Round-robin is up to
  1.55 times `least-active-requests` for `balanced`, 1.41 for `decode-bound`, and 1.47 times `default` at 64
  clients. This is where the pools approach saturation: a router that reads its own counts keeps a short queue where a
  rotation does not. A ranking by TTFT p50 in that regime should not rest on the substitution.
- **p99 time to first token for the `prefill-bound` workload**, up to 1.65 against `least-active-requests` at 64 clients, 1.34 against `queue-scorer` and `default`.
- **Throughput of the `prefill-bound` workload at 16 clients** (1.16 against `least-active-requests`; the load-blind policies and `default` share
  the gap, so it is arrival imbalance across four prefill endpoints and 16 clients), and **at 1,024 clients for
  `balanced`** (1.12 against `default`).

The earlier scoping run's summary (time per output token at most 1.08, throughput at most 1.07, p99 time to first
token at most 1.13) holds for `balanced` to within 0.05 and not for the other two kinds. The table of
ratios is the whole statement; a single "at most" would hide the cells above.

## The prefill-pool check

llm-d's max-score picker draws one counter at every pick of every profile (the entry's `counter-per-pool`
difference), so a request that picks a prefill endpoint and then a decode endpoint draws two successive values. Read
literally, that gives a prefill pool of even size only every other endpoint. The first version of the table
mapped no policy for that reason.

The check uses a workload that saturates the prefill pool (`prefill-bound`), where a pool served through one endpoint
would give the one-endpoint rate. In requests per second, round-robin:

| Prefill endpoints | 16 clients | 64 | 256 | 1,024 | `least-active-requests` |
|---|---:|---:|---:|---:|---|
| 1 (`1P3D`) | 13.6 | 13.6 | 13.6 | 13.6 | 13.6 |
| 2 (`2P3D`) | 27.0 | 25.3 | 26.7 | 26.8 | 27.1 to 27.3 |
| 4 (`4P1D`) | 46.3 | 50.2 | 51.6 | 53.3 | 53.6 to 54.4 |

Two endpoints serve about twice the rate of one; a pool that used one would give 13.6. The prediction was not
reproduced. At 1 to 8 clients (`alternation.csv`, 2P3D) round-robin served 4.66, 9.30, 16.84 and 24.35 requests per second at 1, 2, 4 and 8 clients.
`TestTheEvidenceDoesNotShowAStarvedPrefillPool` pins the comparison.

Why it does not show is **not known**. BLIS's Result lists decode endpoints in `TargetDistribution` and not prefill
endpoints, so a prefill pick cannot be read back; recording both pools is a change to the simulator. Concurrent requests
interleave their picks, which breaks the strict alternation the reading assumes, and that is a guess.

One result of the same check is **unexplained and not about round-robin**. With `least-active-requests`, two prefill
endpoints and 4 clients, two of 500 requests did not complete in seeds 42 and 43 (`timed_out`, requests 55 and 56 of session users 3 and 2, round 14) and the rate was 12.7
requests per second against 16.8 for round-robin. It is in `alternation.csv`; `runs.csv` has no run with a request that did not complete. To reproduce:
`blis run --scenarios <work>/scn --scenario s_2P3D_least-active-requests.yaml --num-instances 5 --prefill-instances 2 --decode-instances 3 --concurrency 4 --num-requests 500 --seed 42 --prompt-tokens 8192 --prompt-tokens-min 8192 --prompt-tokens-max 8192 --prompt-tokens-stdev 0 --output-tokens 16 --output-tokens-min 16 --output-tokens-max 16 --output-tokens-stdev 0`.

## Cost

The 720 runs took about 39 minutes of wall-clock time at 4 concurrent processes on a 10-core arm64 Mac that other jobs were
also using (load average above 100), one run finished every 3.3 seconds on average (each took about 13 seconds of one process); a 60-run trial of one shape took 191 seconds. Peak memory was not measured. Each
run writes a Result of 0.6 to 2 MB that the script deletes after reading six figures from it, so the scratch directory stays small. The
committed evidence is `runs.csv` (98 KB) and `alternation.csv` (6 KB). Run it with `--jobs 3` or fewer on a shared machine.

## What it does not show

- **One model, one part, one engine.** MiniMax-M2.5 on GB200 with vLLM 0.32.0 over NIXL. Another model or part moves where the pools
  saturate, and with it where the large ratios sit.
- **Closed-loop clients, fixed lengths, no shared prefix.** Nothing here has sessions of several turns, a spread of lengths, an
  open-loop arrival rate, or requests that share a prefix. The `kv` mode of Dynamo exists for shared prefixes, which is why its entry maps no policy.
- **Four concurrencies and four pool shapes.** Larger pools and other parallelism, and the points between 16, 64, 256 and 1,024, were not run.
- **A sensitivity, not an error.** No run behind Dynamo was compared with BLIS. The ratios say how far BLIS moves when its policy changes.
- **Three seeds.** A ratio between two load-blind policies below about 1.10 is inside seed noise.
- **In-sample.** The entries' bounds are read from the same runs that motivated the mapping. No held-out set exists for them.

## Where the data came from

Every figure is produced by the simulator named above from the scenario and routers in this directory. No third-party data is used or committed.
The Dynamo code the entries cite is read at the commits the entries name, from [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
(Apache-2.0); llm-d's router code is read from [inference-sim/llm-d-router](https://github.com/inference-sim/llm-d-router) at `b1ab0ff7c0b84090e40b874f8873e78f2ae6897c`
(Apache-2.0). Which recipes run which Dynamo mode comes from the dashboard's own reading of the InferenceX recipes, not from this sweep.
