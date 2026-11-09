# Golden equivalence: Result schema 1 to schema 2

The byte gate's goldens (`testdata/bytegate/golden`) changed in the commit that moved the
Result to schema 2 (blis-substrate `contracts/spec/result`). This page is the evidence that they
changed in shape only: no figure of the six runs moves. A golden file cannot show that, since
every byte of it differs; this does, by running the six cases in two builds and comparing figure by
figure through the rename table in [Metrics & Results](../guide/results.md#from-schema-1-to-schema-2).

## Rerun it

```bash
git worktree add /tmp/v1 1f1a3bb5          # the parent of the schema 2 commit: writes schema 1
(cd /tmp/v1 && go build -o /tmp/blis-v1 .)
go build -o /tmp/blis-v2 .                  # this commit: writes schema 2
SUB=$(go list -m -f '{{.Dir}}' github.com/inference-sim/blis-substrate)
python3 -I scripts/evidence/golden_equivalence.py run /tmp/blis-v1 /tmp/blis-v2 "$SUB" /tmp/eq
python3 -I scripts/evidence/golden_equivalence.py compare /tmp/eq
python3 -I scripts/evidence/golden_equivalence.py selftest /tmp/eq
```

Both builds read the same catalog and registry (the second pin's), at GOMAXPROCS 1.

## Result

```text
moe-closed-loop: 42 compared, 0 differ
moe-cluster: 42 compared, 0 differ
moe-prefix: 42 compared, 0 differ
moe-single: 42 compared, 0 differ
pd: 42 compared, 0 differ
spec-ep: 42 compared, 0 differ
total compared: 252 differ: 0
```

`selftest`, which shows the comparison is not vacuous: a change of 1e-9 to one metric and of one
to one preemption count are both reported.

```text
selftest: injected 1e-9 into ttft_p99_ms and +1 preemption -> ['engine', 'metrics.ttft_p99_ms']
```

## What is compared, and what is not

Compared, per case (42 figures): every schema 1 metric (`itl_*` as `tpot_*`, a number as the
figure's `value`); the step intervals (`engine.cluster.*` as `metrics.*`); goodput; every request
count (`injected`, `rejected` and the `still_*` counts as fates and drops); the window's duration
and request count; and the engine, router and kernel reports (`engine`, `router`, `kernel` as
`reports.*`).

Not compared, because schema 2 adds or replaces them: `schema_version`; the `engine`, `data`,
`selected_entries`, `frame` and `revisions` members (new or reshaped); the `preemptions` metric
(new, a whole-run total); and `runtime`.

The two builds also pin different blis-substrate and blis-vllm commits (817e83e and a8bf6ee for
schema 1, 4a3becf and 44fb7c5 for schema 2). That the figures agree across both pins is what
makes "no pin effect on these six runs" a checked claim. It says nothing about runs outside the six.
