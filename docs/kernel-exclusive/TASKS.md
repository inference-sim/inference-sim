# Per-attention-kind calibration: task list

Goal: give attention a per-kind law, fitted from NVIDIA's own measurement tables, and a
repeatable gate that checks any fitted form against those tables. Both were recommended after
the error decomposition showed the gap is SCATTER, not bias: BLIS and AISimulate carry
identical mean log-error (+0.0461 vs +0.0462) and differ only in spread (sd 0.1677 vs 0.1119).
Removing our bias perfectly would reach 12.65% against their 8.87%, so recalibrating the
existing single law cannot close it. Splitting the law per kind attacks the scatter.

## Why this ordering

Each task's output is the next task's input, and each has a check that fails loudly if the
task is wrong. No task depends on a later one.

## T1. Establish the ground truth mapping (blis-registry, read-only)

Determine, per attention kind we model, which of NVIDIA's parquet families and which column
filters correspond to it. Record row counts.

CORRECTNESS: the mapping is asserted by a script that reads the parquets and fails if a family
is absent, a filter column is missing, or a kind maps to zero rows. A kind with no data must be
reported as such, never silently fitted from the wrong family.

## T2. Fit per-kind decode attention (blis-registry)

`scripts/fit_attention_by_kind.py`, fitting the SAME form the registry already uses --
`floor + kv_bytes/rate` -- separately per kind, per part, on ONE kernel_source lane.

CORRECTNESS:
  - Re-fitting the `gqa` kind on the `attention` family with `window_size == 0` must reproduce
    the EXISTING committed h200 values (floor 11.0, rate 2.784e6) within the grid resolution.
    That is the regression check: a new script that cannot reproduce the old fit on the old
    data is wrong.
  - Every fit reports its point count and geometric error. A fit on fewer than 200 points is
    refused rather than emitted.
  - Lanes are never pooled. Pooling the MoE lanes gave 0.309 where the correct lane gave
    0.625; the same mistake on attention would corrupt every coefficient.

## T3. Emit coefficients under the existing convention (blis-registry)

Add `attention_decode_floor_<kind>` / `attention_decode_rate_<kind>` to
`cost-model-attention.yaml`, following the `recurrent_decode_*_<kind>` naming already in the
registry. NO schema change: `coefficient.Scope` has no kind axis and does not need one.

CORRECTNESS: `validator/validate.py` must pass, and the existing unsuffixed coefficients must
remain untouched so every deployment without a per-kind fit prices exactly as before.

## T4. Dispatch on attention kind in the kernel (blis-latency-kernel)

Mirror the recurrent loop in `new.go`: a `map[model.AttentionKind]string` suffix table,
`ValueOr(..., 0)`, skip-if-absent. In `kernel.go`, select the per-kind floor and rate when
present and fall back to the unsuffixed pair when not.

CORRECTNESS:
  - A behavioural test: two deployments identical except for attention kind must price
    DIFFERENTLY once per-kind coefficients exist, and identically when they do not.
  - `cmd/shape` must be unchanged for any model whose kind has no per-kind fit. That is the
    no-regression guarantee.
  - Mutation: deleting the dispatch must fail a named test.

## T5. Build the table-oracle gate (blis-registry)

`scripts/validate_against_aisimulate_tables.py`: for each fitted form, report the
geometric-mean ratio of our prediction to NVIDIA's measurements, per kind and per part, on the
regime the evaluation corpus actually runs.

CORRECTNESS: it must reproduce, as a known-good baseline, the two hand-measured findings from
this session -- decode attention at 1.294 on h200 GQA in the corpus regime, and the MoE
grouped-GEMM at 0.625 on b200 against the min-latency lane. A gate that cannot reproduce a
known result cannot be trusted on a new one.

## T6. Re-baseline and audit (worktree)

Re-run `cmd/kernelscore` on the vLLM subset under the apples-to-apples criterion already in
place. Report the before and after. State plainly whether SOTA was reached.

CORRECTNESS: the parity tests must still pass, including the reproduction of AISimulate's
published 10.05% from its own data. Any figure that moves must be explained by a specific
change, not by noise: the harness convergence criterion is fixed and the seed is fixed.

## Out of scope, deliberately

Wrapping AISimulate's Rust engine in the runtime path. Its `execute_json` runs their whole
replay -- scheduler, batching and tables -- so embedding it would mean reporting their number
as ours and the comparison would be vacuous. Their `cdylib` exposes no `extern "C"`, only
pyo3, so it would need a shim against an internal API with no stability contract, plus cgo and
per-platform binaries. Using their tables offline, as T2 and T5 do, gets the measurements
without the runtime dependency and keeps the kernel pure Go and portable -- which is the
property the design docs justify the closed form on.
