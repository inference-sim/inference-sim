# Acceptance profiles

<p class="lede">How far speculative decoding advances a request per verification step, by
model, speculative method, K and thinking mode. Generated at build time from
<code>acceptance/</code>, at revision <!-- registry:ref -->; nothing below is typed in by
hand.</p>

An acceptance-profile set is the acceptance counterpart of a coefficient set. Its format,
and how a run resolves one value from it, are on the contracts'
[acceptance reference](../../contracts/reference/acceptance.md)
(https://github.com/inference-sim/blis-schemas/issues/65). Each profile holds the mean **acceptance length** (AL): the tokens a
request gains per verification step, counting the target's bonus token, so it lies in
[1, K + 1]. The InferenceX set stores no per-position rates: a consumer that needs them
derives vLLM v0.31.0's minimum-variance schedule, which is how the InferenceX runs that force
a published AL realise it. The internal-deployment set stores the rates its counters measured.

## `inferencex-golden-al`

SemiAnalysis InferenceX's golden SPEED-Bench curves, the values its AgentX runs force
through vLLM's `synthetic_acceptance_length`. A cell links to its profile in the file. A
dot (·) is a K with no profile, and a lookup for it is refused rather than interpolated.

<!-- registry:acceptance inferencex-golden-al values -->

### Where each curve was measured

One row per InferenceX file. The settings column lists every engine setting the file
states, including the `chat_template_kwargs` of each thinking mode.

<!-- registry:acceptance inferencex-golden-al provenance -->

### Caveats

The set's own `caveats`, as committed:

<!-- registry:acceptance inferencex-golden-al caveats -->

## How the set is made

`scripts/emit_acceptance_profiles.py` writes the file, and its `--check` is the drift
gate, as for the coefficient writers. The rules it follows are in
[methodology §12](../methodology.md#12-speculative-acceptance-profiles).

## `internal-deployment-measurements`

Acceptance measured on a running deployment rather than a benchmark's golden curve: for now
one profile, Muse-Spark-Open with its DSpark drafter on vLLM at K=7. Its per-position rates
are the engine's own counters, pooled over each sweep. Its caveats state
which K, thinking modes and categories it covers; a run outside them borrows or is refused,
as for any other set.

<!-- registry:acceptance internal-deployment-measurements values -->

### Where it was measured

<!-- registry:acceptance internal-deployment-measurements provenance -->

### Caveats

<!-- registry:acceptance internal-deployment-measurements caveats -->

`scripts/emit_internal_acceptance_profiles.py` writes the file from the counters committed in
`validator/testdata/internal/`, and its `--check` is the drift gate.
