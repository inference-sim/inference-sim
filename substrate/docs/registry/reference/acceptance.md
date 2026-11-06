# Acceptance profiles

<p class="lede">How far speculative decoding advances a request per verification step, by
model, speculative method, K and thinking mode. Generated at build time from
<code>acceptance/</code>, at revision <!-- registry:ref -->; nothing below is typed in by
hand.</p>

An acceptance-profile set is the acceptance counterpart of a coefficient set. Its format,
and how a run resolves one value from it, are on the contracts'
[acceptance reference](../../contracts/reference/acceptance.md)
(blis-schemas#65). Each profile holds the mean **acceptance length** (AL): the tokens a
request gains per verification step, counting the target's bonus token, so it lies in
[1, K + 1]. No per-position rates are stored. A consumer that needs them derives vLLM
v0.31.0's minimum-variance schedule, which is how the InferenceX runs that force a
published AL realise it.

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
