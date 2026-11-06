# Engine rules

<p class="lede">An engine release's rules hold what that release accepts and what it refuses.
They live with the engine model that owns the release, and a validator passes them in for the
release a scenario's <code>engine_version</code> names (<code>Bundle.Rules</code>). This schema
registers none of its own.</p>

Rules run after the field layer passes and read the scenario, the deployment and the model
graph together. Each problem names the rule that reported it. A rule that reports an
*error* describes a deployment the engine would refuse at startup, or one a cost model
would price with the wrong constants. A rule that reports a *warning* describes one the
engine would run differently from what the document suggests, or a name the engine may
accept from an out-of-tree plugin.

A pool's `engine` block is opaque to this schema. The release's rules open it and check it
(`EngineRules.CheckEngineBlock`), in the field layer, at the block's paths below
`deployment.pools[i].engine`.

Three problems come from no release's rules:

`engine-rules-given`
:   *error.* The bundle has a scenario but no engine rules were given. Nothing checked its
    engine blocks or its version-specific rules, so it does not pass.

`engine-version-known`
:   *warning.* The rules given know no release of the scenario's `engine_version`. Field
    validation passed, but no version-specific rule ran.

`engine-version-matches`
:   *error.* The rules given answered for a release other than the scenario's. No rule was
    applied.

For vLLM the rules packs, their constants and their reference are blis-vllm's
(`rules/v0_*`).
