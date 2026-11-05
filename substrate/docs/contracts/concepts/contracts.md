# Contracts

<p class="lede">Each document type has its own keys, but all of them follow a few common
rules. Each rule exists to stop a document from looking correct while meaning something
its author did not intend.</p>

## A name has one source

A document's identity comes from one place:

- A chip, a fabric and a workload shape are named by their file: `hardware/h100.yaml` is
  the chip `h100`. The file has no `name` key, and one placed there is rejected.
- A storage class is named by its key in `devices/storage.yaml`.
- A model is named by its directory under `models/`. Its `graph.yaml` and `model.yaml`
  repeat the name in a `name` key, and `validate-catalog` fails if either differs from the
  directory.
- A coefficient set is referred to by its file name under `coefficients/`, and its `name`
  key matches it by convention.
- A scenario, a deployment and an evaluation run carry their name in a `name` key.

Two independent sources for one name would eventually disagree, and a reader could not
tell which was meant.

## A unit is part of the key

Where a number has a unit, the key the author types names it: `ttft_ms`,
`read_bandwidth_mb_s`, `base_latency_us`, `BwPeakTBs`. A unit stated only in a Go field
name or in a comment is invisible to whoever writes the YAML. An author who copies a
datasheet figure given in milliseconds into a field that expects microseconds would pass
every check and be wrong by a factor of a thousand.

There are three exceptions, each with its own reason:

- In Go, durations are `time.Duration`, which carries its unit.
- A coefficient's `value` has its unit in a separate `units` key, because a coefficient set
  holds quantities of many dimensions. `units` takes one of a fixed list of values.
- A trace header has an explicit `time_unit` key, because the timestamps live in a separate
  file and the tools that write traces spell the unit differently.

Adding a unit key beside a key that already names its unit, such as `ttft_unit: ms` next to
`ttft_ms`, is ruled out. It would be a second source that can disagree with the first.

## An unknown key is an error

Every loader decodes strictly. A key the type does not declare fails the load, and the
error names it. Otherwise misspelling `max_num_seqs` as `max_num_seq` would produce a
document that loads, validates and lacks the setting its author meant to make. Nothing
would report the mistake.

Two kinds of key are allowed without being declared:

- **Catalog comments.** The catalog annotates its numbers with prose. In chip, fabric and
  workload-shape files, and inside each storage class, a key that is exactly `_comment` or
  begins with `_comment_` is accepted and discarded. No other underscore-prefixed key is
  accepted, so a fitted factor cannot slip into a datasheet file disguised as a comment.
- **A redundant `name`.** Inside a storage class or a coefficient entry, a `name` key is
  accepted and then replaced by the class's key or the entry's map key. It cannot change
  the identity.

## Vocabularies are closed

An enumerated value must be one the schema lists. An unrecognized unit, evidence method,
attention kind or dtype is an error. Whether a coefficient was measured or assumed is
recorded in its `method`, and that record means something only if `method` takes one of a
fixed list of values. The lists are in [Vocabularies](../reference/vocabularies.md).

Names that belong to an engine release, such as backend names, cache dtypes and
quantization methods, are different. They change from release to release, so the release's
rules pack checks them, not the schema. See [Validation](validation.md#rule-problems).

## Numbers are finite

Every floating-point value must be finite. A `NaN` compares false to every bound and
`+Inf` looks positive, so either would pass a range check and corrupt every calculation
that used it. Validators reject both before they check a range.

## Zero means unset, except in tri-state keys

Most optional numeric keys treat zero as "not stated": a `block_size` of 0 means the engine
chooses. That works because no meaningful value of those keys is zero.

Where false or zero is a meaningful value distinct from silence, the key is tri-state.
`enable_prefix_caching` is the clearest case. vLLM enables prefix caching by default, so a
deployment that omits the key describes one *with* caching, and `enable_prefix_caching:
false` describes one without it. In Go these keys are pointers. In YAML, leaving the key out is
the third state.

## A request is not a resolution

Some deployment settings are requests the engine may override. A deployment records what
was asked for. A cost model reports what it resolved, and each request it overrode, in its
[`Resolution`](../reference/kernel.md#resolution). The reasoning is in
[Where the kernel stops](../design/kernel-boundary.md#the-request-and-the-resolution-are-different-values).
