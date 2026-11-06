# Validate documents from Go

<p class="lede">Load each document with its loader, put the documents in a bundle, and call
<code>Validate</code>. The report keeps the field layer apart from the rule layer and says
which engine release's rules ran.</p>

```sh
go get github.com/inference-sim/blis-substrate@<version>
```

The snippets on this page are the package's example tests. They run under `go test`, and
the output shown is checked against what they print.

## Load, then validate

```go
--8<-- "contracts/example_test.go:validate"
```

```text title="Output"
ok: true rules applied: 0.29.0
```

`must` stands for whatever error handling the caller uses. Loading and validating are
separate calls. A load error means the file is not a document of that kind at all: it does
not parse, or it has a key the type does not declare (`LoadBundle` also refuses a file that
is not a scenario plus a deployment with at least one pool). A validation problem means the
document parsed and says something wrong. They need different fixes, so they arrive
through different channels.

### Loaders

| Loader | Reads | The document's name comes from |
|---|---|---|
| `LoadScenario` | a scenario | its `name` key |
| `LoadDeployment` | a deployment | its `name` key |
| `LoadBundle` | a scenario and its deployment from one file, as two YAML documents separated by `---` | each document's `name` key |
| `LoadModelGraph` | `models/<name>/graph.yaml` | its `name` key |
| `LoadModelIdentity` | `models/<name>/model.yaml` | its `name` key |
| `LoadChip` | `hardware/<name>.yaml` | the file name |
| `LoadFabric` | `networks/<name>.yaml` | the file name |
| `LoadStorageDevices` | `devices/storage.yaml` | each class's key; returned sorted by name |
| `LoadWorkload` | `workloads/<name>.yaml` | the file name |
| `LoadCoefficientSet` | a coefficient set | its `name` key (a scenario refers to it by file name) |
| `LoadEvaluationRun` | an evaluation run | its `name` key |
| `LoadMeasuredRun` | `measurements/<license_ref>/<name>.yaml` | its `name` key |
| `LoadMeasurementSources` | `measurements/sources.yaml` | each source's `id` |

Each decodes one YAML document and rejects keys its type does not declare, except
`LoadBundle`, which decodes two (a scenario, then its deployment) with the same strictness. The loaders do
not compare a `name` key with the directory or file it sits in; `validate-catalog` does that
for model entries and measured runs. A measured run is validated with
`ValidateWithSources(registry)`, which also computes its verdict; it is not part of a
bundle.

### The bundle

A `Bundle` holds any subset of documents: a chip alone, or everything a run needs. Every
document present is checked on its own. A deployment is also checked against its
scenario's cluster, which is why a bundle with a deployment and no scenario is reported as
a problem rather than checked by halves. Rules run when the bundle has a scenario and the
field layer passed; they read the deployment and the model graph when those are present.

## Read the report

```go
--8<-- "contracts/example_test.go:problems"
```

```text title="Output"
error: deployment.pools[0].engine.gpu_memory_utilization: must lie in [0, 1] (0 means unset), got 1.5
error: deployment.pools[0].parallel: needs 32 GPUs (pp 1 x tp 8 x pcp 1 x dp 4, one per rank) but the pool's 1 node(s) of 8 GPUs provide 8
error: deployment.pools[0].parallel.dcp: decode-context parallelism reuses the tensor-parallel ranks when prefill-context parallelism is off, so tp 8 must be divisible by dcp 3
```

All three mistakes are reported at once, sorted by path. Each `Problem` has `Path`,
`Message`, `Severity` (`error` or `warning`) and `Rule`, which is empty for a field
problem. `rep.OK()` is false here because errors are present; warnings alone leave it true.

To treat the layers differently, read `rep.Field` and `rep.Rule`. Each has `OK()`, `All()`
for every problem and `Errors()` for errors only.

```go
--8<-- "contracts/example_test.go:rules"
```

```text title="Output"
field layer ok: true
error [enum-values-known]: : pools[0].engine.cache_dtype: "fp8_e4m3fn" is not accepted by engine 0.29.0
```

A rule problem carries its rule's name in brackets. Rules state the location inside the
message, so the path between the two colons is empty.

If the field layer passed, the bundle has a scenario, and `rep.RulesApplied` is still
empty, then the rules given know no release of the scenario's `engine_version`. The rule layer then
holds a single warning saying so.

## Resolve names against a catalog

`Validate` does not read the disk, so it cannot tell whether `cluster.hardware` names a
chip that exists. `ValidateAgainstCatalog` can, given a catalog root and a registry root:

```go
--8<-- "contracts/example_test.go:resolve"
```

```text title="Output, with the repository's catalog/"
error: references.hardware: "nvidia-h100": looked for hardware/nvidia-h100.yaml; the catalog has [a100-80 a100-sxm b200 b300 gb200-nvl72 gb300 h100 h200 l40s]
```

It checks only that each name resolves. To validate what the names point to, load those
files and pass them to `Validate`.

## Pass the engine's own rules

The contracts hold no engine release's rules. Each engine model owns the rules for the
release it models; for vLLM they are the packs of `github.com/inference-sim/blis-vllm/rules`.
A caller passes them in: `Bundle.Rules` returns a release's rules as a `rules.EngineRules`,
and `Bundle.RulesVersions`, which is optional, lists the releases it knows, for the finding a
scenario naming another one gets.

```go
b.Rules = func(v string) rules.EngineRules {
	if p := vllmrules.Lookup(v); p != nil { // github.com/inference-sim/blis-vllm/rules
		return p
	}
	return nil // a typed nil reads as unknown too
}
b.RulesVersions = vllmrules.Versions
rep := contracts.Validate(b)
```

A release the rules do not know gets the `engine-version-known` warning. Rules that answer
for a release other than the scenario's are an error, `engine-version-matches`, and are not
applied: a lookup that falls back to a nearby release would pass a scenario its own release
refuses. A bundle with a scenario and no rules gets the error `engine-rules-given`.
[Engine rules](../reference/rules.md) lists all three.

The full API is on [pkg.go.dev](https://pkg.go.dev/github.com/inference-sim/blis-substrate/contracts).
