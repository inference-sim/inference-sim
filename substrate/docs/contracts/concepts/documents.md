# Documents

<p class="lede">blis-schemas defines eleven kinds of YAML document. A run needs two that its
user writes, a scenario and a deployment. They refer by name to documents committed in the
catalog and the registry.</p>

--8<-- "docs/contracts/figures/documents.html"

## The problem and the choices

A **scenario** states what is given: the model, the cluster it runs on, the traffic it
serves, the coefficient sets that price it, and the engine version. Nothing in a scenario
is a decision. Two candidate designs for the same problem share one scenario.

A **deployment** states what is decided. It divides the cluster into pools, gives each
pool a parallel layout and engine settings, and describes the offload tiers and the
transfer of KV cache from prefill to decode. A search over designs varies the deployment
and leaves the scenario alone.

The two are separate documents because they change at different rates and for different
reasons. Neither carries the load level. Request rate or concurrency is swept by the run
and recorded with its results.

```yaml title="A scenario"
--8<-- "docs/contracts/examples/scenario.yaml"
```

```yaml title="A deployment made against it"
--8<-- "docs/contracts/examples/deployment.yaml"
```

## The eleven kinds

| Document | Lives in | Its name comes from | Reference |
|---|---|---|---|
| Scenario | wherever its user keeps it | its `name` key | [Scenario](../reference/scenario.md) |
| Deployment | wherever its user keeps it | its `name` key | [Deployment](../reference/deployment.md) |
| Model graph | catalog, `models/<name>/graph.yaml` | the directory name, repeated in its `name` key | [Model graph](../reference/model.md) |
| Model identity | catalog, `models/<name>/model.yaml` | the directory name, repeated in its `name` key | [Model graph](../reference/model.md#model-identity) |
| Chip | catalog, `hardware/<name>.yaml` | the file name | [Hardware](../reference/hardware.md) |
| Fabric | catalog, `networks/<name>.yaml` | the file name | [Hardware](../reference/hardware.md#fabric) |
| Storage classes | catalog, `devices/storage.yaml` | each class's key in the file | [Hardware](../reference/hardware.md#storage-classes) |
| Workload shape or replay workload | catalog, `workloads/<name>.yaml` | the file name | [Workload](../reference/workload.md#shape) |
| Coefficient set | registry, `coefficients/<name>.yaml` | the file name, which a scenario uses; its `name` key matches by convention | [Coefficient set](../reference/coefficients.md) |
| Acceptance-profile set | registry, `acceptance/<name>.yaml` | the file name, which a scenario's `acceptance.profiles` uses; its `name` key matches by convention | [Acceptance profiles](../reference/acceptance.md) |
| Evaluation run | wherever its user keeps it | its `name` key | [Evaluation run](../reference/evaluation.md) |
| Measured run | catalog, `measurements/<license_ref>/<name>.yaml` | its `name` key, which must match the file name | [Measured run](../reference/measurement.md) |
| Licence registry | catalog, `measurements/sources.yaml` | each source's `id` | [Measured run](../reference/measurement.md#licence-registry) |

A model entry in the catalog is a directory. It holds an identity manifest recording where
the model's configuration came from and, once the catalog has derived it, a graph of the
GPU work a forward pass launches. Beside them sits the vendor's own `config.json`. It
belongs to the vendor, so it has no schema here and is checked only for being a non-empty
JSON object.

## References are names

A scenario never contains a chip or a model. It contains the name of one, and the name is
resolved against a catalog checkout when the scenario is used. This keeps a scenario
short, and a correction to the catalog reaches every scenario that names the corrected
entry.

It also means a scenario can be well formed and still name something that does not exist.
`hardware: nvidia-h100` is a valid string; the catalog calls that chip `h100`. Checking
that each name resolves is a separate step, described in
[Validation](validation.md#do-the-names-resolve).

## Traffic at two fidelities

A scenario's `workload` takes one of two forms. A **shape** names a catalog workload,
which describes traffic as distributions: mean prompt and output lengths, their spread and
bounds, and one shared prefix length. A **trace** points to a captured file of real
requests, with their own lengths, arrival times and prefix structure; see
[Workload](../reference/workload.md#trace). A scenario that asks only a capacity question
omits `workload`.

## Evaluation runs

An evaluation run records what a benchmark measured. For each concurrency level it gives
the throughput, the time to first token, the inter-token latency, and optionally KV
utilization and cache hit rate. It names the scenario it measured, so a prediction for that
scenario can be compared with it.
