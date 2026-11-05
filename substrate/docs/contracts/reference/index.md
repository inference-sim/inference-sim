# Reference

<p class="lede">One page per document kind, plus the kernel interface, the engine interface,
the router contract, the closed vocabularies, the engine rules interface and the validator
commands. Each document page opens with an example, then lists every key the document may
contain, then every check its validator makes.</p>

The key tables are tested against the Go types: a key the type declares and the table
omits, or a key the table lists and the type does not declare, fails the build. The
vocabularies are generated from the code. For anything finer than these pages, read the
source; each page links to it.

| Page | Documents | Package |
|---|---|---|
| [Scenario](scenario.md) | `Scenario` | [`spec/scenario`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/scenario) |
| [Deployment](deployment.md) | `Deployment` | [`spec/deployment`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/deployment) |
| [Workload](workload.md) | a scenario's `workload`, a catalog workload shape | [`spec/workload`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/workload) |
| [Model graph](model.md) | `ModelGraph`, a model's identity manifest | [`spec/model`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/model) |
| [Hardware](hardware.md) | chip, fabric, storage classes | [`spec/hardware`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/hardware) |
| [Coefficient set](coefficients.md) | `CoefficientSet` | [`spec/coefficient`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/coefficient) |
| [Acceptance profiles](acceptance.md) | `AcceptanceProfileSet`, inline acceptance values, a scenario's pin | [`spec/acceptance`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/acceptance) |
| [Routing profiles](routing.md) | `RoutingProfileSet`, a scenario's pin | [`spec/routing`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/routing) |
| [Connector registry](connectors.md) | `ConnectorRegistry` | [`spec/connector`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/connector) |
| [Evaluation run](evaluation.md) | `EvaluationRun` | [`spec/evaluation`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/evaluation) |
| [Measured run](measurement.md) | `MeasuredRun`, the licence registry | [`spec/measurement`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/measurement) |
| [Result](result.md) | `Result` | [`spec/result`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/result) |
| [Kernel interface](kernel.md) | the `Kernel` a cost model implements | [`kernel`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/kernel) |
| [Engine interface](engine.md) | the `Engine` an inference-engine model implements, its `Spec` envelope and its contract checks | [`engine`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/engine) |
| [Router contract](embed.md) | the `Runtime` a simulator implements, and the `Router` a hosted router implements | [`embed`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/embed) |
| [Vocabularies](vocabularies.md) | units, methods, scope keys, dtypes, and the rest | [`vocab`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/vocab) and `spec/` |
| [Engine rules](rules.md) | the interface an engine release's rules implement | [`rules`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/rules) |
| [Command line](cli.md) | `validate-catalog`, `validate-registry`, `validate-measurements` | [`validate/cmd`](https://github.com/inference-sim/blis-substrate/tree/main/validate/cmd) |

In the key tables, **Type** is the YAML type. *Required* means that if the key is absent,
the validator reports an error or the loader fails. A key marked *no* may be omitted, and
the table says what omitting it means.
