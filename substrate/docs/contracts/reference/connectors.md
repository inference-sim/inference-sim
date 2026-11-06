# Connector registry

<p class="lede">Which KV-connector names a deployment may use, which engine interface each
implements, and which engine model reproduces its behaviour. A deployment's
`offload.connector` and `pd_transfer.connector` are looked up here rather than in a list
compiled into a simulator, so a connector that behaves like an existing one is data, not
code.</p>

```yaml
--8<-- "docs/contracts/examples/connector-registry.yaml"
```

An entry earns `behaves_like` only by evidence. Implementing an engine's connector interface
is not the same as behaving like one of its connectors, so a vendor connector is registered
only once it passes the differential tests the engine module runs against the engine
release; until then it is unsupported. This page describes the shape. The entries for vLLM
live with the engine model, in blis-vllm.

<!-- fields: connector.Registry -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `ConnectorRegistry`. |
| `name` | string | yes | The registry's identity. |
| `connectors` | mapping | yes | At least one entry, by connector name exactly as the engine registers it (vLLM's `KVConnectorFactory.register_connector`). |

<!-- fields: connector.Entry -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `upstream` | string | yes | The engine interface the connector implements, as the engine names it. Which interfaces exist is the engine module's to check. |
| `behaves_like` | string | yes | The engine model that reproduces the connector's behaviour, in the engine module that owns the registry. |
| `roles` | list | yes | What a deployment may use it for: `offload`, `pd_transfer`, `composite`. Each at most once. |
| `evidence` | string | yes | How the entry earned `behaves_like`, such as `diff-test @ vLLM v0.32.0`. |

## Checks

A registry is invalid if `kind` is not `ConnectorRegistry`, `name` is empty, it has no
connectors, a connector name is blank or has surrounding spaces (names match the engine's
exactly), or an entry omits `upstream`,
`behaves_like` or `evidence`, has no roles, or lists an unknown or repeated role.

Source: [`spec/connector`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/connector).
