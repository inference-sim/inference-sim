# LoRA Adapters

BLIS simulates vLLM's LoRA serving: a registry of adapters, a bounded set of adapters resident on each engine, a charge for loading an adapter that is not resident, a per-step compute overhead for batches that use adapters, and an HBM reservation for the resident set. With no adapters declared the subsystem does nothing, and a run is byte-identical to one without it (INV-6, INV-L1).

## Declaring adapters

Adapters come from a YAML file named by `--lora-config`, with a top-level `lora:` block:

```yaml
lora:
  adapter_capacity: 4        # resident adapter slots per instance
  adapters:
    - id: adapter_0
      rank: 8
    - id: adapter_1
      rank: 16
```

`--lora-adapter-capacity` overrides the capacity. Declaring adapters with a capacity of zero is an error. The cost terms below default to the `lora:` block of `defaults.yaml` (`--defaults-filepath`), and a `--lora-*` flag, when set, overrides the file.

| Term | Flag | Meaning |
|---|---|---|
| Load base latency | `--lora-load-base-latency-us` | Fixed microseconds charged to load an adapter that is not resident. |
| Load bandwidth | `--lora-load-bandwidth-bytes-us` | Bytes per microsecond at which an adapter's weights load (must be positive). |
| Footprint | `--lora-footprint-bytes-per-rank` | HBM bytes one rank unit of an adapter occupies (must be positive). |

## What a run charges

- **Capacity and eviction.** Each instance holds at most `adapter_capacity` adapters (INV-L2). An adapter in use, from the moment a request naming it enters the running batch until that request completes, is never evicted (INV-L5).
- **Cold load.** A request for an adapter that is not resident on its instance pays the load latency once, `base + bytes / bandwidth`, and never when the adapter is warm (INV-L3).
- **Step overhead.** A step that serves adapters costs more by an amount that depends on the ranks in the batch, applied to the cost model's step time.
- **HBM reservation.** The footprint of the full resident set is set aside from the KV budget before the cost model counts blocks, so more or larger adapters mean fewer KV blocks (INV-L4).

## Routing

BLIS has no routing of its own. LoRA-aware routing is the llm-d router's `lora-affinity-scorer`: enable it in a scheduling profile of the scenario's `router:` block (see [Routing](routing.md)). Each engine publishes vLLM's `vllm:lora_requests_info` series with its running and waiting adapter sets, and the router scrapes it like any other metric. A `modelRewrite` target in the router manifests must be the scenario's model or a declared adapter.

## What the figures support

Adapter throughput overhead was validated against a published digital-twin comparison, with at most 20% mean absolute percentage error. Adapter TTFT is not supported: the separately calibrated base differs from the comparison's, so do not read an adapter run's TTFT as a prediction.
