# Storage tiers

[[catalog.count.tiers]] tiers a KV cache can be offloaded to, all in [`devices/storage.yaml`](https://github.com/inference-sim/blis-substrate/blob/[[catalog.ref]]/catalog/devices/storage.yaml).

[[catalog.storage.table]]

A tier is a name and a description. The catalog states no rate and no latency for it, because each such figure was fitted to one host's measurements or chosen as a representative value, and neither is a declared fact of a device. The kernel prices a transfer as the larger of the tier's base latency and the bytes divided by its rate, so the base latency decides the cost of small transfers, and it reads the three terms from the coefficient registry, where each is scoped to the hardware it describes and carries its method and evidence. See the [`cost-model-offload-tiers` reference](../../registry/reference/sets/cost-model-offload-tiers.md) for every value and [how the tier coefficients are written](../../registry/reproducing-coefficients.md#offload-tiers).
