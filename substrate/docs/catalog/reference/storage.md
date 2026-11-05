# Storage tiers

[[catalog.count.tiers]] tiers a KV cache can be offloaded to, all in [`devices/storage.yaml`](https://github.com/inference-sim/blis-catalog/blob/[[catalog.ref]]/devices/storage.yaml).

[[catalog.storage.table]]

The kernel prices a transfer of *b* megabytes as the larger of the tier's base latency and *b* divided by its bandwidth, so the base latency decides the cost of small transfers. It varies across these tiers by more than four orders of magnitude, from 1 µs for CPU memory to 30 ms for object storage.
