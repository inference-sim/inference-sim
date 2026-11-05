# Fabrics

[[catalog.count.fabrics]] inter-node fabric classes, in `networks/`. A fabric belongs to a cluster, not to a chip, so each is defined once and paired with a chip by the scenario that uses it.

[[catalog.networks.table]]

Bandwidth is nominal, per GPU and per direction: the line rate of the network cards serving one GPU, converted to gigabytes per second. The fraction a real transfer achieves is a coefficient in [blis-registry](https://github.com/inference-sim/blis-registry). *RDMA field* shows whether the file sets the optional `RDMA` key; a file that does not set it makes no claim either way. The catalog states no separate bandwidth for moving a KV cache from a prefill server to a decode server. The kernel's own transfer pricing uses the fabric's figure between nodes; inference-sim currently takes the figure from its `--pd-transfer-bandwidth` flag.

## Sources

Each file records the source of its figure, and the conversion applied to it, in its `_comment` keys. They are reproduced here as written.

[[catalog.networks.notes]]
