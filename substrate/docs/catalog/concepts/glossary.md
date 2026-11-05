# Glossary

Terms as this site uses them. Where a term names a key or a value in a catalog file, it is set in `code`.

## The catalog and BLIS

Chip
:   One GPU product, such as `h100`, described by a file in `hardware/`. Its figures are per GPU.

Coefficient
:   A learned number: one fitted to measurements, or assumed, for a cost model, recorded in blis-registry with how it was obtained and the scope it applies to.

Config
:   A model's `config.json`, the configuration file its vendor publishes, committed to the catalog byte for byte.

Declared fact
:   A figure written down from a published source, such as a config or a datasheet. The catalog holds only declared facts. See [Declared, learned, chosen](declared-facts.md).

Deriver
:   `scripts/derive_graph.py`, which translates a config into a model graph. Its `HANDLERS` table maps each supported architecture to a *handler*, a function that builds the layers of that architecture's family.

Fabric
:   An inter-node network class, such as `ib-400g`, described by a file in `networks/`.

Kernel
:   blis-latency-kernel, the cost model that prices a batch from a model graph, a chip, coefficients and a deployment. (A *GPU kernel* is a routine that runs on the GPU; the site says so when it means one.)

Model graph
:   A model expressed as the operations a forward pass launches, in `models/<name>/graph.yaml`. See [Model graphs](model-graphs.md).

Nominal
:   As the source states it, not as a benchmark achieves it. Every catalog figure is nominal.

Pool
:   One serving engine in a deployment. A colocated pool runs prefill and decode together; a disaggregated deployment has separate prefill and decode pools.

Price
:   To compute what an operation or a batch costs in time and memory.

Provenance
:   Where a figure came from. Chip and fabric files state it in the `Provenance` field, as `vendor_spec` or `derived`.

Release
:   A git tag `MAJOR.MINOR.PATCH` with its GitHub release notes. The *docs version* `MAJOR.MINOR` describes the newest patch on that line.

Scenario
:   The file that describes one run: which model, chip, fabric and coefficient sets, and each pool's parallelism and engine settings. The user supplies it; no BLIS repository owns scenarios.

Step
:   One iteration of a serving engine, in which it runs one batch through the model. The simulator asks the kernel for the time of each step.

Storage tier
:   A place a KV cache can be offloaded to, such as CPU memory or an NVMe drive, described by an entry in `devices/storage.yaml`.

Workload
:   The distribution of request lengths, prompt and output, described by a file in `workloads/`. It does not include the arrival rate. A replay workload in the same directory instead names a load generator's trace-replay scenario over a pinned corpus, and leaves the lane count to each run.

## Model graphs

Collective
:   An operation that moves data between GPUs: `AllReduce`, `AllGather`, `ReduceScatter`, `All2All`.

Emit condition
:   The `emit` field of a collective: the parallelism under which it runs. See [Model graphs](model-graphs.md#what-a-graph-contains).

Layer kind
:   One distinct layer structure, written as a small acyclic graph of primitives.

Primitive
:   One of the nine operations a model graph is built from, such as `GEMM` or `Attention`.

Speculator
:   The part of a graph that describes a draft module, which proposes tokens for the main model to verify (speculative decoding). Its `method` is the serving engine's name for the technique, such as `deepseek_mtp`.

Stack
:   The sequence of layer kinds: an optional prologue, an optional pattern repeated some number of times, and an optional epilogue.

## Model architecture

Attention kinds
:   `gqa`: grouped-query attention, in which several query heads share one key and value head; it also covers ordinary multi-head attention.
:   `mla`: multi-head latent attention, which caches one compressed latent vector per token instead of per-head keys and values.
:   `sparse_mla`: MLA that reads less of its cache, by selecting the top-scoring cached tokens (`index_topk`), by reading a compressed cache (`compress_ratio`), or both.
:   `swa`: sliding-window attention, in which each token attends only to the most recent `window` tokens.

Decode
:   The phase that generates output tokens, one per step for each request, or more with a speculator.

KV cache
:   The keys and values (or latent vectors) attention keeps for every token already processed, so that it does not recompute them.

LM head
:   The final matrix multiply that turns the last hidden state into scores over the vocabulary.

Mixture of experts (MoE)
:   A layer with many parallel feed-forward *experts*, of which each token uses only a few: its *top-k*. *Shared experts* process every token.

Multi-token prediction (MTP)
:   A draft module made of one or more extra layers trained to predict tokens beyond the next.

Prefill
:   The phase that processes a request's prompt and fills its KV cache.

PD disaggregation
:   Running prefill and decode in separate pools, so that each request's KV cache moves from a prefill server to a decode server.

Recurrent kinds
:   `mamba2`: a state-space layer whose state has a convolutional part and a state-space part.
:   `kda`: Kimi Delta Attention, a linear attention whose state is a single matrix.
:   `gdn`: Gated DeltaNet, a linear attention of the same family as KDA.

Tensor, expert and sequence parallelism
:   Ways of splitting a model across GPUs: tensor parallelism splits each matrix multiply, expert parallelism places different experts on different GPUs, and sequence parallelism splits activations along the sequence.

## Hardware and number formats

BF16, FP8
:   16-bit brain floating point, and 8-bit floating point. A chip's `TFlopsPeak` is its dense BF16 rate.

HBM
:   High-bandwidth memory, a GPU's main memory.

INT4, NVFP4, MXFP4
:   4-bit weight formats. INT4 stores integers with one scale per group of weights. NVFP4 is NVIDIA's 4-bit float, with an FP8 scale per 16 values. MXFP4 is the Open Compute Project's microscaling 4-bit float, with a power-of-two scale per 32 values.

MFU
:   Model FLOPs utilization: achieved floating-point operations per second, as a fraction of the peak.

NVLink
:   NVIDIA's GPU-to-GPU link inside a node or, on rack-scale systems such as GB200 NVL72, across a rack.

RDMA
:   Remote direct memory access: moving data between machines without passing it through either CPU.

SM
:   Streaming multiprocessor, the unit an NVIDIA GPU's compute is divided into.

Units
:   MB, GB and TB are decimal: 10<sup>6</sup>, 10<sup>9</sup> and 10<sup>12</sup> bytes, as datasheets quote them. GiB is 2<sup>30</sup> bytes. Gb/s is gigabits per second, so 400 Gb/s is 50 GB/s.
