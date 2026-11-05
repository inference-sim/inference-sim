# Concepts

<p class="lede">Four short pages: where things live, what the documents are, how they are
checked, and the rules every document follows. Read them in order if you are new. The
terms below are used with one meaning throughout.</p>

[The BLIS repositories](repositories.md)
:   What each of the six repositories owns, and what each takes from this one.

[Documents](documents.md)
:   The eleven kinds of document, where each lives, and how they refer to each other.

[Validation](validation.md)
:   The two questions a validator asks, why they are kept apart, and what a report holds.

[Contracts](contracts.md)
:   Where a document's name comes from, how units are written, why an unknown key is an
    error.

## Terms

### The project

BLIS
:   The Blackbox Inference Simulator: a simulator for LLM serving, made of the five
    repositories described in [The BLIS repositories](repositories.md).

catalog
:   [blis-catalog](https://github.com/inference-sim/blis-catalog): models, chips, fabrics,
    storage classes and workload shapes, each a fact someone wrote down.

registry
:   [blis-registry](https://github.com/inference-sim/blis-registry): coefficient sets, the
    numbers a cost model uses, each with how it was obtained and where it holds.

cost model
:   A function from a batch of requests and a deployment to memory use and the time of
    one forward pass. To *price* something is to compute that time or memory for it.

kernel
:   A cost model seen through the `Kernel` interface defined here. blis-latency-kernel
    implements it. The word also appears in its ordinary GPU sense, as in "an all-reduce
    kernel"; the context always says which.

### Documents

scenario
:   The problem a run is given: a model, the cluster it may use, the traffic it serves, the
    coefficient sets that price it, and the engine version. Designs compared for one
    problem share one scenario.

deployment
:   The choices made against a scenario: how the cluster is divided into pools, each pool's
    parallel layout and engine settings, the offload tiers, and how KV moves from prefill to
    decode. A search over designs varies the deployment and leaves the scenario alone.

pool
:   The nodes that serve one role (colocated, prefill or decode) with one layout and one
    set of engine settings. A pool may run several engine instances.

storage class, offload tier
:   A storage class is a kind of device, such as `cpu_dram` or `nvme_gen4`, described in the
    catalog. An offload tier is a deployment's use of one storage class for KV, with a
    capacity.

### Serving

engine, engine version
:   The serving engine is the software that runs the model; vLLM today. A scenario's engine
    version is the exact release string, such as `0.29.0`, of the engine whose behavior the
    run describes. Its coefficients are fitted against that release, and it selects the
    rules pack.

prefill, decode
:   Prefill is the forward pass over a request's prompt. Decode is each later step, which
    produces one token (or, with speculative decoding, a few). A deployment may run them on
    the same pool (colocated) or on separate pools (disaggregated).

KV cache
:   The attention keys and values an engine keeps for each request's tokens between steps.
    It is most of a request's memory, and moving it is most of the cost of disaggregation.

rank
:   One GPU's share of an engine. An engine has one rank per GPU.

parallel widths
:   How an engine divides work across ranks. `tp`, tensor parallelism, splits each layer's
    matrices. `pp`, pipeline parallelism, splits the layers into stages. `dp`, data
    parallelism, runs replicas. `pcp` and `dcp`, prefill- and decode-context parallelism,
    split one request's context. `ep`, expert parallelism, spreads a mixture-of-experts
    (MoE) model's experts and is derived from the others. See
    [Parallel layouts](../design/layouts.md).

### Validation

field layer, rule layer
:   The two layers of validation. The field layer finds field problems: a document is
    malformed whatever engine runs it. The rule layer finds rule problems: the named engine
    release would refuse the document, or would run it differently from what it says.

rules pack
:   The checks and accepted values for one engine release, registered under its version
    string.

problem
:   One thing a validator reports, at a path such as `pools[0].parallel.dcp`, with a
    severity: an error, which fails validation, or a warning, which does not.
