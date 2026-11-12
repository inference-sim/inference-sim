# Where the kernel stops

<p class="lede">A simulator and a cost model share the work of deciding how long things take.
The <code>Kernel</code> interface divides it by one test. If a cost depends on when a
request arrived, it belongs to the simulator. If it follows from the batch and the
deployment alone, it belongs to the kernel.</p>

## Purity is the boundary

Every kernel method is a pure function of its arguments and the configuration it was
constructed from. A simulator relies on three consequences. It can call the kernel at any
simulated instant, in any order. It can test the kernel with no scheduler attached. It can
cache a step's cost by the shape of the batch.

Purity then decides ownership. Queueing delay, the order of admission and preemption all
depend on how busy the engine is, which is the simulator's state, so they stay out of the
kernel. Memory per rank, the time of a forward pass, transfer times and per-request host
work follow from the batch and the deployment, so they belong in it.

A KV transfer from prefill to decode shows where the line falls. It runs concurrently with
execution and decides when a request becomes schedulable on the decode side. It therefore
sets an arrival time, not a step time. A kernel that added it to step time would charge
for latency the engine already hides.

## Host work is the kernel's too

Tokenization, detokenization and request teardown are CPU work, not GPU work. They are on
the interface anyway (`AdmissionOverhead`, `AdmissionDemand`, `OutputTokenOverhead`, `CompletionOverhead`),
because the cost model already carries a host term for launching GPU kernels. Splitting host
costs between two owners would leave a caller fetching some coefficients from the kernel
and others from somewhere else.

## No Init

A kernel comes from its implementation's constructor, fully resolved. An `Init` method on
the interface would admit a value that was never initialized, and would make every other
method's behavior depend on whether `Init` had been called. That hidden state is what
purity rules out.

## The request and the resolution are different values

A deployment states requests, and the kernel's constructor may not honor them. A
tensor-parallel group that crosses a node boundary without multi-node NVLink cannot use the
custom all-reduce GPU kernel, whatever the deployment asked for. Async scheduling has its
own disqualifying conditions.

The interface returns both, separately. `Deployment()` returns the pool as the document
stated it. `Resolved()` returns what will run, and lists every request that was overridden
and why.

`Deployment()` is on the interface for a further reason. A simulator needs the pool's
admission settings (`max_num_seqs`, `max_num_batched_tokens`, `block_size`), and the kernel
is the one value that knows which pool it priced. A caller that kept its own copy of the
deployment and a pool index would hold a second answer to "which pool is this". If the two
disagreed, a decode pool would be priced at a prefill pool's parallelism and the simulation
would still run. Returning the pool removes the second answer.

## A step has two bounds

A forward pass is a sequence of layers, each waiting on the one before. Within a layer, work
on different resources (SMs, HBM, NVLink, the NIC) can overlap. The kernel reports a bound
on each side, `Overlap` and `NoOverlap`, defined in
[Kernel interface](../reference/kernel.md#step-estimates).

A single maximum over the whole step would assume every layer overlaps every other. That
understates a step in which no one resource dominates, and single-request decode is such a
step. A configuration whose conclusion changes between the two bounds needs a measurement
rather than a prediction, and a single number would hide that.

`Bottleneck` names the resource that did the most work over the step. It points to the next
action, adding GPUs or raising the batch size, and the answer can change with batch size
within one deployment.

## The batch carries counts, not labels

A request is described by three token counts (scheduled this step, already computed, prompt
length) and its prefix-cache hit, rather than by a "prefill" or "decode" label. The three
counts are what the engine's own scheduler reads. A label would move an engine
implementation detail into the simulator, and under speculative decoding a request that
schedules more than one token may still be decoding.

The cache hit is a count of tokens, not a rate. The simulator has already computed it; a
cost model given a rate would have to guess the count.

Source: [`kernel/kernel.go`](https://github.com/inference-sim/blis-substrate/blob/main/contracts/kernel/kernel.go).
