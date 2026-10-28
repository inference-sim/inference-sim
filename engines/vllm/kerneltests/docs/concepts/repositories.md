# The kernel in BLIS

<p class="lede">BLIS predicts how an LLM serving deployment performs. It is five
repositories, each owning one kind of thing. This one owns pricing: turning a batch of
requests into the time the GPUs take to serve one step of it.</p>

| Repository | Owns | What the kernel takes from it, or gives it |
|---|---|---|
| [blis-schemas](https://github.com/inference-sim/blis-schemas) | the formats | Imports its document types, loaders and validators, and implements its `kernel.Kernel` interface; takes engine rules through its `rules.EngineRules` interface |
| [blis-vllm](https://github.com/inference-sim/blis-vllm) | vLLM's engine behaviour, and its rules packs | Its rules packs are what a caller passes as `Repos.Rules`; this repository's own commands and tests use them, the library does not import them |
| [blis-catalog](https://github.com/inference-sim/blis-catalog) | declared facts: models, chips, fabrics, storage tiers, workloads | Reads a model's graph, a chip, a fabric and the storage tiers |
| [blis-registry](https://github.com/inference-sim/blis-registry) | learned numbers: coefficients, each with its method and scope | Reads the coefficient sets a scenario names |
| **blis-latency-kernel** | pricing | — |
| [inference-sim](https://github.com/inference-sim/inference-sim) | simulation: queues, batching, admission, preemption | Is called by it, once per simulated step, in pull request [#1851](https://github.com/inference-sim/inference-sim/pull/1851) |

--8<-- "docs/figures/repositories.html"

## Pricing apart from simulation

A simulator decides which requests form a batch and when. The kernel decides what that batch
costs. The two questions change for different reasons: scheduling policy on one side,
hardware and kernels on the other. Keeping them apart means either can be replaced without
touching the other, and the kernel can be tested without a scheduler attached.

The line between them follows from one rule: the kernel is a pure function. Anything whose
value depends on *when* a request arrives belongs to the simulator: how long it waits in a
queue, the order requests are admitted in, whether one is preempted. Anything that follows
from a batch and the deployment belongs to the kernel. The interface that encodes this rule,
`kernel.Kernel`, is defined in blis-schemas rather than here, so that the simulator and any
other cost model depend on the same contract.

## Pricing apart from data

The kernel contains no model names, no chip specifications and no coefficients. What it does
contain is the engine's behavior: vLLM's rules and defaults, a few of which depend on a chip's
generation, and the byte layouts of the engine's cache formats. A model arrives
as a graph of a few kinds of operation (matrix multiplies, attention, collectives and a
handful of others), declared in the catalog; a chip arrives as its peak rates and memory; a
coefficient arrives from the registry with the method that produced it and the scope it
applies to. The kernel dispatches on the kinds of operation, never on which model it is
pricing, so adding a model is a change to the catalog alone.

A *scenario* names what is to be served: the model, the chip, the coefficient sets and the
engine version. A *deployment* is one configuration for it: pools of GPUs, each with its
parallelism and engine settings. The two are separate documents, so that a search can vary
the deployment while the scenario stays fixed;
[Scenarios and their inputs](../guides/inputs.md) shows both.

The coefficients are what make the laws match measured hardware: how far below its peak a
matrix multiply runs at a given size, the fixed cost and the bandwidth of each collective, the
host time per step. Each comes from the registry, not from this repository, recorded there
with the method that produced it, and a kernel lists every one in scope for its deployment.
[Refuse, assume, override](resolution.md) explains how.

## Dependencies run one way

This module imports blis-schemas and nothing else in BLIS. It does not import the simulator:
the simulator imports it, and an import in the other direction would make the two
repositories depend on each other. The command that compares this kernel with the simulator's
older latency models therefore lives in the simulator, as `cmd/blisbaseline`.

The catalog and the registry are not Go modules. The kernel reads their files at the paths a
scenario implies, through blis-schemas' loaders, and validates them with blis-schemas'
validators. The catalog's and the registry's own CI run the same validators, each at the
blis-schemas release it pins, so a malformed file is caught where it is written as well as
where it is read.

## How the simulator uses it

In pull request #1851 of inference-sim, the package `sim/kernelmodel` adapts a kernel to the simulator's latency interface. For each step it
turns the scheduled requests into a `kernel.Batch`, asks the kernel for the step, and takes
the `NoOverlap` estimate as the step's duration. It computes no cost of its own.
[Use the kernel from a simulator](../guides/simulator.md) describes what such an adapter
must supply.

## Versions

Every input is pinned to a release, and each repository moves to a newer one when it is
ready. [Sources of truth](../reference/sources.md) lists the pins this version of the kernel
uses.
