# The kernel in BLIS

<p class="lede">BLIS predicts how an LLM serving deployment performs. The kernel owns
pricing: turning a batch of requests into the time the GPUs take to serve one step of it. It
is the directory <code>kernel/</code> of blis-substrate, one Go module that also holds the
contracts, the catalog and the registry.</p>

| Part | Owns | What the kernel takes from it, or gives it |
|---|---|---|
| [`contracts/`](../../contracts/index.md) | the formats and the interfaces | Imports its document types, loaders and validators, and implements its `kernel.Kernel` interface; takes engine rules through its `rules.EngineRules` interface |
| [`catalog/`](../../catalog/index.md) | declared facts: models, chips, fabrics, storage tiers, workloads | Reads a model's graph, a chip, a fabric and the storage tiers |
| [`registry/`](../../registry/index.md) | learned numbers: coefficients, each with its method and scope | Reads the coefficient sets a scenario names |
| **`kernel/`** | pricing | — |
| [blis-vllm](https://github.com/inference-sim/blis-vllm) | vLLM's engine behaviour, and its rules packs | Its rules packs and its engine-settings decoder are what a caller passes as `Repos.Rules` and `Repos.EngineSettings`; the kernel does not import blis-vllm |
| [inference-sim](https://github.com/inference-sim/inference-sim) | simulation: queues, batching, admission, preemption | Imports the kernel and calls it once per simulated step |

--8<-- "docs/kernel/figures/repositories.html"

## Pricing apart from simulation

A simulator decides which requests form a batch and when. The kernel decides what that batch
costs. The two questions change for different reasons: scheduling policy on one side,
hardware and kernels on the other. Keeping them apart means either can be replaced without
touching the other, and the kernel can be tested without a scheduler attached.

The line between them follows from one rule: the kernel is a pure function. Anything whose
value depends on *when* a request arrives belongs to the simulator: how long it waits in a
queue, the order requests are admitted in, whether one is preempted. Anything that follows
from a batch and the deployment belongs to the kernel. The interface that encodes this rule,
`kernel.Kernel`, is defined in `contracts/kernel` rather than in `kernel/`, so that the
simulator and any other cost model depend on the same contract.

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
host time per step. Each comes from the registry, not from the kernel's code, recorded there
with the method that produced it, and a kernel lists every one in scope for its deployment.
[Refuse, assume, override](resolution.md) explains how.

## Dependencies run one way

The kernel imports the packages under `contracts/` and nothing else in BLIS. It imports no
engine and not the simulator: the simulator imports it, and an import in the other direction
would make the two depend on each other. The engine repositories and the router fork import
only `contracts/`, so a change to pricing or to a coefficient needs no engine or router
release.

The catalog and the registry are data, not Go packages. The kernel reads their files at the
paths a scenario implies, through the `contracts` loaders, and validates them with the same
validators that CI runs over `catalog/` and `registry/` on every pull request. A malformed file
is therefore caught where it is written as well as where it is read, and both checks run at the
same commit.

## How the simulator uses it

inference-sim imports the kernel, the data loader and the embedded data. For each step it
turns the scheduled requests into a `kernel.Batch`, asks the kernel for the step, and takes the
`NoOverlap` estimate as the step's duration. It computes no cost of its own.
[Use the kernel from a simulator](../guides/simulator.md) describes what such an adapter must
supply.

A run reads the catalog and registry embedded in the substrate release the simulator is built
with, unless it names other directories with `--catalog` and `--registry`. Every run records
which it used, with each one's content hash.

## Versions

The kernel, the contracts, the catalog and the registry share one version: a blis-substrate
release `vX.Y.Z` fixes all four at one commit. A change that touches several of them is one
pull request and one release. [Sources of truth](../reference/sources.md) lists the vLLM
release whose behavior the kernel prices.
