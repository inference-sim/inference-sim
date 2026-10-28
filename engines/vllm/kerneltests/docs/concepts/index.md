# Concepts

<p class="lede">Five pages on what the kernel computes and why it is built the way it is.
Read them in order the first time; each assumes the ones before it.</p>

[The kernel in BLIS](repositories.md)
:   The five BLIS repositories, what this one takes from each, and why pricing lives apart
    from simulation and from data.

[Pricing a step](step.md)
:   How one forward pass is turned into time: the resources a step uses, the laws that price
    each, and how they compose into the two estimates a step returns.

[Memory](memory.md)
:   What a GPU holds before any request arrives, and what each request adds as its context
    grows.

[Parallelism](parallelism.md)
:   Tensor, data, expert, prefill-context and decode-context parallelism: what each divides,
    what each costs, and why it matters which GPUs end up in each group.

[Refuse, assume, override](resolution.md)
:   What the kernel does with a deployment it cannot price exactly, and how it says so.

The pages explain the model rather than list it. Where a statement rests on a detail of
the engine, the page names the source file, and the comment there cites the vLLM lines.
