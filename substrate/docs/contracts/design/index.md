# Design notes

<p class="lede">The reasoning behind particular choices in the schemas, for readers who
want to know why a document has the shape it has. Each note stands alone.</p>

[Why a model is a graph](model-graph.md)
:   Why a model is described as the GPU work it launches rather than in its vendor's
    configuration format, and why the derived graph is committed.

[Where the kernel stops](kernel-boundary.md)
:   How the `Kernel` interface divides work between a cost model and a simulator, and why
    a request and its resolution are kept apart.

[Parallel layouts](layouts.md)
:   How a pool's widths determine its rank count and expert group, and the checks that
    decide whether a layout could be launched.

For more detail than these notes give, read the source. Each package begins with a comment
that says what it is for and what it leaves out.
