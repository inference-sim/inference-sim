---
hide:
  - navigation
---

# Models

[[catalog.count.models]] models. Each has a page with its layer stack, its dimensions, every node of its model graph, and where its config came from.

Weights
:   The format the parameters are stored in, followed by any format individual nodes override it with.

Sequence mixing
:   Every attention and recurrent kind in the main stack, including the small attention an indexer uses to choose which tokens a sparse layer reads. Kinds used only by the speculator are not counted. The [glossary](../../concepts/glossary.md#model-architecture) defines each kind.

Routed experts
:   For the widest expert layer in the main stack, the number of experts and the number each token is routed to.

Speculator
:   The serving engine's name for the draft method, and the number of tokens it proposes per step.

[[catalog.models.table]]
