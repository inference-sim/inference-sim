# Research

<p class="lede">How well the kernel predicts measured serving performance, where it falls
short and why, and how its laws are tested. For readers deciding how far to trust a
prediction, or working on the model itself.</p>

[Accuracy](accuracy.md)
:   Where end-to-end figures are published, and the forward-pass comparison that chooses
    between the kernel's two estimates.

[How the laws are tested](testing.md)
:   The kinds of test the repository relies on, and why most of them compare against the
    engine's own source or against an exact relation rather than against fixed numbers.

The places where the kernel knowingly prices something differently from the engine are
listed on [Known divergences](../reference/divergences.md).
