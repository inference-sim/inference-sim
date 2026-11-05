# Research

<p class="lede">How well the kernel's step price matches measured forward passes, where it
falls short and why, and how its laws are tested. For readers deciding how far to trust a
prediction, or working on the model itself.</p>

[Accuracy](accuracy.md)
:   The comparison with measured forward passes, which also chooses between the kernel's two
    estimates, and where the known error comes from.

[How the laws are tested](testing.md)
:   The kinds of test the repository relies on, and why most of them compare against the
    engine's own source or against an exact relation rather than against fixed numbers.

The places where the kernel knowingly prices something differently from the engine are
listed on [Known divergences](../reference/divergences.md).
