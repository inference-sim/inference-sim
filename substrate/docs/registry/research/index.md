# Methodology

Three documents record how the registry's numbers are obtained and checked. They are
written for someone who wants to challenge a value: each claim comes with the
measurement behind it, and the claims later found wrong are kept, marked as retracted,
next to what replaced them.

[Methodology](../methodology.md)
:   The rules: what a coefficient is, which dataset may be used for what, how fits are
    held out, and the record of each decision made under those rules.

[Reproducing the coefficients](../reproducing-coefficients.md)
:   The mechanics: where the data is, and which command regenerates which family.

[Choosing the overlap band](../band-selection.md)
:   One model-selection decision, worked through: which of the kernel's two step-time
    estimates a consumer should use.

## The argument in brief

1. **A coefficient is a constant of a law the kernel evaluates**, never a per-model
   correction. Its value and its citation move together.
   [§1](../methodology.md#1-what-a-coefficient-is)
2. **Fit the engine you predict.** AISimulate measures several engines; the right lane is
   decided per operator, by evidence. [§2](../methodology.md#2-the-lane-rule-fit-the-engine-you-predict)
3. **Each dataset has one job.** Operator sweeps fit; whole-forward measurements validate
   and select; end-to-end runs only evaluate. A whole-pass total cannot identify a
   per-primitive constant. [§3](../methodology.md#3-the-three-level-data-separation)
4. **Hold out along a dimension that generalization must cross**: a part, a model, a
   parallelism, a shape axis. Never a random row split.
   [§4](../methodology.md#4-train-validate-evaluate), [§7](../methodology.md#7-held-out-validation-of-the-lane-decisions)
5. **A better fit to one primitive can be a worse model.** Nothing ships without an
   end-to-end check. [§3](../methodology.md#3-the-three-level-data-separation)
6. **Gaps are declared.** An `assumed` value says what it rests on and what would replace
   it. [§6.0](../methodology.md#60-what-is-reproducible-and-what-is-not)

## Where to look for a specific question

| Question | Section |
|---|---|
| Why is this family on this engine's measurements? | [§2](../methodology.md#2-the-lane-rule-fit-the-engine-you-predict), [§8](../methodology.md#8-lane-provenance-every-coefficient-family-and-every-remaining-borrow) |
| Which values can be reproduced, and which cannot? | [§6.0](../methodology.md#60-what-is-reproducible-and-what-is-not) |
| How do I reproduce an accuracy figure exactly? | [§6.1](../methodology.md#61-reproducing-the-end-to-end-evaluation-tables) |
| Why is a part missing from a family? | [§5](../methodology.md#5-per-part-provenance) |
| How are 8- and 16-rank collectives priced on parts never swept at those widths? | [§8.5](../methodology.md#85-rack-scale-rank-widths-and-the-one-place-this-registry-extrapolates) |
| What did the whole-forward data find? | [§9](../methodology.md#9-the-fpm-mixed-rows-and-what-they-found) |
| Why does sparse MLA have no rate of its own? | [§10](../methodology.md#10-sparse-mla-the-byte-count-is-the-fix-and-the-rate-is-not-fittable) |
| How do the memory terms compose? | [§11](../methodology.md#11-memory-occupancy-what-composes-and-where-each-term-comes-from) |

## Reading the numbers in these documents

The methodology is a record kept as the work proceeded, and its tables give the counts
and figures measured when each section was written. Counts of entries move with every
refit. For the registry as it stands, use the [reference pages](../reference/index.md),
which are generated from the data on every build. Where the two disagree on a count, the
reference pages are current.
