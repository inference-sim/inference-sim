# Using the catalog

A run of BLIS uses the catalog embedded in the blis-substrate release it was built with, or a
directory you name instead. It never downloads catalog data and never writes to it.

To use the catalog:

1. **Use the embedded copy, or name a directory.** [Point BLIS at the catalog](point-blis.md)
   gives the flags, how a run names the entries it needs, and what each result records about
   the catalog it used.
2. **Read the formats of your release.** The loaders parse catalog files strictly, so a
   directory named with `--catalog` must match the formats of the simulator's substrate
   release. [Releases and pinning](releases.md) explains why.
3. **Experiment in a copy.** To try a hypothetical model or chip, edit a copy of the catalog
   and name it when you run. [Experiment with a candidate catalog](candidate.md) shows how.
