---
hide:
  - navigation
  - toc
---

# cost-model-recurrent

The constants for recurrent (linear-attention) state updates in hybrid models, priced
like the other primitives as a floor plus a rate:

> latency = `recurrent_decode_floor_<kind>` + tokens / `recurrent_decode_rate_<kind>`

Three kinds are fitted: KDA (Kimi-K3), GDN, and Mamba2 (Nemotron-3).

**Coverage differs by kind, and it matters more than the values.** The KDA sweep covers
the whole layer. The Mamba2 sweep covers only the convolution; the selective scan that
makes up the rest of a Mamba2 layer is in no AISimulate collection. The Mamba2 entries
are therefore a *lower bound* on a Mamba2 layer's cost, and their rationale says so.

**The geometry is part of the fit.** These constants carry no head-count term, so each
is fitted at the head geometry the catalog's model declares. Pooling geometries fits
none of them.

L40S is deliberately absent: its fits are markedly worse than on every other part, and
the set's header records the figures.

<!-- registry:set cost-model-recurrent -->

**Regenerate:** `relane_recurrent_kda.py` (KDA) and
`relane_recurrent_family.py --family {gdn,mamba2}`.
