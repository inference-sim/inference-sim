# Known divergences

<p class="lede">The places where the kernel knowingly prices something differently from the
engine it models, or leaves part of the engine's behavior out. Each is a note in the source,
beside the code it concerns, and this page is generated from those notes, so it lists exactly
what the code admits to.</p>

Each entry gives a short summary, then the note itself, which records the evidence and cites
the vLLM v0.31.0 lines it rests on. A note is marked in the source by one of the phrases
`KNOWN DIVERGENCE`, `KNOWN over-charge` or `COVERAGE LIMIT`, and `internal/docgen` collects
every one outside the tests.

--8<-- "docs/generated/divergences.md"
