---
hide:
  - navigation
  - toc
---

# cost-model-host-overheads

CPU time spent by the engine around the GPU work: tokenizing a prompt, launching kernels
or replaying a captured CUDA graph, detokenizing each output token, and tearing down a
finished request.

**Every entry is `assumed`.** No public dataset measures an engine's host path:
AISimulate times kernels, FPM times one synchronized forward pass, and InferenceX is
used for evaluation only. The values rest on reasoning and, for three of them, on the
order of magnitude of an earlier BLIS fit against a different functional form. Each
entry's rationale states its reasoning and the measurement that would replace it.

The set is kept separate from the measured sets so that its provenance is not mistaken
for theirs. Its scope lists every catalog part because the work is CPU-side and has no
per-part value; listing them makes adding a part a visible edit here.

Assumed is not the same as unimportant. The largest single accuracy movement recorded in
this project, measured-tier TTFT error from 52.94% to 31.47%, came from one of these
seven ([methodology §6.0](../../methodology.md#60-what-is-reproducible-and-what-is-not)).

<!-- registry:set cost-model-host-overheads -->

**Regenerate:** nothing generates this set; it is edited by hand, and every edit should
say what evidence moved the value.
