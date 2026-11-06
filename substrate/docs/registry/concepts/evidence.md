# Where the numbers come from

The registry draws on public data only, and gives each dataset one job. Keeping the jobs
separate is what lets an accuracy figure mean something: a dataset used to fit a number
cannot also be the evidence that the number is right.

| Dataset | What it measures | Its job here |
|---|---|---|
| [NVIDIA AISimulate](https://github.com/ai-dynamo/aisimulate) operator sweeps | One kernel at a time (a GEMM, an attention call, a collective) across many shapes, per GPU and per serving engine | **Fitting.** Every `measured` entry comes from these sweeps. |
| AISimulate system descriptors and memory model | NVIDIA's own per-part corrections and capacity figures | **Transcription.** The source of every `vendor_spec` entry. |
| [AISimulate FPM](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset) | One whole forward pass at a known batch and KV composition | **Validation and model selection.** It may fit only a term that is itself about the whole pass. |
| SemiAnalysis InferenceX | End-to-end serving runs, with a scheduler and a client | **Evaluation only.** Never used to fit or select anything. |
| vLLM start-up logs in vLLM's issue tracker | CUDA-graph capture memory on real deployments | **Anchoring** the magnitude of four `assumed` memory entries. |

## Why a sweep can fit a constant and a forward pass cannot

An operator sweep varies one primitive with everything else held fixed, so it identifies
that primitive's two or three constants and nothing else. A whole-forward measurement is
one number per batch: the sum over every layer and every resource. Many assignments of
constants produce the same sum, so fitting a per-primitive constant to it would attribute
to that constant whatever every other term gets wrong. This is a fact about
identifiability, not a policy, and it does not change as the dataset grows.

## Fit the engine you predict

AISimulate measures the same operator under several serving engines (vLLM, SGLang,
TensorRT-LLM) and collective libraries (NCCL). BLIS predicts vLLM. Where the engines run
different kernels, the measurements differ, sometimes in sign between GPU generations, so
an entry is fitted on the lane that matches what vLLM executes. Where vLLM delegates to
NCCL, as it does for every collective except all-reduce, the NCCL lane *is* vLLM's path.
Where the engines share kernels, the lanes agree and the choice is made on data volume.

The rule is applied per operator, and the decisions are recorded with their evidence in
[the methodology](../methodology.md#2-the-lane-rule-fit-the-engine-you-predict). The
current state, computed from each entry's primary citation when this page was built:

<!-- registry:sources -->

## A better fit can be a worse model

The kernel's accuracy rests partly on errors that cancel. A correction that improves one
primitive's fit can make the end-to-end estimate worse, because it removes an error that
was offsetting another. This has happened six times in this project. So no per-primitive
fit is committed without an end-to-end check, and a correction whose offsetting term has
not been identified waits.

## Gaps are declared, not filled

Where no dataset measures a quantity, the entry says so: it is `assumed`, its rationale
states the reasoning, and it names the measurement that would replace it. The host
overheads are the clearest case. No public dataset times an engine's Python path on the
parts these entries price, so all seven are assumed, and each entry describes the
experiment that would measure it. (FPM's GLM-5.3-Flash vLLM leaves on GB300 time the
scheduler's per-request cost indirectly; they are held-out evidence, not a fit, see the
methodology §10.7.)

## Every measured number can be re-derived

Each generated set is written by a script that its header names, and the scripts are pure
functions of their input data: given the same AISimulate collection, they print the same
file. The registry's derivation tests re-run them and compare the output with what is
committed, so a value edited by hand, or a citation that no longer reproduces its value,
fails.

Those tests need a local clone of AISimulate. CI does not fetch one, so on a pull request
they skip and say why; the schema gate and the property tests run on every pull request
regardless. [Validation](../reference/validation.md) lists what runs where, and
[Reproducing the coefficients](../reproducing-coefficients.md) is the step-by-step guide.

For the full argument behind each rule, with the measurements and the retractions, read
the [methodology](../research/index.md).
