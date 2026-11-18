# inference-sim (BLIS, the Blackbox Inference Simulator)

BLIS is a discrete-event simulator of an LLM serving stack: arrivals and workloads, a cluster of
engine instances, the embedded llm-d router (admission, flow control, routing), KV caching and
offload, prefill/decode disaggregation and autoscaling. It is CPU-only and deterministic: the
same seed gives byte-identical results. The design is at
**https://inference-sim.github.io/blis-design/**, and it is the place to read everything not on
this page.

## What this repository is, owns, and never holds

- It **is** the simulator: the clock and event loop, workload generation and trace replay, the
  cluster and its placement, the control loops that run beside the router (autoscaler, LoRA
  placement), metrics, and the `blis` command.
- It **owns** the simulation of the whole stack and the Result document of a run. It drives each
  instance's engine only through the engine interface of blis-substrate's contracts, and asks the
  engine for every step.
- It **never holds** an engine's scheduling logic (blis-vllm), a price (the latency kernel in
  blis-substrate), a catalog fact or a fitted coefficient (blis-substrate), or a router
  algorithm (the llm-d router fork). It has no routing, admission or flow-control logic of its
  own. Nothing is fetched at run time.

## How it connects

`stack.json` records what this release pins, and the build selects exactly those versions.

- **Pins** blis-substrate (contracts, the cost model, the catalog and the coefficient registry,
  embedded in its module), blis-vllm (the vLLM engine) and the llm-d router fork
  `inference-sim/llm-d-router`, which replaces `github.com/llm-d/llm-d-router` at a `-blis.N`
  tag. The dependencies run one way: blis-substrate, then blis-vllm, then inference-sim.
- **Pinned by** blis-config-search and accuracy-dashboard, which pin the inference-sim release
  tag. No other repository pins it.
- **Implements and consumes** blis-substrate's contracts: the Scenario and Deployment documents
  it reads, the `engine.Engine` interface it drives, the router `embed` contract it hosts, and
  the Result document it writes. Non-test code imports blis-vllm at one build site,
  `cmd/engine_build.go`.

The reference is printed from the data and the code, not written: `blis schema`, `blis fields`,
`blis metrics`, `blis invariants`, `blis catalog` and `blis router-configs`, with `--help` on
every command, the JSON Schemas and the Go doc comments.

## Build, test and run

<!-- BEGIN GENERATED from stack.json: toolchain -->
Requires Go 1.27.2, the toolchain blis-substrate's contracts/toolchain names at the pinned release.
<!-- END GENERATED -->

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
go test ./...                      # CI runs the full set; run the packages you touch
./blis run --scenarios examplescenarios \
  --scenario deepseek-v4.1-flash-h100-tp8-dspark5.yaml \
  --rate 10 --num-requests 100
```

Every `blis run` and `blis replay` needs a scenario directory (`--scenarios <dir>`, any directory
of scenario YAMLs; `examplescenarios/` ships ready-to-run ones) and a scenario (`--scenario <file>`
within it). The catalog and the coefficient registry default to the copies embedded in the
blis-substrate module this build pins; `--catalog`/`BLIS_CATALOG` and
`--registry`/`BLIS_REGISTRY` replace them whole (the flag wins). The Result records which data the
run read under `data`. To read a catalog of your own:

<!-- BEGIN GENERATED from stack.json: data-checkout -->
```bash
git clone https://github.com/inference-sim/blis-substrate.git
git -C blis-substrate checkout fa28d5116739dc582efdb7694acf15a294cdc62b
export BLIS_CATALOG=$PWD/blis-substrate/catalog   # optional: replaces the embedded catalog (or pass --catalog)
```
<!-- END GENERATED -->

Contributing: [CONTRIBUTING.md](./CONTRIBUTING.md) holds the engineering standards and the pull
request workflow; [CLAUDE.md](./CLAUDE.md) is the operating context for agents.

## Citation

If you use BLIS in your research, please cite our paper (accepted at ACM SIGMETRICS,
to appear). Preprint: [arXiv:2610.14255](https://arxiv.org/abs/2610.14255).

```bibtex
@article{guhathakurta2026blis,
  title     = {Simulate Before You Scale: High-Fidelity LLM Inference Planning Without GPUs},
  author    = {Guhathakurta, Dipanwita and Toslali, Mert and Parthasarathy, Srinivasan and Chen, Jing and Kalantar, Michael and Masluk, Nicholas A. and Ramani, Vishakha and Tantawi, Asser and Oliveira, Fabio},
  journal   = {Proc. ACM Meas. Anal. Comput. Syst.},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  note      = {To appear},
  eprint    = {2610.14255},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PF}
}
```

## License

Apache License, Version 2.0. See [LICENSE](./LICENSE).
