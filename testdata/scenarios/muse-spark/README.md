# Muse-Spark-Open-816B-A42B scenarios

Muse-Spark-Open-816B-A42B (NVFP4 checkpoint, catalog `muse-spark-open-816b-a42b-nvfp4`) on the
three clusters it is served on. These are the setups its vLLM recipe
(https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B) and internal deployment
measurements serve it with, at engine release 0.32.0.

| Cluster | Files | Hardware | Fabric |
|---|---|---|---|
| H100 over RoCE | `muse-h100-roce-*` | two 8-GPU H100 nodes (the NVFP4 weights need 16 GPUs) | `roce-200g` |
| H200 over InfiniBand | `muse-h200-ib-*` | one or two 8-GPU H200 nodes | `ib-400g` |
| GB200 NVL72 | `muse-gb200-nvl-*` | four-GPU GB200 trays in one 72-GPU rack (`gpus_per_rack: 72`), joined by the rack's NVLink switch fabric | `ib-400g` between racks |

## Layouts

Each layout has a `-dspark` twin that drafts 7 tokens with the DSpark drafter
(`speculative.draft_model: muse-spark-open-dspark`).

| Layout | Parallelism |
|---|---|
| `tp8` | TP8 |
| `tep8` | TP8 + expert parallelism, the recipe's default |
| `dep8` | DP8 + EP8 |
| `tp16` | TP16 |
| `tep16` | TP16 + EP16 |
| `pp2tp8` | PP2 x TP8 + EP8 within each stage |

The H100 files also carry the measured arms by name:

| Arm | Setup |
|---|---|
| `armB` | PP2 x TP8, 256k window |
| `armC` | B + DSpark |
| `armB1M` | B at 1,048,576 tokens |
| `armW` | TP16 x EP16 |
| `armD` | C + fp8 KV cache + vLLM's OffloadingConnector with a 512 GiB CPU tier per node (1 TiB for the two-node engine) |

H100 and H200 serve the NVFP4 experts with Marlin (`moe_backend: marlin`). That is the only Hopper backend Muse's MoE has.

DSpark acceptance is pinned in each file at an acceptance length of 2.84. That figure comes from internal deployment measurements of 1.84 accepted tokens per draft with a fixed 7-token verify.

## P/D

`muse-h100-roce-1p1d.yaml` is the measured P/D arm: 1P1D with TP16 + EP16 on each side, on four
RoCE H100 nodes. NIXL pulls the prompt's KV to the decode side. Neither side runs PP, so vLLM's
NIXL worker accepts the layout. The prefill pool runs NIXL as `kv_producer` and the decode pool as
`kv_consumer`. Each engine gets that role from the deployment's `pd_transfer` and its pool's
role. llm-d's disaggregating configuration routes it. Run it with `--num-instances 2
--prefill-instances 1 --decode-instances 1`.

## Routed

Three files (`*-routed.yaml`) serve one measured layout per cluster as two replicas behind the
llm-d router configuration of internal deployment measurements,
[`router/precise-affinity-flow-control.yaml`](router/precise-affinity-flow-control.yaml):

- precise prefix-cache routing, fed by the engines' KV events;
- `prefix-cache-affinity-filter`, which keeps a request on the pods that cache at least half its
  prompt, unless their estimated TTFT is 4.5 s worse (in-flight tokens over 15,000 tokens/s);
- `token-load-scorer`;
- flow control, holding requests once a pod has 32 in flight or a queue builds.

The deployment's pod discovery and service URL are replaced by the simulated pool. Run them
with `--num-instances 2`.

| File | Layout |
|---|---|
| `muse-h100-roce-armC-routed.yaml` | arm C (PP2 x TP8 + DSpark), two replicas on four nodes |
| `muse-h200-ib-tep8-dspark-routed.yaml` | TP8 + EP + DSpark, two one-node replicas |
| `muse-gb200-nvl-tep8-dspark-routed.yaml` | TP8 + EP + DSpark, two two-tray replicas |

`TestMuseSparkPrecisePrefixRoutingFollowsTheCache` checks that the router routes by the engines'
caches. Eight agents each send turns on their own 16k-token context. Behind this router the two
H200 replicas hit their prefix caches more often than behind a load-only router.

## What vLLM refuses

- `muse-h200-ib-dep8-dspark.yaml` at 262,144 tokens is refused at start-up. A rank's pool is smaller than one full-context request's full-attention group, and vLLM v0.32.0 refuses the same pool. The `-128k` twin runs.
- Two layouts are not in the set because vLLM refuses them:
  - DP2 x TP8 x EP16 on Hopper: Muse's Marlin MoE path requires DP1.
  - P/D with a PP2 decode: NIXL refuses a KV consumer at PP > 1.

`TestMuseSparkRunsOnEveryLayoutAndCluster` (`cmd/muse_spark_matrix_test.go`) runs every file here and checks two things: each completes, and the three clusters price the same layout apart.
