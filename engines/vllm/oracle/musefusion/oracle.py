# Original to blis-vllm.
"""Muse Spark fusion oracle: which of the three fused kernels vLLM v0.33.0 builds for a Muse Spark
deployment, and for which step sizes it runs them.

vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5) runs Muse Spark through
vllm/models/muse_spark/nvidia/. Three of its fused kernels replace nodes the catalog graph states
as separate launches:

  qk_norm_rope_kv_insert   QK-norm, RoPE and the KV-cache write as one launch
                           (MuseSpark1xAttention, model.py:1138-1146; the op,
                           common/ops/fused_qk_norm_rope_kv.py:23-38)
  moe_finalize_all_reduce  the MoE finalize, its post-expert norm and the expert-parallel combine
                           all-reduce as one launch (MusePostExpertFinalize._create_all_reduce and
                           __call__, nvidia/muse_moe_runner.py:200-274)
  column_gather_boundary   the sharded unpooler's all-gather, the gated residual and the next
                           RMSNorm as one launch (MuseSpark1xModel._configure_column_gather,
                           nvidia/model.py:1577-1610; _all_gather_norm_residual_then_norm, :214-252)

None of the three is a torch.compile pass: MuseSpark1xModel is @ignore_torch_compile
(model.py:1500-1501), so each is decided by Python at model build and at each call. This oracle
runs those decisions as vLLM wrote them on a CPU, for each case of DEPLOYMENTS:

  the attention backend          CudaPlatform.get_valid_backends (platforms/cuda.py), through the
                                 block-size oracle's loader (oracle/blocksize/oracle.py, imported,
                                 not changed), for the Muse layer shapes
  supports_fused_kv_insert       common/ops/fused_qk_norm_rope_kv.py:23-38, on that backend class
  FusedMoEParallelConfig.make    model_executor/layers/fused_moe/config.py:1116-1246, for the layout
  needs_pre_norm_all_reduce      models/muse_spark/common/moe.py:114-119
  MusePostExpertFinalize._create_all_reduce, with the workspace factory
                                 maybe_create_moe_finalize_all_reduce and _create_on_all_tp_ranks
                                 (nvidia/ops/mnnvl_fusions.py:32-58, 105-126) run as written
  MuseSpark1xModel._configure_column_gather, with maybe_create_all_gather_dual_norm
                                 (mnnvl_fusions.py:61-82) run as written
  MuseSpark1xMoE.shards_unpooler and MuseSpark1xDecoderLayer._defers_column_gather
                                 (model.py:638-644, :1493-1497), for each step size in TOKENS

Four decisions sit inline in __init__ bodies that cannot be built on a CPU without a process group
per rank, so they are set from the values the code computes, each cited where it is set below
(INLINE): MuseSpark1xAttention's fused_qk_norm_rope_kv_insert conjunction (model.py:1138-1146)
over the model's own config, MuseSpark1xMoE's replicated_tokens and tep_default (:505-517),
min_sharded_unpooler_tokens (:535-539), and MuseSpark1xModel.fuse_across_layers (:1533).

What a CPU cannot know is replaced (REPLACEMENTS, printed with the table): the GPU (the block-size
oracle's Sim platform at the case's compute capability), FlashInfer's MNNVL workspaces (a stand-in
that records it was built; whether one CAN be built is the case's `multicast` input, which
_create_on_all_tp_ranks reads through flashinfer.comm.mnnvl.is_multicast_supported), the
collective that makes every TP rank agree (one process stands for every rank, which agree by
construction), and tp_group_spans_nodes (the case's `spans_nodes` input). The attention backend is
resolved under every setting of the block-size oracle's facts; a case whose answer depends on a
fact is refused.

Usage:
    python3 -I oracle/musefusion/oracle.py <vllm checkout> json            print the cases (the Go test's input)
    python3 -I oracle/musefusion/oracle.py <vllm checkout> check <file>    fail unless <file> equals what it prints
The Muse config.json is read from $BLIS_ORACLE_CATALOG/<model>/config.json, as the block-size
oracle reads its models.
"""
import itertools
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCKSIZE = os.path.join(os.path.dirname(HERE), "blocksize")
sys.path.insert(0, BLOCKSIZE)

COMMIT = "4ba787ec9959e566d3b587a796dff07673324da5"
MODEL = "muse-spark-open-816b-a42b-nvfp4"
CATALOG = os.environ.get("BLIS_ORACLE_CATALOG", "")

# The step sizes the token-dependent checks are run at: each side of every bound the code compares
# a step's tokens with (_MULTI_NODE_SHARDED_UNPOOLER_MIN_TOKENS 32, model.py:154;
# _MULTI_NODE_FUSED_ALL_REDUCE_MAX_TOKENS 128 and _FUSED_ALL_REDUCE_MAX_TOKENS 256,
# muse_moe_runner.py:151-152; _SHARDED_PROJECTION_MAX_TOKENS 256, model.py:151).
TOKENS = [1, 31, 32, 128, 129, 256, 257]

REPLACEMENTS = []


def deployments():
    """Every layout the decisions read, on Hopper (9.0) and Blackwell (10.0)."""
    out = []
    for cap, tp, pp, dp, ep, kv, spans, multicast in itertools.product(
            (9, 10), (1, 2, 8), (1, 2), (1, 2), (False, True), ("auto", "fp8", "nvfp4"),
            (False, True), (False, True)):
        if tp == 1 and spans:
            continue  # a group of one rank spans nothing
        if cap == 9 and multicast:
            continue  # read only on capability family 100 (mnnvl_fusions.py:35-38)
        if cap == 9 and kv == "nvfp4":
            continue  # no attention backend takes an NVFP4 KV cache on SM90, so vLLM does not start
        out.append(dict(cap=cap, tp=tp, pp=pp, dp=dp, pcp=1, dcp=1, ep=ep, kv_cache_dtype=kv,
                        spans_nodes=spans, multicast=multicast))
    # Decode- and prefill-context parallelism, each on its own, from the Blackwell TP8+EP layout.
    for pcp, dcp in ((2, 1), (1, 2)):
        out.append(dict(cap=10, tp=8, pp=1, dp=1, pcp=pcp, dcp=dcp, ep=True, kv_cache_dtype="auto",
                        spans_nodes=True, multicast=True))
    return out


def text_config():
    with open(os.path.join(CATALOG, MODEL, "config.json")) as f:
        c = json.load(f)
    return types.SimpleNamespace(**c.get("text_config", c))


def load(vllm_dir):
    import oracle as blocksize  # oracle/blocksize/oracle.py

    v = blocksize.load_vllm(vllm_dir)
    REPLACEMENTS.extend(v.stubs.REPLACEMENTS)

    class Workspace:
        """Stands for a FlashInfer MNNVL workspace: built, and compiled ahead of capture."""

        def __init__(self, *a, **k):
            self.args = a

        def compile(self, **k):
            pass

    multicast = {"on": False}
    mods = {
        "flashinfer.comm.mnnvl": dict(is_multicast_supported=lambda dev: multicast["on"]),
        "flashinfer.comm.mnnvl_cutedsl_moe_finalize": dict(MNNVLCuteDSLMoEFinalizeAllReduceWorkspace=Workspace),
        "flashinfer.comm.mnnvl_cutedsl_ag": dict(MNNVLCuteDSLAllGatherDualNormWorkspace=Workspace,
                                                 LLAllGatherDualNormTuning=lambda **k: None,
                                                 MNNVLCuteDSLAllGatherNormQuantWorkspace=Workspace),
    }
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
    REPLACEMENTS.append(("flashinfer.comm MNNVL workspaces",
                         "a stand-in that records it was built; whether one can be is the case's multicast "
                         "input, read by vLLM's own _create_on_all_tp_ranks (mnnvl_fusions.py:32-58)"))

    import torch
    from unittest import mock

    torch.ops._C = mock.MagicMock(name="torch.ops._C")
    REPLACEMENTS.append(("torch.ops._C",
                         "compiled ops that quantization modules probe when imported (for example "
                         "w8a8_utils.py:28-41); none of them feeds the three decisions"))
    from vllm.config import ParallelConfig
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    import vllm.models.muse_spark.common.ops.fused_qk_norm_rope_kv as qkv_ops
    import vllm.models.muse_spark.common.moe as common_moe
    import vllm.models.muse_spark.nvidia.ops.mnnvl_fusions as mnnvl
    import vllm.models.muse_spark.nvidia.muse_moe_runner as runner
    import vllm.models.muse_spark.nvidia.model as model

    tp_group = types.SimpleNamespace(world_size=1, rank_in_group=0, device_group=None, cpu_group=None)
    get_tp_group = lambda: tp_group  # noqa: E731
    mnnvl.get_tp_group = runner.get_tp_group = get_tp_group
    mnnvl._all_tp_ranks = lambda flag: bool(flag)
    torch.accelerator.current_device_index = lambda: 0
    REPLACEMENTS.append(("the TP group and _all_tp_ranks",
                         "one process stands for every TP rank: world_size is the case's tp, and the MIN "
                         "all-reduce over ranks that agree by construction returns the flag"))
    import vllm.model_executor.layers.fused_moe.config as moe_cfg

    rank0 = types.SimpleNamespace(rank_in_group=0)
    moe_cfg.get_dp_group = moe_cfg.get_pcp_group = lambda: rank0
    moe_cfg.get_tensor_model_parallel_rank = lambda: 0
    REPLACEMENTS.append(("fused_moe/config.py get_dp_group, get_pcp_group, get_tensor_model_parallel_rank",
                         "rank 0 of each group: the ranks only say which rank this process is, and every "
                         "flag the decisions read is the same on every rank"))
    spans = {"on": False}
    runner.tp_group_spans_nodes = model.tp_group_spans_nodes = lambda: spans["on"]
    REPLACEMENTS.append(("tp_group_spans_nodes", "the case's spans_nodes input (mnnvl_fusions.py:20-23)"))
    return types.SimpleNamespace(v=v, blocksize=blocksize, torch=torch, ParallelConfig=ParallelConfig,
                                 FusedMoEParallelConfig=FusedMoEParallelConfig, qkv_ops=qkv_ops,
                                 common_moe=common_moe, runner=runner, model=model, tp_group=tp_group,
                                 multicast=multicast, spans=spans)


def attention_backends(L, cfg, d, facts):
    """The backend class vLLM selects for each Muse attention layer kind, under one fact setting."""
    from vllm.v1.attention.selector import AttentionSelectorConfig

    v = L.v
    v.stubs.FACTS.update(facts)
    v.Sim.cap = (d["cap"], 0)
    heads = max(1, cfg.num_attention_heads // d["tp"])
    kv_heads = max(1, cfg.num_key_value_heads // d["tp"])
    vc = v.VllmConfig()
    vc.model_config = types.SimpleNamespace(
        get_num_attention_heads=lambda pc: heads, get_num_kv_heads=lambda pc: kv_heads,
        get_head_size=lambda: cfg.head_dim, hf_text_config=cfg,
        is_mm_prefix_lm=False, is_diffusion=False, is_hybrid=False)
    vc.parallel_config.decode_context_parallel_size = d["dcp"]
    out = []
    with v.set_current_vllm_config(vc):
        # config.layer_types: sliding_attention and full_attention layers (model.py:1101-1110).
        for window in sorted(set(t == "sliding_attention" for t in cfg.layer_types)):
            sel = AttentionSelectorConfig(head_size=cfg.head_dim, dtype=L.torch.bfloat16,
                                          kv_cache_dtype=d["kv_cache_dtype"], block_size=None, use_mla=False,
                                          has_sink=False, use_sparse=False, has_sliding_window=window,
                                          use_dcp=d["dcp"] > 1)
            valid, _ = v.Sim.get_valid_backends(v.DeviceCapability(d["cap"], 0), sel, num_heads=heads)
            out.append(min(valid, key=lambda c: c.priority).backend_class if valid else None)
    return out, vc


def qk_fused(L, cfg, d):
    """MuseSpark1xAttention.fused_qk_norm_rope_kv_insert for the target model's layers, and whether
    vLLM finds an attention backend under every setting of the facts."""
    answers = {}
    for setting in itertools.product((False, True), repeat=len(L.v.stubs.FACTS)):
        facts = dict(zip(sorted(L.v.stubs.FACTS), setting))
        backends, vc = attention_backends(L, cfg, d, facts)
        if any(b is None for b in backends):
            answers[tuple(sorted(facts.items()))] = None
            continue
        with L.v.set_current_vllm_config(vc):
            supported = all(L.qkv_ops.supports_fused_kv_insert(types.SimpleNamespace(
                attn_backend=b, kv_sharing_target_layer_name=None, kv_cache_dtype=d["kv_cache_dtype"]))
                for b in backends)
        # INLINE, model.py:1138-1146 on the target model (is_draft False): use_qk_norm is True
        # (:975), qk_norm is built has_weight=False (:997-1007), attention temperature tuning is a
        # drafter's only (:1112-1118), head_dim is config.head_dim, and the default dtype is the
        # model's bfloat16.
        inline = (cfg.head_dim in (64, 128, 256))
        answers[tuple(sorted(facts.items()))] = bool(supported and inline)
    # A fact setting under which vLLM finds no backend for a layer is one where it refuses to start
    # (NVFP4 KV on SM90, and on SM100 without TRT-LLM's kernels), which is not this decision's.
    vals = set(a for a in answers.values() if a is not None)
    if len(vals) != 1:
        raise SystemExit(f"case {d}: the fused insert depends on a fact ({answers})")
    return vals.pop(), None not in answers.values()


def moe_finalize(L, cfg, d):
    """Whether MusePostExpertFinalize builds the fused all-reduce, and its token bound."""
    L.v.Sim.cap = (d["cap"], 0)
    L.tp_group.world_size = d["tp"]
    L.multicast["on"] = d["multicast"]
    L.spans["on"] = d["spans_nodes"]
    # The fields make reads (fused_moe/config.py:1198-1246, the rest of make), at ParallelConfig's own defaults; a
    # real ParallelConfig cannot be built on a machine with no GPUs (its world-size check).
    fields = L.ParallelConfig.__dataclass_fields__
    pc = types.SimpleNamespace(enable_expert_parallel=d["ep"], all2all_backend=fields["all2all_backend"].default,
                               enable_eplb=fields["enable_eplb"].default)
    mpc = L.FusedMoEParallelConfig.make(tp_size_=d["tp"], pcp_size_=d["pcp"], dp_size_=d["dp"],
                                        sp_size_=1, vllm_parallel_config=pc)
    moe_config = types.SimpleNamespace(moe_parallel_config=mpc, experts_per_token=cfg.num_experts_per_tok,
                                       skip_final_all_reduce=False, in_dtype=L.torch.bfloat16)
    norm = types.SimpleNamespace(hidden_size=cfg.moe_hidden_size, variance_size_override=None,
                                 variance_epsilon=cfg.post_norm_eps, fused_gain_center=1.0,
                                 applies_weight_before_norm=True)
    # INLINE, model.py:505-517 and :613-617: the runner is asked to fuse when tep_default holds
    # (the config states no muse_spark_tep_fused_moe_all_reduce).
    fuse = tep_default(d)
    # The workspace cache lives for one engine process (muse_moe_runner.py:154); each case is
    # an engine of its own.
    L.runner._FUSED_ALL_REDUCE_WORKSPACES.clear()
    fin = object.__new__(L.runner.MusePostExpertFinalize)
    fin._norm = norm
    fin.pre_norm_all_reduce = L.common_moe.needs_pre_norm_all_reduce(moe_config)
    ws = L.runner.MusePostExpertFinalize._create_all_reduce(fin, moe_config) if fuse else None
    if ws is None:
        return False, 0
    return True, fin._all_reduce_max_tokens


def tep_default(d):
    """INLINE, model.py:505-517: replicated_tokens (tp above 1, no MegaMoE, no sequence parallelism
    of either kind, data parallelism 1, no batch invariance), on CUDA with capability 10.0 or above.
    MegaMoE needs moe_backend deep_gemm_mega_moe (:392) and sequence parallelism follows it (:361-370);
    the sequence-parallel MoE needs data parallelism above 1 (config/parallel.py:768-785)."""
    return d["tp"] > 1 and d["dp"] == 1 and d["cap"] >= 10


def column_gather(L, cfg, d):
    """Whether _configure_column_gather gives the sharded MoE layers the fused boundary, and the
    step sizes at which a layer defers its gather to it."""
    L.v.Sim.cap = (d["cap"], 0)
    L.tp_group.world_size = d["tp"]
    L.multicast["on"] = d["multicast"]
    L.spans["on"] = d["spans_nodes"]
    M = L.model
    shard = tep_default(d)  # INLINE: shard_unpooler, model.py:521-523, with no config override

    def norm():
        return types.SimpleNamespace(has_weight=True, variance_epsilon=cfg.rms_norm_eps, fused_gain_center=1.0)

    layers = []
    for _ in range(2):
        moe = object.__new__(M.MuseSpark1xMoE)
        moe.shard_unpooler = shard
        # INLINE, model.py:535-539.
        moe.min_sharded_unpooler_tokens = 32 if shard and d["spans_nodes"] else 1
        layer = object.__new__(M.MuseSpark1xDecoderLayer)
        torch_module_init(layer)
        layer.feed_forward = moe
        layer.self_attn = types.SimpleNamespace(pre_qkv_proj_norm=norm())
        layer.post_ffn_norm = norm()
        layer.all_gather_dual_norm = None
        layers.append(layer)
    m = types.SimpleNamespace(layers=layers, start_layer=0, end_layer=len(layers), norm=norm(),
                              config=types.SimpleNamespace(hidden_size=cfg.hidden_size),
                              fuse_across_layers=d["pp"] == 1)  # INLINE, model.py:1533
    M._dual_norm_constants = lambda post, pre: {}
    M.MuseSpark1xModel._configure_column_gather(m)
    fused = layers[0].all_gather_dual_norm is not None
    sharded = [n for n in TOKENS if layers[0].feed_forward.shards_unpooler(n)]
    deferred = [n for n in TOKENS if M.MuseSpark1xDecoderLayer._defers_column_gather(layers[0], n)]
    return fused, sharded, deferred


def torch_module_init(obj):
    import torch.nn as nn

    nn.Module.__init__(obj)


def run(vllm_dir):
    L = load(vllm_dir)
    cfg = text_config()
    rows = []
    for d in deployments():
        qk, always_builds = qk_fused(L, cfg, d)
        ar, ar_max = moe_finalize(L, cfg, d)
        fused, sharded, deferred = column_gather(L, cfg, d)
        rows.append(dict(case=d, qk_norm_rope_kv_insert=qk, backend_under_every_fact=always_builds, moe_finalize_all_reduce=ar,
                         moe_finalize_all_reduce_max_tokens=ar_max, column_gather_boundary=fused,
                         unpooler_sharded_at=sharded, gather_deferred_at=deferred))
    return dict(commit=COMMIT, model=MODEL, head_dim=cfg.head_dim, tokens=TOKENS,
                replacements=[dict(what=w, why=y) for w, y in REPLACEMENTS], cases=rows)


def main(argv):
    if len(argv) < 3 or argv[2] not in ("json", "check"):
        print(__doc__, file=sys.stderr)
        return 2
    out = json.dumps(run(argv[1]), indent=1, sort_keys=True) + "\n"
    if argv[2] == "json":
        sys.stdout.write(out)
        return 0
    with open(argv[3]) as f:
        if f.read() != out:
            print(f"{argv[3]} differs from what vLLM {COMMIT} decides; regenerate it", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
