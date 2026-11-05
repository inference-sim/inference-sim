"""Tests for the ModelGraph deriver.

The deriver's output is committed, so a reader inspects graph.yaml rather than rerunning
the translation. That makes two failure modes worth guarding. A graph can be wrong in a
way every per-model check still passes — the layer count right, the parameters positive,
and the layer body copied from a different architecture. And a graph can be edited by
hand, which silently decouples it from the config it claims to come from.
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DERIVER = ROOT / "scripts" / "derive_graph.py"


def model_dirs() -> list[Path]:
    return sorted(p for p in MODELS.iterdir() if p.is_dir() and (p / "config.json").is_file())


def graph_of(d: Path) -> dict:
    return yaml.safe_load((d / "graph.yaml").read_text())


def config_of(d: Path) -> dict:
    return json.loads((d / "config.json").read_text())


def text_config(raw: dict) -> dict:
    return raw.get("text_config", raw)


def layer_sequence(stack: dict) -> list[str]:
    return (list(stack.get("prologue", []))
            + list(stack.get("pattern", [])) * stack.get("repeat", 0)
            + list(stack.get("epilogue", [])))


def test_every_model_has_a_graph():
    missing = [d.name for d in model_dirs() if not (d / "graph.yaml").is_file()]
    assert not missing, f"no derived graph for: {missing}"


def test_rederivation_is_stable():
    """--check must pass: a committed graph equals a fresh derivation.

    This is what makes the committed file trustworthy. It fails if someone edits a graph
    by hand, or changes a config without re-deriving."""
    r = subprocess.run([sys.executable, str(DERIVER), str(ROOT), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"re-derivation differs:\n{r.stderr}"


def test_hand_edit_is_detected(tmp_path):
    """The negative case for the check above: a modified graph must be reported.

    A check that cannot fail proves nothing, so this mutates a copy of the tree and
    asserts the deriver notices."""
    import shutil

    tree = tmp_path / "catalog"
    shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = next(p for p in (tree / "models").iterdir() if (p / "graph.yaml").is_file())
    g = yaml.safe_load((target / "graph.yaml").read_text())
    g["global"]["hidden_size"] = int(g["global"]["hidden_size"]) + 8
    (target / "graph.yaml").write_text(yaml.safe_dump(g, sort_keys=False))

    r = subprocess.run([sys.executable, str(DERIVER), str(tree), "--check"],
                       capture_output=True, text=True)
    assert r.returncode != 0, "a hand-edited graph was not detected"
    assert target.name in r.stderr


def test_a_failure_names_its_model_once(tmp_path):
    """Each failure line names the model exactly once (blis-catalog#28).

    A DeriveError carries the model name itself, because derive() is called directly
    and its message has to stand alone there; main() prefixed it again, printing
    `qwen9-1t: qwen9-1t: no handler ...`. Both failure classes are covered: a
    DeriveError, which must not gain a prefix, and a missing config.json, which has no
    other source for the name and must keep it. The third class, an exception the
    deriver does not raise itself, is test_a_deriver_crash_is_reported_against_its_model."""
    models = tmp_path / "models"
    (models / "qwen9-1t").mkdir(parents=True)
    (models / "qwen9-1t" / "config.json").write_text(
        json.dumps({"architectures": ["Qwen9ForCausalLM"]}))
    (models / "no-config").mkdir()

    r = subprocess.run([sys.executable, str(DERIVER), str(tmp_path), "--check"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    lines = r.stderr.splitlines()
    handler = [ln for ln in lines if "no handler for architecture" in ln]
    assert handler == [
        "qwen9-1t: no handler for architecture 'Qwen9ForCausalLM'; add one rather "
        "than letting a default misprice the model"], r.stderr
    assert "no-config: no config.json" in lines, r.stderr


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_layer_count_matches_config(d: Path):
    """Depth must agree with the config, whichever way the config states it.

    Most configs state num_hidden_layers. One shipped hybrid states no layer count at
    all, and its depth is the length of a per-layer type vector."""
    g, t = graph_of(d), text_config(config_of(d))
    derived = len(layer_sequence(g["stack"]))
    stated = t.get("num_hidden_layers")
    vector = t.get("layers_block_type")
    expected = stated if stated is not None else (len(vector) if vector else None)
    if expected is None:
        pytest.skip("the config states neither a layer count nor a layer vector")
    assert derived == expected


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_digest_pins_the_config(d: Path):
    """The recorded digest must be the config's, so a drift between them is detectable."""
    import hashlib

    g = graph_of(d)
    actual = hashlib.sha256((d / "config.json").read_bytes()).hexdigest()
    assert g["derived_from"]["sha256"] == actual


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_attention_shape_matches_config(d: Path):
    """Head counts must come from the config rather than a template.

    A graph whose layer body was copied from another architecture would pass a layer-count
    check and fail here."""
    g, t = graph_of(d), text_config(config_of(d))
    nodes = [n for k in g["layer_kinds"] for n in k["nodes"] if n["op"] == "Attention"]
    if not nodes:
        pytest.skip("no attention layer")
    for n in nodes:
        if n.get("role") == "block_index_scores":
            # A block-sparse layer runs a second, much narrower attention to score which
            # blocks the first one reads. Its head geometry is the indexer's own, not the
            # model's, so asserting the model's head count here would require the deriver
            # to misreport it. Checked against the config it does come from instead.
            #
            # Two families spell those fields differently: MiniMax-M3 nests them in
            # sparse_attention_config, DeepSeek-V4 states index_n_heads and
            # index_head_dim at the top level. Both are read, and a config declaring
            # neither fails rather than skipping the check.
            sparse = t.get("sparse_attention_config") or {}
            heads = sparse.get("sparse_num_index_heads", t.get("index_n_heads"))
            dim = sparse.get("sparse_index_dim", t.get("index_head_dim"))
            assert heads is not None and dim is not None, (
                "an indexer node whose config states no index head geometry"
            )
            assert n["n_q"] == heads
            assert n["d_h"] == dim
            continue
        assert n["n_q"] == t["num_attention_heads"]
        if n["kind"] in ("mla", "sparse_mla"):
            # A latent cache holds one vector per token whatever the config's KV head
            # count says, so the engine pins it to one.
            assert n["n_kv"] == 1
        else:
            assert n["n_kv"] == t["num_key_value_heads"]


# --- The k-pool indexer, exercised directly -----------------------------------------
# No committed config on this branch declares index_kpool, so a catalog-driven test
# would skip and leave the production branch unrun in CI. These call the deriver's own
# helper against a config built from the published GLM-5.3-Flash values, so the branch
# that model will consume is executed here, with its negative case alongside it.

DERIVE = None


def deriver():
    """The deriver module, imported once for the direct-call tests."""
    global DERIVE
    if DERIVE is None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import derive_graph
        DERIVE = derive_graph
    return DERIVE


def kpool_config(**over) -> dict:
    """A config carrying GLM-5.3-Flash's published indexer and latent geometry.

    Values are the pinned revision's (eb9eb208): hidden 4096, 64 heads, kv_lora_rank
    512, q_lora_rank 1536, qk_nope 256, v_head_dim 256, and the indexer's 32 heads of
    128 with index_topk 2048 and index_kpool 4.

    qk_rope_head_dim is 64 here where that config states 0. The published model declares
    mla_use_nope, and the branch that accepts a zero rope width on that declaration is
    the one adding the model -- not this one. These tests are about the indexer, which
    the rope width does not enter, so a nonzero value keeps them independent of that
    change rather than coupling this branch's CI to it."""
    cfg = {
        "hidden_size": 4096, "num_attention_heads": 64, "num_key_value_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 1536,
        "qk_nope_head_dim": 256, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
        "index_kpool": 4, "dtype": "bfloat16", "num_hidden_layers": 45,
        # The architecture matters: the FP32 head gate and the fused A/norm layout are
        # per-family, resolved through this string.
        "architectures": ["Glm5NextForConditionalGeneration"],
    }
    cfg.update(over)
    return cfg


def indexer_nodes(cfg: dict) -> list[dict]:
    """Run attention_block + lightning_indexer, as a handler does."""
    d = deriver()
    nodes = d.attention_block(cfg, "synthetic", cfg["hidden_size"], raw=cfg)
    return d.lightning_indexer(cfg, "synthetic", cfg["hidden_size"], nodes, cfg)


def test_kpool_indexer_projections_come_from_the_latent_rank_and_fuse_wk():
    """The three projections vLLM builds, at the widths it builds them.

    wq_b reads q_lora_rank, not hidden -- that was the single largest error in the old
    fused node. wk and weights_proj are ONE GEMM of [head_dim, n_head]
    (deepseek_v32/attention.py:78-85, glm5next/common/attention.py:272-281). Each width
    is confirmed against the pinned GLM-5.3-Flash safetensors headers: wq_b [4096,1536],
    wk [128,4096] + weights_proj [32,4096], gate [128,4096]."""
    cfg = kpool_config()
    gemms = {n.get("role"): (n["n"], n["k"]) for n in indexer_nodes(cfg)
             if n["op"] == "GEMM"}
    heads, dim = cfg["index_n_heads"], cfg["index_head_dim"]
    assert "index_qk_proj" not in gemms, "the fused node sources the query from hidden"
    assert gemms["index_wq_b"] == (heads * dim, cfg["q_lora_rank"])
    assert gemms["index_wk_weights_proj"] == (dim + heads, cfg["hidden_size"])
    assert gemms["index_kpool_compress_gate"] == (dim, cfg["hidden_size"])


def test_kpool_indexer_prices_the_fp32_head_gate_as_its_own_launch():
    """The head gate is a second GEMM at FP32, not part of the fused projection.

    The forward keeps the first head_dim rows of wk_weights_proj for K, then caches the
    remaining n_head rows transposed to FP32 and runs its own matmul against FP32
    activations: `weights = torch.mm(hidden_states.float(), self._wp_fp32)`
    (glm5next/common/attention.py:341-350). That is a distinct launch and a resident
    FP32 weight copy, so modelling only the merged layer undercounts both."""
    nodes = indexer_nodes(kpool_config())
    gate = next((n for n in nodes if n.get("role") == "index_head_gate_proj"), None)
    assert gate is not None, "no head-gate GEMM; the forward runs a second matmul"
    assert (gate["n"], gate["k"]) == (32, 4096), (
        f"the head gate is {gate['n']}x{gate['k']}; it projects n_head from hidden")
    assert gate.get("weight_dtype") == "fp32", (
        "the head-gate weights are cached as FP32; the global width would understate "
        "the resident bytes and the compute")


def test_kpool_scorer_is_bounded_by_pooling_and_not_by_the_topk_it_computes():
    """index_topk must NOT bound the scoring pass.

    The scorer scores every pooled candidate in order to DISCOVER the top-k; the
    selection then bounds the main MLA read, which carries index_topk itself. In the
    schema index_topk means "this node reads only a top-k of its cache", so putting it
    on the scorer makes it read only the answers it exists to find and caps the one
    context-proportional term in the layer at a constant. DeepSeek-V4's representation
    draws the same line: block_index_scores is unbounded and only the downstream sparse
    MLA node carries the top-k."""
    cfg = kpool_config()
    nodes = indexer_nodes(cfg)
    scorer = next(n for n in nodes if n.get("role") == "block_index_scores")
    assert "index_topk" not in scorer, (
        "the scorer carries index_topk, so it is bounded by the selection it is "
        "supposed to produce")
    assert scorer["compress_ratio"] == cfg["index_kpool"], (
        f"the scorer states compress_ratio {scorer.get('compress_ratio')}; pooling is "
        f"{cfg['index_kpool']}:1 and that is its only bound")
    assert "window" not in scorer, "a window would make the selection look bounded"
    # The main attention keeps the selection it really applies.
    main = next(n for n in nodes
                if n["op"] == "Attention" and n.get("role") != "block_index_scores")
    assert main.get("index_topk") == cfg["index_topk"], (
        "the main latent read must carry the top-k the indexer selects for it")


def test_kpool_prices_the_compression_pass_and_its_ape_state():
    """Pooling is real per-token work, not just a projection.

    vLLM softmax-weights each group of index_kpool keys against a FP32
    [index_kpool, index_head_dim] APE table, then rotates, quantizes and writes one
    pooled state to the cache (glm5next/common/attention.py:394-417). A gate GEMM alone
    does not represent that, so the pass is its own node with its traffic stated: the
    key read, the APE table, and the pooled write."""
    cfg = kpool_config()
    nodes = indexer_nodes(cfg)
    comp = next((n for n in nodes if n.get("role") == "index_kpool_compress"), None)
    assert comp is not None, "no compression pass; pooling is work, not just a gate"
    assert comp["op"] == "Elementwise"
    assert comp["bytes_per_token"] > 0
    # The exact figure is pinned by
    # test_kpool_compression_traffic_is_per_pool_amortized, which derives it from the
    # kernel's own reads. Here the point is that the pass exists at all.
    #
    # The APE is deliberately NOT a node: the only primitive that could carry its bytes
    # is a GEMM, and a GEMM means 2*n*k FLOPs per token plus a launch
    # (blis-latency-kernel plan.go:266-269), which the storage-only table does not run.
    # Its read traffic is inside the figure above.
    assert not any(n.get("role") == "index_kpool_compress_ape" for n in nodes), (
        "the APE is emitted as a node; the only primitive available is a GEMM, which "
        "would invent FLOPs and a launch for a storage-only table")


def test_kpool_indexer_widths_are_visible_against_a_quantized_global():
    """With a BF16 global the overrides are unobservable, so assert against fp8.

    glm5next builds its whole MLA module with quant_config=None -- the call says so,
    `quant_config=None,  # MLA projections are BF16 in checkpoint`
    (glm5next/common/model.py:394) -- so wq_b, wk_weights_proj and the compression gate
    all serve at the base width however the checkpoint is quantized. The APE is FP32 and
    the head gate's cached copy likewise."""
    cfg = kpool_config(quantization_config={"quant_method": "fp8",
                                            "fmt": "e4m3"})
    nodes = indexer_nodes(cfg)
    by_role = {n.get("role"): n for n in nodes}
    for role in ("index_wq_b", "index_wk_weights_proj",
                 "index_kpool_compress_gate"):
        assert by_role[role].get("weight_dtype") == "bf16", (
            f"{role} inherits the quantized global width; vLLM builds this module "
            f"unquantized")
    assert by_role["index_head_gate_proj"]["weight_dtype"] == "fp32"


def test_kpool_compression_traffic_is_per_pool_amortized():
    """The compression kernel runs one program per POOL, not per token.

    kpool_compress.py:165-245 reads slot_score in BOTH passes (the per-dim max for
    softmax stability, then the weighted sum), the FP32 APE row likewise, slot_k once,
    and writes one fp8 vector plus its fp32 scale. bytes_per_token is charged once per
    model token, so the figure is that per-pool total divided by index_kpool.

    Asserted against the kernel's own reads rather than against the deriver's
    expression, and separately asserted NOT to be the naive per-token form that charges
    the whole APE table every token."""
    cfg = kpool_config()
    kpool, dim = cfg["index_kpool"], cfg["index_head_dim"]
    comp = next(n for n in indexer_nodes(cfg)
                if n.get("role") == "index_kpool_compress")
    # Each tensor at the width it is STORED at, which is the width an HBM read costs.
    # slot_score is bf16 -- gate_score is F.linear over bf16 hidden states and the bf16
    # compression gate, and the tail cache calls its copy the "bf16 gate score"
    # (common/attention.py:172). Only the APE is fp32 in memory. The Triton loads cast
    # to fp32 in registers, which moves no bytes.
    BF16, FP32, FP8 = 2, 4, 1
    per_pool = (2 * kpool * dim * BF16    # slot_score, both passes
                + 2 * kpool * dim * FP32  # APE rows, both passes
                + kpool * dim * BF16      # slot_k
                + dim * FP8 + FP32)       # fp8 vector + its scale
    assert comp["bytes_per_token"] == per_pool // kpool, (
        f"compression states {comp['bytes_per_token']} B/token; the kernel moves "
        f"{per_pool} per pool over {kpool} tokens")
    # Two specific wrong answers this must not give.
    naive = dim * BF16 + kpool * dim * FP32 + dim
    assert comp["bytes_per_token"] != naive, (
        "compression is priced as if the whole APE table were read every token")
    all_fp32 = (2 * kpool * dim * FP32 + 2 * kpool * dim * FP32
                + kpool * dim * BF16 + dim + FP32) // kpool
    assert comp["bytes_per_token"] != all_fp32, (
        f"compression states {comp['bytes_per_token']}, the figure you get by sizing "
        f"slot_score from the fp32 accumulation width rather than its bf16 storage")


def test_a_config_without_kpool_gets_no_pooling_nodes_or_bound():
    """The negative case: drop index_kpool and the pooled parts must disappear.

    Same config otherwise, so this isolates the field. A family that declares no pooling
    scans every token, and a compress_ratio there would bound a scan that has nothing
    pooled to bound."""
    cfg = kpool_config()
    cfg.pop("index_kpool")
    nodes = indexer_nodes(cfg)
    roles = {n.get("role") for n in nodes}
    assert "index_kpool_compress_gate" not in roles
    assert "index_kpool_compress" not in roles
    scorer = next(n for n in nodes if n.get("role") == "block_index_scores")
    assert not scorer.get("compress_ratio"), (
        "an unpooled scorer states a compress_ratio, bounding a full-context scan")
    assert scorer["kind"] == "gqa", (
        "an unpooled scorer claims the sparse-latent kind, which implies a bound")
    # The projections are unchanged: vLLM builds the same ones either way.
    gemms = {n.get("role") for n in nodes if n["op"] == "GEMM"}
    assert {"index_wq_b", "index_wk_weights_proj", "index_head_gate_proj"} <= gemms


def test_a_kpool_config_without_a_latent_rank_is_refused():
    """wq_b is sized from q_lora_rank, so a config lacking it must raise.

    Defaulting the width would price the dominant projection on a guess."""
    cfg = kpool_config()
    cfg.pop("q_lora_rank")
    with pytest.raises(deriver().DeriveError, match="q_lora_rank"):
        indexer_nodes(cfg)


# --- Properties and metamorphic relations over the indexer -------------------------
# These assert INVARIANTS rather than committed numbers: a property holds for every
# config in a family, and a metamorphic relation says how the output must move when an
# input moves. Both survive a config update that changes literals, and both catch
# classes of error a fixed-value assertion cannot.


@pytest.mark.parametrize("kpool", [2, 3, 4, 8, 16])
def test_pooled_scan_bound_scales_inversely_with_the_pool_size(kpool: int):
    """METAMORPHIC: multiplying index_kpool must multiply the scan bound identically.

    One K state is cached per index_kpool tokens (attention.py:128-131), and the scan is
    sized max_model_len // index_kpool (:314). So the bound the graph records has to
    track the field exactly -- not approximately, and not saturating at some value. A
    deriver that hard-coded 4, or clamped, or inverted the relation would pass a
    single-value test and fail here."""
    cfg = kpool_config(index_kpool=kpool)
    scorer = next(n for n in indexer_nodes(cfg)
                  if n.get("role") == "block_index_scores")
    assert scorer["compress_ratio"] == kpool, (
        f"index_kpool {kpool} gives compress_ratio {scorer.get('compress_ratio')}; the "
        f"pooled scan is 1/kpool of the token context")
    assert "index_topk" not in scorer, (
        "the scorer is bounded by pooling alone, whatever the pool size")


@pytest.mark.parametrize("kpool", [2, 4, 8])
def test_compression_traffic_falls_as_pools_get_larger(kpool: int):
    """METAMORPHIC: per-token compression traffic must DECREASE as kpool grows.

    The kernel runs once per pool and its per-pool cost grows linearly in kpool (it
    loops over POOL_SIZE slots), while the number of tokens sharing that cost grows
    linearly too -- so the amortized per-token figure converges rather than exploding.
    The one term that does NOT scale is the single pooled write, which is amortized over
    more tokens as kpool grows.

    This is the relation that catches the error the review found: charging the whole
    [kpool, dim] APE table per TOKEN makes traffic RISE with kpool instead."""
    small = next(n for n in indexer_nodes(kpool_config(index_kpool=kpool))
                 if n.get("role") == "index_kpool_compress")["bytes_per_token"]
    large = next(n for n in indexer_nodes(kpool_config(index_kpool=kpool * 2))
                 if n.get("role") == "index_kpool_compress")["bytes_per_token"]
    assert large <= small, (
        f"per-token compression traffic rose from {small} to {large} when the pool size "
        f"doubled; the kernel runs once per pool, so a larger pool amortizes its cost "
        f"over more tokens. A rise means the per-pool cost is being charged per token")


@pytest.mark.parametrize("dim,heads", [(64, 16), (128, 32), (128, 64), (192, 48)])
def test_indexer_projection_widths_track_the_declared_geometry(dim: int, heads: int):
    """PROPERTY: every indexer projection is a stated function of the config.

    wq_b is heads*dim from the LATENT rank; the fused wk_weights_proj is dim+heads from
    hidden; the gate is dim from hidden. Checked across geometries so a deriver that
    happened to match GLM-5.3-Flash's numbers by coincidence -- or that swapped dim and
    heads, which both being powers of two makes easy -- fails."""
    cfg = kpool_config(index_head_dim=dim, index_n_heads=heads)
    g = {n.get("role"): (n["n"], n["k"]) for n in indexer_nodes(cfg)
         if n["op"] == "GEMM"}
    assert g["index_wq_b"] == (heads * dim, cfg["q_lora_rank"])
    assert g["index_wk_weights_proj"] == (dim + heads, cfg["hidden_size"])
    assert g["index_kpool_compress_gate"] == (dim, cfg["hidden_size"])
    assert g["index_head_gate_proj"] == (heads, cfg["hidden_size"])
    assert "index_kpool_compress_ape" not in g, (
        "the APE is a storage-only table and must not be a GEMM")


@pytest.mark.parametrize("freq,offset,layers", [
    (1, 2, 16), (2, 2, 16), (4, 3, 78), (4, 2, 32), (8, 1, 64), (3, 0, 12),
])
def test_indexed_layer_count_matches_the_skip_formula_for_any_frequency(
        freq: int, offset: int, layers: int):
    """PROPERTY: the indexed set is exactly vLLM's formula, over many (freq, offset).

    skip_topk = max(layer_id - offset + 1, 0) % freq != 0
    (deepseek_v32/attention.py:166-173). Asserted against an independent re-expression
    of that formula across a grid, so an off-by-one in the offset, a wrong modulus
    direction, or a clamp that only shows at particular values cannot hide. Also checks
    the invariant that freq=1 indexes EVERY layer, which is the default path the
    families declaring neither field take."""
    d = deriver()
    cfg = {"num_hidden_layers": layers, "index_topk": 2048, "index_n_heads": 32,
           "index_topk_freq": freq, "index_skip_topk_offset": offset}
    got = d.indexed_layers(cfg, "synthetic", layers)
    want = [max(i - offset + 1, 0) % freq == 0 for i in range(layers)]
    assert got == want, (
        f"freq={freq} offset={offset}: indexed {sum(got)} of {layers}, formula gives "
        f"{sum(want)}")
    if freq == 1:
        assert all(got), "freq 1 must index every layer; it is the default"


def test_an_indexed_layer_never_costs_less_than_an_unindexed_one():
    """METAMORPHIC: skipping the indexer may only REMOVE work, never add or change any.

    A skipped layer omits the scoring pass and its projections and keeps everything else
    -- including the bounded main read, since the top-k comes from a shared buffer
    (mla_attention.py:564-566). So the unindexed layer's cost multiset must be a strict
    SUBSET of the indexed one's. This catches two opposite errors at once: a skipped
    layer that kept the scorer (no saving), and one that also lost its top-k bound or
    changed its attention kind (too much removed)."""
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if t.get("compress_ratios"):
            # DeepSeek-V4's layers differ by COMPRESSION RATIO, not by whether they
            # index: its ratio-128 layers read a more heavily compressed stream and
            # legitimately price an attention the ratio-4 layers do not. That is a
            # different axis from the skip_topk one this relation is about, and it is
            # pinned by test_v4_only_ratio_4_layers_have_an_indexer.
            continue
        g = graph_of(d)
        kinds = {k["id"]: k for k in g["layer_kinds"]}
        seq = set(layer_sequence(g["stack"]))
        scoring = {k for k in seq
                   if any(n.get("role") == "block_index_scores"
                          for n in kinds[k]["nodes"])}
        skipping = {k for k in seq - scoring
                    if any(n["op"] == "Attention" for n in kinds[k]["nodes"])}
        if not (scoring and skipping):
            continue
        # Compare like with like: same MLP flavour on both sides.
        for sk in sorted(skipping):
            mlp = "moe" if any(n["op"] == "GroupedGEMM"
                               for n in kinds[sk]["nodes"]) else "dense"
            peers = [k for k in sorted(scoring)
                     if (("moe" if any(n["op"] == "GroupedGEMM"
                                       for n in kinds[k]["nodes"]) else "dense") == mlp)]
            if not peers:
                continue
            # Compared on the nodes a skip cannot legitimately change. The latent norm
            # IS legitimately different: where the family folds the index norm into it,
            # a skipped layer's is narrower, because fused_norm_rope substitutes dummies
            # and skips the indexer program under HAS_INDEXER
            # (deepseek_v32/attention.py:339-360). That reduction is the point of
            # skipping, not added work, and it is pinned by
            # test_the_folded_index_norm_width_is_paired_with_the_layer_running_the_indexer.
            def comparable(kid):
                return collections.Counter(
                    node_cost(n) for n in kinds[kid]["nodes"]
                    if n.get("role") != "qkv_a_layernorm")

            lo, hi = comparable(sk), comparable(peers[0])
            extra = lo - hi
            assert not extra, (
                f"{d.name}: the unindexed layer {sk!r} prices work the indexed layer "
                f"{peers[0]!r} does not: {sorted(extra)[:2]}. Skipping the indexer only "
                f"removes the scoring pass and narrows the fused norm; it cannot add or "
                f"alter anything else")
            assert hi - lo, (
                f"{d.name}: the unindexed layer {sk!r} costs exactly what the indexed "
                f"{peers[0]!r} costs, so skipping the indexer saved nothing")
            checked += 1
    assert checked, "no model mixes indexed and unindexed layers"


def test_the_folded_index_norm_width_is_paired_with_the_layer_running_the_indexer():
    """Only a layer that RUNS the indexer carries its folded norm width.

    A skipped layer takes the else at deepseek_v32/attention.py:339-360 with
    has_indexer = False, and fused_norm_rope substitutes dummies and skips the indexer
    program under HAS_INDEXER. So its norm moves the two latents alone, while an indexed
    layer in the same graph moves the latents plus index_head_dim.

    Asserted as an exact pair ACROSS kinds of one graph, not as two independent values:
    the per-layer gating is the property, and a graph where both kinds agree has lost it
    in one direction or the other. This is the case the conservation test cannot see --
    that one builds both families through indexer_nodes(), so both sides are indexed."""
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        # The folded width is fused_norm_rope's, over the q and kv LoRA latents.
        # DeepSeek-V4.1 does not run that kernel: on SM90 it norms its latents with
        # fused_q_kv_rmsnorm on every layer alike and gives the index key its own
        # k_norm launch (deepseek_v41/attention.py:757-763, 1288-1298), which
        # test_deepseek_v41_charges_each_launch_the_traffic_its_kernel_moves pins.
        # DeepSeek-V4 does not either: it states head_dim rather than a kv_lora_rank,
        # and norms both latents with the same fused_q_kv_rmsnorm on every layer
        # (deepseek_v4/attention.py:528-535).
        if t.get("kv_source_layer_ids") or (
                "kv_lora_rank" not in t and t.get("model_type") == "deepseek_v4"):
            continue
        g = graph_of(d)
        kinds = {k["id"]: k for k in g["layer_kinds"]}
        seq = set(layer_sequence(g["stack"]))
        def norm_of(kid):
            return next((n.get("bytes_per_token") for n in kinds[kid]["nodes"]
                         if n.get("role") == "qkv_a_layernorm"), None)
        indexed = [k for k in seq
                   if any(n.get("role") == "block_index_scores"
                          for n in kinds[k]["nodes"])]
        skipped = [k for k in seq if k not in indexed
                   and norm_of(k) is not None]
        if not (indexed and skipped):
            continue
        act = 2  # bf16; these configs declare bfloat16
        latents = 2 * (int(t["q_lora_rank"]) + int(t["kv_lora_rank"])) * act
        folded = latents + 2 * int(t["index_head_dim"]) * act
        for kid in sorted(indexed):
            assert norm_of(kid) == folded, (
                f"{d.name}/{kid}: runs the indexer and states {norm_of(kid)} B/token; "
                f"fused_norm_rope normalises index_k too, which gives {folded}")
        for kid in sorted(skipped):
            assert norm_of(kid) == latents, (
                f"{d.name}/{kid}: has no block_index_scores node yet states "
                f"{norm_of(kid)} B/token; has_indexer is False for such a layer, so its "
                f"norm moves the two latents alone, which is {latents}")
        assert folded != latents, "index_head_dim is zero, so this pair proves nothing"
        checked += 1
    if not checked:
        pytest.skip("no graph mixes indexed and skipped layers with a fused index norm")


def test_a_family_that_launches_its_index_norm_separately_keeps_the_node():
    """The negative case for the fusion, exercised directly.

    glm5next runs _fused_indexer_k_norm as its OWN call
    (glm5next/common/attention.py:352) after fusing only the two latent norms
    (fuse_qkv_rmsnorm=True, :604), so its indexer must carry an index_k_norm node.
    deepseek_v32 folds all four into fused_norm_rope and must not.

    Called directly because no committed config on this branch is glm5next -- that model
    arrives with the branch consuming this one. Without this, suppressing the node for
    EVERY family would pass: nothing here keeps it."""
    d = deriver()
    cfg = kpool_config()
    nodes = indexer_nodes(cfg)
    kn = [n for n in nodes if n.get("role") == "index_k_norm"]
    assert len(kn) == 1, (
        f"a glm5next-shaped config prices {len(kn)} index_k_norm nodes; it launches that "
        f"norm itself, so exactly one is right")
    dim = cfg["index_head_dim"]
    assert kn[0]["bytes_per_token"] == 2 * dim * 2, (
        f"index_k_norm states {kn[0]['bytes_per_token']} B/token; a bf16 read and write "
        f"over {dim} is {2 * dim * 2}")

    # And the fused family, same helper, must not carry it -- but must account for its
    # bytes in the surviving node. Checked as a CONSERVATION property: total normalisation
    # traffic is the same either way, since fusion removes a launch and not the HBM work.
    fused_cfg = dict(cfg, architectures=["GlmMoeDsaForCausalLM"])
    fused_nodes = indexer_nodes(fused_cfg)
    assert not any(n.get("role") == "index_k_norm" for n in fused_nodes), (
        "a GlmMoeDsa-shaped config prices a separate index_k_norm, but that family "
        "folds it into the same fused_norm_rope launch as the latent norms")

    def norm_bytes(ns):
        return sum(n.get("bytes_per_token", 0) for n in ns
                   if n["op"] == "Elementwise"
                   and str(n.get("role") or "") in
                   ("qkv_a_layernorm", "index_k_norm"))

    # indexer_nodes() already splices the attention_block stages in, so summing it
    # ALONE is the whole layer -- adding attn_stages() again double-counts the latent
    # norm, which is how the first draft of this assertion failed.
    sep = norm_bytes(indexer_nodes(cfg))
    fus = norm_bytes(fused_nodes)
    assert sep == fus, (
        f"the separate-launch family moves {sep} B/token of normalisation and the fused "
        f"one {fus}; fusing removes a kernel launch, not the bytes it reads and writes, "
        f"so the totals must agree")


def test_the_glm_dsa_speculative_method_is_the_one_vllm_rewrites_to():
    """model_type glm_moe_dsa resolves to deepseek_mtp, not glm4_moe_mtp.

    vLLM lists it in one branch with the DeepSeek types --
    `if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
    hf_config.model_type = "deepseek_mtp"` (config/speculative.py:678-685) -- and
    glm4_moe_mtp is set on a different branch (:780) for a different family. Naming the
    wrong one records a draft method no engine would select for these checkpoints.

    Keyed off the config's declared model_type rather than its architecture, because that
    is the field vLLM's resolution actually reads."""
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if t.get("model_type") != "glm_moe_dsa":
            continue
        g = graph_of(d)
        spec = g.get("speculator")
        if spec is None:
            continue
        assert spec["method"] == "deepseek_mtp", (
            f"{d.name}: declares model_type glm_moe_dsa and draft method "
            f"{spec['method']!r}; vLLM rewrites that model_type to deepseek_mtp")
        checked += 1
    assert checked, "no glm_moe_dsa model with a speculator in the catalog"


def test_a_fused_index_norm_is_not_also_priced_as_its_own_node():
    """Where vLLM folds the indexer k-norm into the latent-norm launch, it is one node.

    deepseek_v32 calls fused_norm_rope once per layer, unconditionally on the shared code
    path, and passes the q norm, the kv norm, the rope cache AND the indexer's k-norm
    weight/bias/eps into that single launch (deepseek_v32/attention.py:378-392). Emitting
    a separate index_k_norm there charges a normalisation kernel the family does not
    launch -- its traffic is already inside the fused latent norm.

    glm5next is the negative case: it fuses the two LATENT norms (fuse_qkv_rmsnorm=True,
    glm5next/common/attention.py:604) and runs _fused_indexer_k_norm as its own call
    (:352), so it keeps the node. That model arrives with the branch consuming this one,
    so the negative half is asserted there and here only by the layout table's default.

    Named by architecture rather than read from MLA_LAYOUTS: a test that imports the
    table it checks moves with it and catches nothing."""
    FUSES_INDEX_NORM = {"GlmMoeDsaForCausalLM"}
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        arches = (list(config_of(d).get("architectures") or ())
                  + list(t.get("architectures") or ()))
        if not any(a in FUSES_INDEX_NORM for a in arches):
            continue
        g = graph_of(d)
        for k in g["layer_kinds"]:
            if not any(x.get("role") == "block_index_scores" for x in k["nodes"]):
                continue
            assert not any(x.get("role") == "index_k_norm" for x in k["nodes"]), (
                f"{d.name}/{k['id']}: prices a separate index_k_norm, but this family "
                f"folds it into the same fused_norm_rope launch as the latent norms")
            # The latent norm it was folded into must still be there AND must carry the
            # folded width. fusion removes a LAUNCH, not the HBM traffic: that kernel
            # still reads, normalises and writes index_k, so suppressing the separate
            # node without widening this one deletes bytes the kernel really moves.
            fused = next((x for x in k["nodes"]
                          if x.get("role") == "qkv_a_layernorm"), None)
            assert fused is not None, (
                f"{d.name}/{k['id']}: the fused latent norm is absent, so the folded "
                f"k-norm traffic is unaccounted for")
            act = 2  # bf16; these configs declare bfloat16
            want = 2 * (int(t["q_lora_rank"]) + int(t["kv_lora_rank"])
                        + int(t["index_head_dim"])) * act
            assert fused["bytes_per_token"] == want, (
                f"{d.name}/{k['id']}: the fused norm states "
                f"{fused['bytes_per_token']} B/token; it normalises the two latents AND "
                f"index_k in one launch, which over "
                f"{t['q_lora_rank']} + {t['kv_lora_rank']} + {t['index_head_dim']} at "
                f"{act} bytes is {want}")
            # The latents-only figure is the specific wrong answer: it is what you get
            # by suppressing the index norm's node and forgetting its bytes.
            latents_only = 2 * (int(t["q_lora_rank"]) + int(t["kv_lora_rank"])) * act
            assert fused["bytes_per_token"] != latents_only, (
                f"{d.name}/{k['id']}: the fused norm states the latents-only figure "
                f"{latents_only}; the folded index-norm traffic has been dropped rather "
                f"than moved")
            checked += 1
    assert checked, (
        "no model in the catalog fuses its index norm, so this test guards nothing")


def test_only_glm5next_recomputes_the_fp32_head_gate():
    """The FP32 head-gate matmul is one architecture's, not every indexer's.

    glm5next caches wk_weights_proj's trailing n_head rows transposed to FP32 and runs
    its own matmul against FP32 activations (common/attention.py:341-350). deepseek_v32
    does NOT: it slices index_weights straight out of the same GEMM's result and hands it
    to fused_q (deepseek_v32/attention.py:309-315, 433-446), with no FP32 copy and no
    second launch.

    So the node must appear for glm5next and NOT for GlmMoeDsaForCausalLM, which is the
    negative case -- emitting it there would invent an FP32 launch and a resident FP32
    weight copy on every indexed glm-5* layer."""
    positive = negative = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        g, raw = graph_of(d), config_of(d)
        # Named by architecture string rather than read from the deriver's predicate: a
        # test that imports what it checks moves with it and catches nothing.
        arches = (list(raw.get("architectures") or ())
                  + list(t.get("architectures") or ()))
        expected = "Glm5NextForConditionalGeneration" in arches
        for k in g["layer_kinds"]:
            if not any(n.get("role") == "block_index_scores" for n in k["nodes"]):
                continue
            gate = [n for n in k["nodes"]
                    if n.get("role") == "index_head_gate_proj"]
            if expected:
                assert len(gate) == 1, (
                    f"{d.name}/{k['id']}: this architecture recomputes the head gate "
                    f"as its own FP32 matmul and the graph prices none")
                assert gate[0].get("weight_dtype") == "fp32", (
                    f"{d.name}/{k['id']}: the head gate is cached as FP32; the global "
                    f"width understates its resident bytes")
                positive += 1
            else:
                assert not gate, (
                    f"{d.name}/{k['id']}: prices an FP32 head-gate matmul, but this "
                    f"architecture slices the gate out of wk_weights_proj instead")
                negative += 1
    assert negative, (
        "no architecture in the catalog takes the slice-and-reuse path, so the negative "
        "case this test exists for is unchecked")


def test_the_indexer_projection_vllm_leaves_unquantized_carries_the_base_width():
    """wk_weights_proj is constructed with quant_config=None, so it serves at base.

    Both implementations do this (deepseek_v32/attention.py:78-86,
    glm5next/common/attention.py:274-281), so the matrix is unquantized whatever the
    checkpoint's global width. The fp8 GLM DSA graphs would otherwise inherit fp8 for a
    tensor the runtime keeps at bf16.

    Checked only where the global width differs from the base, since that is the only
    case where an override is observable."""
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        g = graph_of(d)
        if g["global"]["weight_dtype"] == "bf16":
            continue
        for k in g["layer_kinds"]:
            wk = next((n for n in k["nodes"]
                       if n.get("role") == "index_wk_weights_proj"), None)
            if wk is None:
                continue
            assert wk.get("weight_dtype") == "bf16", (
                f"{d.name}/{k['id']}: index_wk_weights_proj inherits the global "
                f"{g['global']['weight_dtype']}, but vLLM builds it with "
                f"quant_config=None")
            checked += 1
    assert checked, (
        "no indexed model has a non-bf16 global width, so this override is unobservable "
        "and the test has nothing to check")


def test_the_indexer_norm_prices_activation_width_not_the_accumulation_dtype():
    """index_k_norm reads and writes BF16; the FP32 cast is an in-kernel temporary.

    glm5next's _fused_indexer_k_norm is `F.layer_norm(x.float(), ...).type_as(x)`
    (common/attention.py:55-59), and x is the BF16 output of the unquantized
    wk_weights_proj. Sizing the traffic from fp32 would double it."""
    # Only the families that launch the k-norm SEPARATELY have a node to check.
    # deepseek_v32 folds it into fused_norm_rope with the two latent norms
    # (attention.py:378-392), so its graphs carry none by design -- asserted by
    # test_a_fused_index_norm_is_not_also_priced_as_its_own_node.
    FUSES_INDEX_NORM = {"GlmMoeDsaForCausalLM"}
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        arches = (list(config_of(d).get("architectures") or ())
                  + list(t.get("architectures") or ()))
        if any(a in FUSES_INDEX_NORM for a in arches):
            continue
        dim = int(t["index_head_dim"])
        for k in graph_of(d)["layer_kinds"]:
            n = next((x for x in k["nodes"]
                      if x.get("role") == "index_k_norm"), None)
            if n is None:
                continue
            assert n.get("bytes_per_token") == 2 * dim * 2, (
                f"{d.name}/{k['id']}: index_k_norm states "
                f"{n.get('bytes_per_token')} B/token; a BF16 read and write over "
                f"{dim} is {2 * dim * 2}")
            checked += 1
    if not checked:
        pytest.skip("every indexed model in the catalog fuses its k-norm, so there is "
                    "no separately-launched one to size")


def kpool_models() -> list[Path]:
    """Models whose config declares a pooled (k-pool) DSA indexer."""
    return [d for d in model_dirs()
            if int(text_config(config_of(d)).get("index_kpool") or 0) > 1]


@pytest.mark.parametrize("d", kpool_models(), ids=lambda d: d.name)
def test_kpool_indexer_prices_its_own_projections_and_pooled_scan(d: Path):
    """A pooled indexer runs three projections and scores pooled candidates.

    The fused single-GEMM form was wrong three ways for this family. vLLM builds wq_b
    from the LATENT rank, not hidden; it fuses wk and weights_proj into one GEMM of
    [head_dim, n_head]; and it adds a kpool compression gate (common/attention.py
    255-282). Sourcing all of it from hidden at one fused width overstated the
    projection work by 2.3x on GLM-5.3-Flash.

    The scan is the bigger error. index_kpool pools the cache -- one K state per kpool
    tokens, which vLLM sizes as max_model_len // index_kpool -- so an unpooled scorer
    overstates the one term that grows with context by that factor.

    Every width is asserted from the config, so a config update does not silently
    invalidate the test."""
    g, t = graph_of(d), text_config(config_of(d))
    hidden = int(t["hidden_size"])
    heads = int(t["index_n_heads"])
    dim = int(t["index_head_dim"])
    kpool = int(t["index_kpool"])
    topk = int(t["index_topk"])
    q_lora = int(t.get("q_lora_rank") or 0)
    assert q_lora, "a pooled indexer sizes wq_b from the latent rank"

    indexed = [k for k in g["layer_kinds"]
               if any(n.get("role") == "block_index_scores" for n in k["nodes"])]
    assert indexed, f"{d.name}: no layer kind carries an indexer scorer"

    for kind in indexed:
        gemms = {n.get("role"): (n["n"], n["k"]) for n in kind["nodes"]
                 if n["op"] == "GEMM"}
        assert "index_qk_proj" not in gemms, (
            f"{d.name}/{kind['id']}: still carries the fused index_qk_proj, which "
            f"sources the query stage from hidden instead of the latent rank")
        assert gemms["index_wq_b"] == (heads * dim, q_lora), (
            f"{d.name}/{kind['id']}: index_wq_b is {gemms['index_wq_b']}; the config "
            f"gives {heads} * {dim} from the latent rank {q_lora}")
        assert gemms["index_wk_weights_proj"] == (dim + heads, hidden), (
            f"{d.name}/{kind['id']}: index_wk_weights_proj is "
            f"{gemms['index_wk_weights_proj']}; vLLM fuses wk and weights_proj into "
            f"one GEMM of [{dim}, {heads}] from {hidden}")
        assert gemms["index_kpool_compress_gate"] == (dim, hidden), (
            f"{d.name}/{kind['id']}: no kpool compression gate at {dim} from {hidden}")

        scorer = next(n for n in kind["nodes"]
                      if n.get("role") == "block_index_scores")
        assert scorer["compress_ratio"] == kpool, (
            f"{d.name}/{kind['id']}: the scorer states compress_ratio "
            f"{scorer.get('compress_ratio')}; index_kpool is {kpool}, and an unpooled "
            f"scan overstates the dominant long-context term by that factor")
        # index_topk must NOT be here: the scorer scans every pooled candidate to
        # DISCOVER the top-k, and the selection bounds the main read instead. Pinned
        # directly by test_kpool_scorer_is_bounded_by_pooling_and_not_by_the_topk_it_computes;
        # this catalog-driven case covers the same separation once a k-pool model is
        # committed, and asserted the opposite until it was corrected.
        assert "index_topk" not in scorer, (
            f"{d.name}/{kind['id']}: the scorer carries index_topk, bounding it by the "
            f"selection it is supposed to produce")
        main = next(n for n in kind["nodes"]
                    if n["op"] == "Attention"
                    and n.get("role") != "block_index_scores")
        assert main.get("index_topk") == topk, (
            f"{d.name}/{kind['id']}: the main latent read selects "
            f"{main.get('index_topk')}; index_topk is {topk}")
        assert scorer["d_h"] == dim and scorer["n_q"] == heads
        assert scorer["n_kv"] == 1, "the indexer scores against one pooled K stream"
        # The APE must NOT appear as a GEMM: that primitive carries FLOPs and a launch
        # the storage-only table does not run. Its read traffic is inside
        # index_kpool_compress.
        assert not any(n.get("role") == "index_kpool_compress_ape"
                       for n in kind["nodes"]), (
            f"{d.name}/{kind['id']}: the APE is emitted as a node, inventing "
            f"{2 * kpool * dim} FLOPs/token and a launch for a storage-only table")


def test_an_unpooled_indexer_prices_no_pooled_compression():
    """A config without index_kpool must not be given the pooled shape.

    Every family reaching lightning_indexer() now gets the same projections, because
    vLLM builds the same ones -- deepseek_v32 and glm5next are identical there. What
    pooling adds is the compression gate, the compression pass, and the compress_ratio
    bound on the scorer, and a config that declares no index_kpool must have none of
    them: there is nothing to pool, so a ratio would bound a scan that reads every
    token."""
    checked = []
    for path in model_dirs():
        t = text_config(config_of(path))
        if "index_topk" not in t or int(t.get("index_kpool") or 0) > 1:
            continue
        # A family whose cache is itself compressed scans that compressed stream, so
        # its scorer is bounded by the layer's own compress_ratios entry rather than by
        # a pooling the indexer adds. DeepSeek-V4.1 is the case: it still must price no
        # k-pool work, but its scorer's ratio is pinned against its KV source by
        # test_deepseek_v41_each_layer_pays_for_what_its_source_vectors_assign.
        # DeepSeek-V4 likewise: its indexer compresses its own keys at ratio 4
        # (deepseek_v4/attention.py:1022-1031), pinned by
        # test_deepseek_v4_each_layer_pays_for_what_its_ratio_assigns.
        stream_compressed = bool(t.get("compress_ratios"))
        g = graph_of(path)
        for kind in g["layer_kinds"]:
            roles = {n.get("role") for n in kind["nodes"]}
            if "block_index_scores" not in roles:
                continue
            checked.append(f"{path.name}/{kind['id']}")
            assert "index_kpool_compress_gate" not in roles, (
                f"{path.name}/{kind['id']}: declares no index_kpool yet prices a "
                f"pooled compression gate")
            assert "index_kpool_compress" not in roles, (
                f"{path.name}/{kind['id']}: declares no index_kpool yet prices the "
                f"pooling pass")
            if stream_compressed:
                continue
            scorer = next(n for n in kind["nodes"]
                          if n.get("role") == "block_index_scores")
            assert not scorer.get("compress_ratio"), (
                f"{path.name}/{kind['id']}: an unpooled scorer states a compress_ratio, "
                f"so it prices a pooled scan the config never declares")
    assert checked, (
        "no unpooled indexer in the catalog; this test guards their shape and has "
        "nothing to check")


def test_indexer_runs_only_on_the_layers_the_config_indexes():
    """A DSA config may skip the indexer, and the expanded stack must show that.

    vLLM computes a per-layer skip_topk and builds the indexer only when it is false
    (models/deepseek_v32/attention.py:166-200, which is where GlmMoeDsaForCausalLM is
    served -- its __init__ aliases GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM):

        skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0

    glm-5.2, glm-5.2-fp8 and glm-5.3 declare freq 4 and offset 3 over 78 layers, which
    indexes 21. Charging all 78 for the scorer and its projections overprices the
    dominant long-context term by 78/21.

    Asserted on the EXPANDED stack, by exact count and exact placement, because a
    per-kind check cannot see how many layers use each kind. The defaults matter too:
    glm-5 declares neither field, so freq 1 and offset 2 index every layer, and this
    test must confirm that case is unchanged rather than silently reduced."""
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        if t.get("compress_ratios"):
            # DeepSeek-V4 places its indexer by its per-layer compression-ratio vector
            # instead -- only the ratio-4 layers build one -- which is a different rule
            # from skip_topk and is pinned by its own test. Keyed off the vector rather
            # than the model name so a future family declaring one is also excluded.
            continue
        if t.get("layer_types"):
            # A hybrid states which layers have latent attention at all: GLM-5.3-Flash
            # is KDA on 34 of 45 layers, and a linear-attention layer reads no KV cache
            # to index. skip_topk selects among the layers that COULD index, so on a
            # hybrid the two rules compose and this test's expansion does not describe
            # the result. Those models' placement is pinned against layer_types by
            # their own handler's tests.
            continue
        g = graph_of(d)
        layers = int(t.get("num_hidden_layers") or 0)
        assert layers, f"{d.name}: no layer count to expand against"
        kinds = {k["id"]: k for k in g["layer_kinds"]}
        scores = {kid: any(n.get("role") == "block_index_scores" for n in k["nodes"])
                  for kid, k in kinds.items()}
        seq = layer_sequence(g["stack"])
        assert len(seq) == layers, (
            f"{d.name}: stack expands to {len(seq)} layers for {layers} declared")
        got = [i for i, kid in enumerate(seq) if scores[kid]]

        pattern = t.get("index_topk_pattern")
        if pattern is not None:
            want = [i for i in range(layers) if pattern[i] != "S"]
        else:
            freq = int(t.get("index_topk_freq") or 1)
            offset = t.get("index_skip_topk_offset")
            offset = 2 if offset is None else int(offset)
            want = [i for i in range(layers)
                    if max(i - offset + 1, 0) % freq == 0]
        assert got == want, (
            f"{d.name}: scorers at {len(got)} layers {got[:8]}...; the config's "
            f"skip_topk rule gives {len(want)} at {want[:8]}...")

        # A layer that skips the indexer omits the SCORING work, not the bounded read.
        # vLLM constructs the shared MLA module with use_sparse=True on every layer and
        # passes the common topk_indices_buffer (deepseek_v32/attention.py:202-218);
        # mla_attention.py:564-566 says why: "Sparse MLA reads top-k indices from a
        # shared buffer. Pass it explicitly so backbone 'skip' layers (indexer=None)
        # still find it." Making such a layer plain mla would price its main KV read as
        # unbounded, which is a far larger error than the scorer it correctly drops.
        for kid in {k for k, has in scores.items() if not has} & set(seq):
            main = next(n for n in kinds[kid]["nodes"]
                        if n["op"] == "Attention"
                        and n.get("role") != "block_index_scores")
            assert main.get("kind") == "sparse_mla", (
                f"{d.name}/{kid}: a skipped layer dropped the sparse-latent kind; it "
                f"still reads a top-k from the shared buffer")
            assert main.get("index_topk") == int(t["index_topk"]), (
                f"{d.name}/{kid}: a skipped layer records index_topk "
                f"{main.get('index_topk')}; it reads the same shared selection as every "
                f"other layer, only without computing it")
            assert not any(n.get("role") == "index_head_gate_proj"
                           for n in kinds[kid]["nodes"]), (
                f"{d.name}/{kid}: a skipped layer prices indexer projections")
        checked += 1
    assert checked, "no indexed model in the catalog to check"


def test_the_draft_module_indexes_even_where_the_base_rule_skips():
    """An MTP layer always builds an indexer, whatever the frequency says.

    vLLM evaluates the draft at layer_id == num_hidden_layers and then constructs under
    `if not skip_topk or is_mtp_layer` (deepseek_v32/attention.py:179-200), so the
    is_mtp_layer term forces it. On a frequency-4 config the last DECLARED base kind is
    the UNINDEXED one, so derive()'s mirror-a-target-layer fallback made the draft stack
    unindexed -- underpricing the draft indexer on every speculative step.

    Asserted on the draft layer's own nodes rather than on its kind id, so renaming a
    kind leaves this passing and dropping the scorer fails it."""
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        if t.get("compress_ratios"):
            # DeepSeek-V4's draft is uncompressed by its OWN rule -- vLLM falls to
            # compress_ratio 1 past num_hidden_layers, and a ratio-1 layer carries no
            # indexer. That is pinned by
            # test_v4_draft_module_is_an_uncompressed_layer_of_its_own; the skip_topk
            # override this test is about does not apply to it.
            continue
        g = graph_of(d)
        spec = g.get("speculator")
        if not spec:
            continue
        kinds = {k["id"]: k for k in g["layer_kinds"]}
        base_scoring = {kid for kid, k in kinds.items()
                        if kid in set(layer_sequence(g["stack"]))
                        and any(n.get("role") == "block_index_scores"
                                for n in k["nodes"])}
        for lid in layer_sequence(spec["stack"]):
            nodes = kinds[lid]["nodes"]
            # BEHAVIOURAL: the draft pass must pay for scoring. Asserted on the work in
            # the layer, not on which kind id it happens to reuse.
            assert any(n.get("role") == "block_index_scores" for n in nodes), (
                f"{d.name}: the draft layer {lid!r} prices no indexer scorer, but vLLM "
                f"forces one on the MTP layer regardless of the skip pattern")
            assert any(n.get("role") == "index_wq_b" for n in nodes), (
                f"{d.name}: the draft layer {lid!r} has a scorer with no projection "
                f"to feed it")
            # METAMORPHIC: the draft's INDEXER work must cost exactly what an indexed
            # base layer's does. vLLM builds the draft from the same decoder-layer code
            # with the indexer forced on, so the scoring pass and the projections that
            # feed it cannot differ.
            #
            # Restricted to the indexer nodes rather than the whole attention side,
            # because a draft layer legitimately differs elsewhere and those
            # differences are real, not errors: glm5next's MTP module adds an eh_proj
            # over the concatenated [hidden, embedding] pair and three norms of its own,
            # and carries NO mHC, since vLLM gates that on `not is_mtp_layer`
            # (glm5next/common/model.py:439). A relation over the full attention prefix
            # would flag those as faults.
            if base_scoring:
                ref = sorted(base_scoring)[0]
                def indexer_cost(ns):
                    return collections.Counter(
                        node_cost(n) for n in ns
                        if str(n.get("role") or "").startswith("index_")
                        or n.get("role") == "block_index_scores")
                draft_cost = indexer_cost(nodes)
                ref_cost = indexer_cost(kinds[ref]["nodes"])
                assert draft_cost == ref_cost, (
                    f"{d.name}: the draft layer {lid!r} and the indexed base layer "
                    f"{ref!r} price their INDEXER differently; vLLM builds both from "
                    f"the same decoder layer with it forced on. Only in draft: "
                    f"{sorted(draft_cost - ref_cost)[:2]}; only in base: "
                    f"{sorted(ref_cost - draft_cost)[:2]}")
            checked += 1
    assert checked, "no indexed model with a speculator to check"


def test_kimi_attention_weights_keep_the_width_quantization_skips():
    """A compressed-tensors ignore list naming self_attn keeps those weights at base.

    kimi-k2.5 is globally int4 and kimi-k3 mxfp4, and both name `re:.*self_attn.*` in
    their compressed-tensors ignore list, so every attention projection stays
    unquantized at runtime. Pricing them at the global width charges 4 bits for tensors
    the runtime never quantizes.

    This covers the KDA mixer too, not just the latent path: vLLM builds Kimi-K3's KDA
    as `self.self_attn` with prefix "{...}.self_attn" (kimi_k3/nvidia/model.py:888-891),
    so the same pattern covers it. The routed experts are NOT in the ignore list and
    must keep the quantized global width -- that is the half of the distinction a blanket
    override would destroy."""
    checked = 0
    for name in ("kimi-k2.5", "kimi-k3"):
        d = ROOT / "models" / name
        if not d.is_dir():
            continue
        g, t = graph_of(d), text_config(config_of(d))
        quant = (t.get("quantization_config")
                 or config_of(d).get("quantization_config") or {})
        ignore = [str(x) for x in (quant.get("ignore") or ())]
        assert any("self_attn" in x for x in ignore), (
            f"{name}: this test is about a config excluding self_attn; the ignore list "
            f"is now {ignore}")
        base = str(t.get("dtype") or "")
        assert base.startswith("bfloat16"), (
            f"{name}: expected a bfloat16 base width, got {base!r}")
        glob = g["global"]["weight_dtype"]
        assert glob != "bf16", (
            f"{name}: the global width is already bf16, so this test cannot tell an "
            f"override from the default")

        attention_roles = {
            "qkv_proj", "qkv_a_proj", "kv_a_proj", "q_proj", "q_b_proj", "kv_b_proj",
            "o_proj", "index_wq_b", "index_wk_weights_proj", "index_head_gate_proj",
        }
        seen_attn = seen_expert = 0
        for k in g["layer_kinds"]:
            for n in k["nodes"]:
                role = str(n.get("role") or "")
                if n["op"] not in ("GEMM", "GroupedGEMM"):
                    continue
                if role in attention_roles or role.startswith("kda_"):
                    seen_attn += 1
                    assert n.get("weight_dtype") == "bf16", (
                        f"{name}/{k['id']}: {role} is priced at the global {glob}; the "
                        f"ignore list leaves it unquantized at the bf16 base")
                elif "expert" in role:
                    seen_expert += 1
                    assert n.get("weight_dtype") != "bf16", (
                        f"{name}/{k['id']}: {role} is overridden to bf16, but the "
                        f"experts are what this checkpoint actually quantizes")
        assert seen_attn, f"{name}: no attention projections found to check"
        assert seen_expert, f"{name}: no expert GEMMs found; the negative half is unchecked"
        checked += 1
    if not checked:
        pytest.skip("no Kimi models in the catalog")


def mla_models() -> list[Path]:
    """The models whose config declares a compressed latent KV."""
    return [d for d in model_dirs()
            if text_config(config_of(d)).get("kv_lora_rank")]


@pytest.mark.parametrize("d", mla_models(), ids=lambda d: d.name)
def test_mla_prices_its_low_rank_stages_not_one_fused_projection(d: Path):
    """A latent layer runs a chain of low-rank GEMMs, and each must be priced.

    An MLA layer does not project hidden straight to the query width. It compresses to
    kv_lora_rank (and, where declared, q_lora_rank), normalises, then up-projects: the
    cached latent through kv_b_proj into per-head nope + value widths, and the query
    through q_b_proj. Collapsing that into one query-width `qkv_proj` priced the wrong
    number of GEMMs at the wrong widths -- and for GLM-5.3-Flash it OVERSTATED the
    parameter count by 1.33x while coincidentally matching on bytes, because kv_b_proj
    is BF16 where its siblings are FP8. Byte traffic alone cannot catch this, so every
    stage is asserted by shape.

    Each width comes from the config, so this survives a config update."""
    g, t = graph_of(d), text_config(config_of(d))
    hidden = int(t["hidden_size"])
    nq = int(t["num_attention_heads"])
    nope = int(t["qk_nope_head_dim"])
    rope = int(t["qk_rope_head_dim"])
    v = int(t.get("v_head_dim") or nope)
    kv_lora = int(t["kv_lora_rank"])
    q_lora = int(t.get("q_lora_rank") or 0)

    latent = [k for k in g["layer_kinds"]
              if any(n.get("role") == "kv_b_proj" for n in k["nodes"])]
    assert latent, (
        f"{d.name}: no layer kind prices a kv_b_proj, so the cached latent is never "
        f"up-projected; the layer is still one fused qkv_proj")

    for kind in latent:
        gemms = {n.get("role"): (n["n"], n["k"]) for n in kind["nodes"]
                 if n["op"] == "GEMM"}
        roles = [n.get("role") for n in kind["nodes"]]

        # Whether the A-projections and latent norms are ONE launch or two is
        # per-architecture. Named here by ARCHITECTURE STRING rather than read from the
        # deriver's own table: a test that imports the table it is checking moves with
        # it, so flipping an entry would flip the assertion too and catch nothing.
        #
        #   generic DeepseekV2Attention -- separate q_a_proj, two norm calls
        #     (deepseek_v2.py:493, 598, 605, 608). Serves DeepseekV2ForCausalLM and
        #     DeepseekV3ForCausalLM, which registry.py:89-90 both send to deepseek_v2;
        #     kimi-k2.5's text_config declares DeepseekV3ForCausalLM.
        #   fused -- one A-projection and one norm over both latents. deepseek_v32
        #     (registry.py:118 for GlmMoeDsaForCausalLM), kimi_k3, glm5next.
        # DeepseekV2DecoderLayer selects DeepseekV2MLAAttention whenever use_mla is
        # true (deepseek_v2.py:1286-1291), and every latent model here takes that class
        # -- the graphs assert it themselves by pricing kind=mla with n_kv=1. That class
        # builds ONE fused A-projection (:1050-1057) with SEPARATE norms (:1081, :1097),
        # since the wrapper never sets fuse_qkv_rmsnorm. So fused_a holds for every
        # q-lora family, and only these opt into the norm fusion.
        FUSED_NORM_ARCHES = {
            "GlmMoeDsaForCausalLM", "DeepseekV4ForCausalLM",
            "KimiK3ForConditionalGeneration", "KimiLinearForCausalLM",
            "Glm5NextForConditionalGeneration",
        }
        arches = (list(config_of(d).get("architectures") or ())
                  + list(t.get("architectures") or ()))
        layout = {"fused_a": True,
                  "fused_norm": any(a in FUSED_NORM_ARCHES for a in arches)}

        assert "qkv_proj" not in gemms, (
            f"{d.name}/{kind['id']}: still carries the fused qkv_proj alongside the "
            f"low-rank stages, so the projection work is counted twice")

        # The up-projection of the cache: kv_lora_rank -> heads * (nope + v).
        assert gemms["kv_b_proj"] == (nq * (nope + v), kv_lora), (
            f"{d.name}/{kind['id']}: kv_b_proj is {gemms['kv_b_proj']}; the config "
            f"gives {nq} * ({nope} + {v}) from {kv_lora}")
        if q_lora:
            # vLLM fuses the two A-stages into one GEMM over
            # [q_lora_rank, kv_lora_rank + qk_rope_head_dim].
            # An output gate rides in this same launch where the config declares one:
            # vLLM swaps the fused A-projection for KimiK3MergedQKVGateLinear and splits
            # [qkv_a_rows, num_local_heads * v_head_dim] off the single result
            # (kimi_k3/nvidia/mla.py:218-228, 574-576). kimi-k3 declares it, which adds
            # num_heads * v_head_dim rows; no other committed config does.
            gate_rows = nq * v if t.get("mla_use_output_gate") else 0
            if layout["fused_a"]:
                assert gemms["qkv_a_proj"] == (
                    q_lora + kv_lora + rope + gate_rows, hidden), (
                    f"{d.name}/{kind['id']}: qkv_a_proj is {gemms['qkv_a_proj']}; the "
                    f"config gives {q_lora} + {kv_lora} + {rope} + {gate_rows} (gate) "
                    f"from {hidden}")
            else:
                # Two launches, as deepseek_v2.py:493 and :605 build them.
                assert gemms["q_a_proj"] == (q_lora, hidden), (
                    f"{d.name}/{kind['id']}: q_a_proj is {gemms.get('q_a_proj')}; this "
                    f"architecture projects {q_lora} from {hidden} on its own")
                assert gemms["kv_a_proj"] == (kv_lora + rope, hidden)
                assert "qkv_a_proj" not in gemms, (
                    f"{d.name}/{kind['id']}: prices a fused A-projection, but this "
                    f"architecture builds the two separately")
                assert not gate_rows, (
                    f"{d.name}/{kind['id']}: an output gate is a shard of a fused "
                    f"A-projection, which this architecture does not build")
            if gate_rows:
                gate = next((n for n in kind["nodes"]
                             if n.get("role") == "mla_output_gate"), None)
                assert gate is not None and gate.get("bytes_per_token"), (
                    f"{d.name}/{kind['id']}: declares mla_use_output_gate but prices "
                    f"no sigmoid-multiply over the attention output")
            assert gemms["q_b_proj"] == (nq * (nope + rope), q_lora), (
                f"{d.name}/{kind['id']}: q_b_proj is {gemms['q_b_proj']}; the config "
                f"gives {nq} * ({nope} + {rope}) from {q_lora}")
            # The generic DeepseekV2Attention builds a separate q_a_proj and calls the
            # two norms separately (deepseek_v2.py:493, 598, 605, 608); kimi_k3 and the
            # DSA/glm5next families fuse both.
            if layout["fused_norm"]:
                latent_norms = [next((n for n in kind["nodes"]
                                      if n.get("role") == "qkv_a_layernorm"), None)]
                assert latent_norms[0] is not None, (
                    f"{d.name}/{kind['id']}: this family fuses the two latent norms "
                    f"into one launch and the graph prices no qkv_a_layernorm")
                assert "q_a_layernorm" not in roles and "kv_a_layernorm" not in roles, (
                    f"{d.name}/{kind['id']}: prices separate latent norms alongside "
                    f"the fused one")
            else:
                latent_norms = [next((n for n in kind["nodes"]
                                      if n.get("role") == r), None)
                                for r in ("q_a_layernorm", "kv_a_layernorm")]
                assert all(n is not None for n in latent_norms), (
                    f"{d.name}/{kind['id']}: this family calls the two latent norms "
                    f"separately and the graph does not price both")
                assert "qkv_a_layernorm" not in roles, (
                    f"{d.name}/{kind['id']}: prices a fused latent norm, but this "
                    f"architecture makes two calls")
            # Sub-hidden traffic must be stated: these widths are the latent ranks, not
            # hidden, so an omitted bytes_per_token prices them as hidden-wide norms.
            for n in latent_norms:
                assert n.get("bytes_per_token"), (
                    f"{d.name}/{kind['id']}: {n.get('role')} states no "
                    f"bytes_per_token, so it prices as a hidden-width pass")
                assert n["bytes_per_token"] < 2 * hidden * 4, (
                    f"{d.name}/{kind['id']}: {n.get('role')} states "
                    f"{n['bytes_per_token']} B/token, not below a hidden-width norm's "
                    f"traffic; the latents are narrower than hidden")
            assert "q_proj" not in gemms, (
                f"{d.name}/{kind['id']}: declares q_lora_rank {q_lora} yet prices a "
                f"direct q_proj; the query path is low-rank")
        else:
            # No q_lora_rank: the query is projected directly at full width, which is
            # the branch vLLM takes and the shape deepseek-v2-lite really runs.
            assert gemms["q_proj"] == (nq * (nope + rope), hidden), (
                f"{d.name}/{kind['id']}: q_proj is {gemms['q_proj']}; with no "
                f"q_lora_rank it runs {nq} * ({nope} + {rope}) from {hidden}")
            assert gemms["kv_a_proj"] == (kv_lora + rope, hidden)
            # Only the KV side is compressed, so there is one latent norm to state.
            kvn = next((n for n in kind["nodes"]
                        if n.get("role") == "kv_a_layernorm"), None)
            assert kvn is not None and kvn.get("bytes_per_token"), (
                f"{d.name}/{kind['id']}: the kv latent norm states no bytes_per_token")
            assert "qkv_a_proj" not in gemms, (
                f"{d.name}/{kind['id']}: declares no q_lora_rank, so there is no "
                f"query A-stage to fuse the KV one with")

        # The output projection reads the per-head value width, not the query width.
        assert gemms["o_proj"] == (hidden, nq * v), (
            f"{d.name}/{kind['id']}: o_proj is {gemms['o_proj']}; the config gives "
            f"{hidden} from {nq} * {v}")


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_expert_shape_matches_config(d: Path):
    """Expert count and top-k must come from the config, under whichever alias."""
    g, t = graph_of(d), text_config(config_of(d))
    # The ROUTED expert node. A shared-expert node is a separate grouped GEMM where the
    # checkpoint stores the two at different widths, and it is sized by the shared count
    # rather than the routed one, so it is checked below instead.
    #
    # A draft layer may route over its own expert set: DeepSeek-V4.1's DSpark layers
    # take dspark_n_routed_experts at dspark_num_experts_per_tok
    # (deepseek_v41/nvidia/model.py:118-145), so kinds only the speculator uses are
    # checked against those fields where the config states them.
    target_kinds = set(layer_sequence(g["stack"]))
    draft_only = (set(layer_sequence(g["speculator"]["stack"])) - target_kinds
                  if g.get("speculator") else set())
    if draft_only and t.get("dspark_n_routed_experts"):
        for k in g["layer_kinds"]:
            if k["id"] not in draft_only:
                continue
            for n in k["nodes"]:
                if n["op"] == "GroupedGEMM" and n.get("role") != "shared_experts":
                    assert n["experts"] == t["dspark_n_routed_experts"]
                    assert n["top_k"] == t["dspark_num_experts_per_tok"]
        g = dict(g, layer_kinds=[k for k in g["layer_kinds"] if k["id"] not in draft_only])
    nodes = [n for k in g["layer_kinds"] for n in k["nodes"]
             if n["op"] == "GroupedGEMM" and n.get("role") != "shared_experts"]
    experts = (t.get("num_local_experts") or t.get("n_routed_experts")
               or t.get("num_experts"))
    if not nodes:
        assert not experts, "the config declares experts but the graph has no expert layer"
        return
    top_k = t.get("num_experts_per_tok") or t.get("num_experts_per_token")
    for n in nodes:
        assert n["experts"] == experts
        assert n["top_k"] == top_k
        assert n["top_k"] <= n["experts"]

    # Where a shared-expert node exists, it must carry the config's SHARED count, and
    # every token passes through all of them rather than a routed subset.
    shared_nodes = [n for k in g["layer_kinds"] for n in k["nodes"]
                    if n.get("role") == "shared_experts"]
    shared = t.get("n_shared_experts") or t.get("num_shared_experts")
    for n in shared_nodes:
        assert n["experts"] == shared, (
            f"shared-expert node states {n['experts']}, config says {shared}")
        assert n["top_k"] == n["experts"], (
            "a shared expert is dense: every token passes through all of them")


# Architecture families expected to produce identical layer structures, with the reason.
# Every other pair of models must differ: a deriver that emitted one template for
# everything would pass every per-model check above.
EXPECTED_IDENTICAL = {

    frozenset({"minimax-m2.5", "minimax-m2.7"}):
        "one architecture across two MiniMax-M2 generations. Every cost-relevant field is "
        "identical: 62 layers, hidden 3072, 256 experts at top-8, intermediate 1536, 48 "
        "query heads over 8 KV heads at head_dim 128, vocab 200064, fp8 weights. The two "
        "configs differ only in max_position_embeddings (196608 against 204800) and a "
        "dtype label, neither of which the cost model reads",

    frozenset({"glm-5.2", "glm-5.2-fp8", "glm-5.3"}):
        "one architecture across three GLM-5 generations. Every cost-relevant field is "
        "identical: 78 layers, hidden 6144, 256 experts at top-8, moe_intermediate 2048, "
        "3 leading dense layers, vocab 154880, the same sparse-MLA geometry, and the "
        "same indexer pattern — index_topk_freq 4 with index_skip_topk_offset 3, which "
        "indexes 21 of the 78 layers. They differ in quantization_config, rope theta, "
        "context length, transformers_version and — between glm-5.2 and glm-5.3 — in "
        "head_dim, 64 against 192. That last one looks like it should matter and does "
        "not: an MLA layer's per-head width is kv_lora_rank + qk_rope_head_dim, which "
        "both state as 512 + 64, so head_dim is not a term the cost model reads for this "
        "attention kind. "
        "glm-5 is NOT in this group: it declares neither index_topk_freq nor "
        "index_skip_topk_offset, so vLLM's defaults (freq 1, offset 2) index all 78 "
        "layers against these three's 21. That is a real cost difference — 57 extra "
        "scorers and their projections per forward pass — so grouping it here would "
        "assert away the very distinction the indexer-frequency derivation exists to "
        "capture",
    frozenset({"deepseek-v3", "deepseek-r1-0528"}):
        "one checkpoint geometry: DeepSeek-R1-0528 is a post-trained DeepSeek-V3 whose "
        "config.json differs from V3's only in transformers_version",
    frozenset({"kimi-k2.5", "kimi-k2.6"}):
        "one checkpoint geometry: Kimi-K2.6's config.json differs from K2.5's only in "
        "eos_token_id (163585 against 163586), which the cost model does not read",
    frozenset({"qwen3.5-397b-a17b", "qwen3.5-397b-a17b-fp8"}):
        "one architecture in two checkpoints: the FP8 release adds a quantization_config "
        "(block fp8) and differs in nothing else the cost model reads; the signature "
        "compares node shapes, not the global weight width, as for glm-5.2 and glm-5.2-fp8",
    frozenset({"llama-2-70b-hf", "llama-3.1-70b-instruct"}):
        "identical compute shapes; they differ in vocab, context length and rope theta",
}


def cost_signature(g: dict) -> tuple:
    """What a graph costs to run, as a canonical multiset.

    This is deliberately a BEHAVIOURAL signature rather than a structural one. Two graphs
    collide here only if they price identically: the same work, the same number of times,
    on the same shapes. It ignores everything a cost model does not read — role labels,
    layer-kind identifiers, node ordering within a layer, and how the stack is spelled
    (a prologue-plus-pattern and a literal sequence that expand to the same layers are the
    same deployment).

    A structural comparison would report a difference when a deriver renamed a role or
    reordered two independent nodes, neither of which changes a single step time. It would
    also report sameness for two graphs whose stacks happen to have matching patterns but
    different expansions.

    It covers the speculator's draft stack as well as the target stack, because a draft
    pass is work a deployment pays for. It does NOT cover the head: every model here has
    the same final-norm-plus-lm_head shape, and its cost is driven by vocab_size, which
    test_distinct_architectures_cost_differently is not the right instrument for."""
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    work: collections.Counter = collections.Counter()
    for layer_id in layer_sequence(g["stack"]):
        for n in kinds[layer_id]["nodes"]:
            work[node_cost(n)] += 1

    # The draft stack a speculative decode runs, which is work the target pays for on
    # every accepted token and is NOT part of g["stack"]. Tagged as a separate component
    # rather than folded into the counter above, so a draft layer and a target layer of
    # the same shape stay distinguishable: moving a layer between the two stacks changes
    # what a step costs. Excluded from this and verified nowhere, a wrong draft length or
    # a mirrored-from-the-wrong-kind draft stack collided with a correct graph and passed
    # every collision test.
    spec = g.get("speculator")
    if spec:
        draft: collections.Counter = collections.Counter()
        for layer_id in layer_sequence(spec["stack"]):
            for n in kinds[layer_id]["nodes"]:
                draft[node_cost(n)] += 1
        work[("speculator", spec.get("method"), spec.get("num_spec_tokens"),
              tuple(sorted(draft.items(), key=repr)))] += 1
    return tuple(sorted(work.items(), key=repr))


def node_cost(n: dict) -> tuple:
    """The parameters that change what a primitive costs, and nothing else.

    `emit` is excluded: it selects whether a node exists in a given deployment, which is a
    property of the layout rather than of the model. Those conditions are checked
    structurally instead, by test_collectives_land_where_the_parallelism_needs_them."""
    return (
        n["op"], n.get("n"), n.get("k"), n.get("n_q"), n.get("n_kv"), n.get("d_h"),
        n.get("experts"), n.get("top_k"), n.get("shared_experts"),
        n.get("shared_intermediate_size"), n.get("latent_size"), n.get("kind"),
        n.get("recurrent_kind"), n.get("state_size"), n.get("n_heads"),
        n.get("n_groups"), n.get("conv_kernel"), n.get("intermediate_size"),
        n.get("window"), n.get("index_topk"),
        # compress_ratio bounds the compressed half of a sparse latent read, and a
        # per-node weight_dtype overrides the global width for the tensors it names.
        # Both change what a node costs, and both are distinctions this suite relies on
        # elsewhere -- the C4A/C128A split and gpt-oss's expert-only MXFP4 -- so leaving
        # them out made the collision tests blind to exactly the cases they were added
        # to protect.
        n.get("compress_ratio"), n.get("weight_dtype"),
        # An Elementwise node's traffic is derived from hidden_size unless it states
        # otherwise, so bytes_per_token is the only thing distinguishing a node that
        # moves a multi-stream residual bundle from the ordinary norm it replaced.
        # Omitted here, the signature read GLM-5.3-Flash's mHC mixing as two plain
        # norms.
        n.get("bytes_per_token"),
        # What blis-schemas#55 prices beyond the shape: how a node splits over tensor
        # parallelism, the caches a read holds and their records, a launch that writes a
        # cache record only every per_tokens tokens, work proportional to the context, a
        # gather's table and where it lives, and a node that reuses another's weights.
        # Each changes a step time or a residency, so a signature blind to them would
        # call a FlashMLA read over a 584 B record and one over a 528 B record the same.
        n.get("tp"), frozen(n.get("caches")), n.get("per_tokens"), n.get("writes_cache"),
        frozen(n.get("context")), n.get("rows"), n.get("row_bytes"),
        n.get("rows_per_token"), n.get("out_bytes_per_token"), n.get("residency"),
        n.get("shard"), n.get("aliases"), frozen(n.get("weight_scale")),
    )


def frozen(v):
    """A hashable, order-preserving copy of a node's nested field."""
    if isinstance(v, dict):
        return tuple(sorted((k, frozen(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(frozen(x) for x in v)
    return v


def test_distinct_architectures_cost_differently():
    """Only the documented groups may price identically.

    Without this, a deriver bug that applied one architecture's template to every model
    would leave every other test passing: the layer counts would still match, the
    parameters would still be positive, and the graphs would still validate."""
    groups = collections.defaultdict(list)
    for d in model_dirs():
        groups[cost_signature(graph_of(d))].append(d.name)

    unexplained = []
    for names in groups.values():
        if len(names) < 2:
            continue
        if frozenset(names) not in EXPECTED_IDENTICAL:
            unexplained.append(sorted(names))
    assert not unexplained, (
        "these models price identically and are not a documented family; "
        f"one of each group is likely derived from the wrong template: {unexplained}")


def test_expected_identical_groups_still_cost_the_same():
    """The exemptions above must stay true.

    If two models in a documented group diverge, either a config changed or the deriver
    started treating them differently — both worth knowing."""
    by_name = {d.name: cost_signature(graph_of(d)) for d in model_dirs()}
    for group, reason in EXPECTED_IDENTICAL.items():
        present = [n for n in group if n in by_name]
        if len(present) < 2:
            continue
        first = by_name[present[0]]
        for other in present[1:]:
            assert by_name[other] == first, (
                f"{present[0]} and {other} were expected to price identically "
                f"({reason}) and no longer do")


def test_weight_dtypes_vary_across_the_catalog():
    """A deriver that defaulted every dtype would pass the per-model checks.

    Reads node overrides as well as the global, because a checkpoint that states its
    mixed-precision layout explicitly keeps the BASE dtype global and carries the
    quantized widths on the nodes they apply to. Looking only at the global would miss
    nvfp4 entirely now that the Nemotron variants express it that way."""
    seen = {graph_of(d)["global"]["weight_dtype"] for d in model_dirs()}
    for d in model_dirs():
        g = graph_of(d)
        seen |= {n["weight_dtype"] for k in g["layer_kinds"] for n in k["nodes"]
                 if n.get("weight_dtype")}
    assert len(seen) >= 4, f"only {seen} appear; a default is suspected"
    # mxfp4 and nvfp4 are distinct formats, not spellings of one another, and both are in
    # the catalog. Collapsing them would misprice one.
    assert {"mxfp4", "nvfp4"} <= seen, f"expected both 4-bit formats, got {seen}"


# --- The signature's own behaviour -------------------------------------------------
# cost_signature underpins the two collision tests above, so its sensitivity is worth
# establishing rather than assuming. It must ignore changes that cost nothing and notice
# changes that cost something; a signature that failed the first half would report
# differences a cost model cannot observe, and one that failed the second half would let a
# wrong graph pass.

def a_reference_graph() -> dict:
    return graph_of(next(d for d in model_dirs() if d.name == "llama-3.1-8b-instruct"))


@pytest.mark.parametrize("mutate,description", [
    (lambda g: [n.update(role="renamed") for k in g["layer_kinds"] for n in k["nodes"]],
     "renaming every role"),
    (lambda g: g["layer_kinds"][0]["nodes"].reverse(),
     "reordering nodes within a layer"),
    (lambda g: g.update(head=[]),
     "dropping the head, which is priced separately from the stack"),
])
def test_signature_ignores_changes_that_cost_nothing(mutate, description):
    import copy

    g = a_reference_graph()
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    mutate(variant)
    assert cost_signature(variant) == before, (
        f"{description} changed the cost signature, but a cost model cannot observe it")


def test_signature_ignores_how_the_stack_is_spelled():
    """A prologue-plus-pattern and a literal sequence that expand alike are one deployment.

    The deriver compresses a uniform run into a pattern where it can. That is a storage
    choice, so two graphs differing only in the compression must compare equal."""
    import copy

    g = a_reference_graph()
    literal = copy.deepcopy(g)
    literal["stack"] = {"prologue": layer_sequence(g["stack"])}
    assert cost_signature(literal) == cost_signature(g)


def test_signature_ignores_layer_kind_identifiers():
    import copy

    g = a_reference_graph()
    renamed = copy.deepcopy(g)
    old = renamed["layer_kinds"][0]["id"]
    renamed["layer_kinds"][0]["id"] = "some-other-name"
    for field in ("prologue", "pattern", "epilogue"):
        if field in renamed["stack"]:
            renamed["stack"][field] = ["some-other-name" if x == old else x
                                       for x in renamed["stack"][field]]
    assert cost_signature(renamed) == cost_signature(g)


@pytest.mark.parametrize("mutate,description", [
    (lambda g: _widen_first_gemm(g), "widening one projection"),
    (lambda g: g["stack"].update(repeat=g["stack"]["repeat"] - 1), "dropping a layer"),
    (lambda g: _change_kv_heads(g), "changing the KV head count"),
    (lambda g: _narrow_first_gemm_dtype(g), "narrowing one node's weight dtype"),
])
def test_signature_notices_changes_that_cost_something(mutate, description):
    import copy

    g = a_reference_graph()
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    mutate(variant)
    assert cost_signature(variant) != before, (
        f"{description} left the cost signature unchanged, so the collision tests would "
        f"not catch it")


def _widen_first_gemm(g: dict) -> None:
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "GEMM":
                n["n"] += 64
                return
    pytest.skip("no GEMM node to widen")


def _narrow_first_gemm_dtype(g: dict) -> None:
    """A per-node dtype override changes that node's weight bytes fourfold."""
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "GEMM":
                n["weight_dtype"] = "mxfp4"
                return
    pytest.skip("no GEMM node")


def _change_kv_heads(g: dict) -> None:
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "Attention":
                n["n_kv"] = max(1, n["n_kv"] // 2)
                return
    pytest.skip("no attention node")


@pytest.mark.parametrize("model,field,value,description", [
    ("deepseek-v4-pro", "compress_ratio", 4,
     "retuning a compressed stream's ratio"),
    ("gpt-oss-120b", "weight_dtype", "bf16",
     "widening a node's weight dtype"),
])
def test_signature_notices_a_changed_node_parameter(model, field, value, description):
    """cost_signature must see the two fields its contract most depends on.

    Both were omitted from node_cost: mutating either left the signature unchanged, so
    the collision tests were blind to exactly the distinctions this suite relies on --
    the C4A/C128A compression split, and gpt-oss's expert-only MXFP4 scope. A signature
    that cannot see them would let a graph with the wrong one collide with a correct
    graph and pass."""
    import copy

    d = ROOT / "models" / model
    if not d.is_dir():
        pytest.skip(f"{model} is not in the catalog")
    g = graph_of(d)
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    changed = False
    for k in variant["layer_kinds"]:
        for n in k["nodes"]:
            if field in n and n[field] != value:
                n[field] = value
                changed = True
    assert changed, f"no node in {model} carries {field} to mutate"
    assert cost_signature(variant) != before, (
        f"{description} left the cost signature unchanged, so a graph with the wrong "
        f"{field} would pass the collision tests")


# --- Fidelity to the pinned implementations ---------------------------------------
# The gates in this repo establish SELF-CONSISTENCY: a graph re-derives from its config,
# validates against the schema, and prices distinctly from its siblings. None of that
# checks fidelity to the implementation the deriver claims to follow, which is where
# review found five defects. These tests assert against the pinned upstream behaviour,
# so a regeneration cannot quietly restore the old shape.


def test_v4_every_layer_retains_its_sliding_window():
    """The read set is the UNION of the window and the compressed stream.

    vLLM v0.31.0 gives every layer its own sliding-window cache, MTP included
    (deepseek_v4/attention.py:345-352), and decode passes it as the main cache with the
    compressed stream as the extra one (nvidia/flashmla.py:248-264); prefill merges the
    two position sets with combine_topk_swa_indices (:350-380). The
    deriver previously replaced the window with a fixed topk//ratio -- 256 for ratio 4
    and 8 for ratio 128 -- the latter smaller than the 128-token window the config
    declares, so a C128A layer claimed to read 8 tokens where the implementation reads
    at least 128."""
    g, t = _v4()
    declared = t["sliding_window"]
    assert declared > 0, "this test is about a config that declares a window"
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
    assert main, "no attention nodes"
    for n in main:
        assert n["window"] == declared, (
            f"a layer states window {n['window']} where the config declares "
            f"{declared}; the window is retained at every compression ratio")


def test_v4_compressed_bound_is_a_ratio_not_a_token_count():
    """A ratio-128 layer's compressed term is context/128, which no constant expresses.

    The C128A path reads a positional run -- num_compressed = (position + 1) //
    compress_ratio (deepseek_v4/sparse_mla.py:403-437) -- rather than selecting a
    ranked top-k. A graph
    storing a fixed token count there is wrong at every context but one."""
    g, t = _v4()
    ratios = set(t["compress_ratios"][: t["num_hidden_layers"]])
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    seq = layer_sequence(g["stack"])
    for idx, ratio in enumerate(t["compress_ratios"][: t["num_hidden_layers"]]):
        n = next(x for x in kinds[seq[idx]]["nodes"]
                 if x["op"] == "Attention" and x.get("role") != "block_index_scores")
        assert n.get("compress_ratio") == ratio, (
            f"layer {idx} has compress_ratio {n.get('compress_ratio')}, config says "
            f"{ratio}")
    # Only the selecting ratio carries a top-k; the other selects nothing at all.
    for ratio in ratios:
        idx = t["compress_ratios"].index(ratio)
        n = next(x for x in kinds[seq[idx]]["nodes"]
                 if x["op"] == "Attention" and x.get("role") != "block_index_scores")
        if ratio == 4:
            assert n.get("index_topk") == t["index_topk"], (
                "a ratio-4 layer selects up to index_topk compressed positions")
        else:
            assert not n.get("index_topk"), (
                f"a ratio-{ratio} layer reads a positional run and selects no top-k, "
                f"so index_topk {n.get('index_topk')} describes a selection it never "
                f"makes")


def test_v4_only_ratio_4_layers_have_an_indexer():
    """vLLM builds an indexer under `if self.compress_ratio == 4`, noting "Only C4A uses
    sparse attention and hence has indexer" (attention.py:297-317).

    The deriver spliced the indexer into every ratio, so all 31 C128A layers carried an
    index_qk_proj GEMM and a block_index_scores attention. The indexer is the only
    context-proportional term in the layer, so pricing it on a layer that has none
    charges a full context scan 31 times per forward pass."""
    g, t = _v4()
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    seq = layer_sequence(g["stack"])
    seen = {}
    for idx, ratio in enumerate(t["compress_ratios"][: t["num_hidden_layers"]]):
        nodes = kinds[seq[idx]]["nodes"]
        has = any(n.get("role") == "block_index_scores" for n in nodes)
        # The indexer's query projection, by shape: q_lora_rank -> index heads x dim.
        proj = any(n["op"] == "GEMM" and (n["n"], n["k"]) == (
                       t["index_n_heads"] * t["index_head_dim"], t["q_lora_rank"])
                   for n in nodes)
        assert has == proj, (
            f"layer {idx}: an indexer and its projection must appear together")
        seen.setdefault(ratio, set()).add(has)
    assert seen[4] == {True}, "every ratio-4 layer needs its indexer"
    for ratio, flags in seen.items():
        if ratio == 4:
            continue
        assert flags == {False}, (
            f"ratio-{ratio} layers carry an indexer; the implementation builds one only "
            f"for ratio 4")


def test_v4_draft_module_is_an_uncompressed_layer_of_its_own():
    """An MTP layer takes compress_ratio 1, not a target layer's ratio.

    vLLM reads compress_ratios by layer index and falls to `self.compress_ratio = 1` for
    any index at or beyond num_hidden_layers (attention.py:232-235), so the trailing 0 in
    this config's vector is never read. Ratio 1 builds no compressor, no indexer and no
    compressed cache (attention.py:298-299, 857-860): the layer reads its 128-token
    window alone (nvidia/flashmla.py:260-262). Mirroring a target kind would give the
    draft a compressed read and, from a ratio-4 kind, an indexer it does not have."""
    g, t = _v4()
    assert len(t["compress_ratios"]) == t["num_hidden_layers"] + 1, (
        "this test is about the trailing draft entry; the vector no longer has one")
    spec = g["speculator"]
    drafted = layer_sequence(spec["stack"])
    assert drafted, "no draft stack"
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    target_ids = set(layer_sequence(g["stack"]))
    for lid in drafted:
        assert lid not in target_ids, (
            f"the draft stack reuses target layer kind {lid!r}, whose compression bound "
            f"is not the draft module's")
        n = next(x for x in kinds[lid]["nodes"]
                 if x["op"] == "Attention" and x.get("role") != "block_index_scores")
        assert not n.get("compress_ratio"), (
            f"the draft layer states compress_ratio {n.get('compress_ratio')}; an MTP "
            f"layer is uncompressed (ratio 1)")
        assert not any(x.get("role") == "block_index_scores"
                       for x in kinds[lid]["nodes"]), (
            "the draft layer carries an indexer, which only a ratio-4 layer has")
        assert n["window"] == t["sliding_window"], (
            "the draft layer keeps the model's sliding window")


def test_kimi_k3_leading_layers_are_dense():
    """first_k_dense_replace makes the leading layers dense MLPs, not routed experts.

    KimiDecoderLayer builds a KimiSparseMoeBlock only when
    `layer_idx >= first_k_dense_replace` and otherwise a KimiMLP at the config's own
    intermediate_size (modeling_kimi_linear.py:893-900). The handler defined only routed
    kinds, so layer 0 was priced with a 896-expert GroupedGEMM in place of one dense
    7168 -> 2*33792 -> 7168 MLP."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    first_dense = int(t["first_k_dense_replace"])
    assert first_dense > 0, "this test is about a config that declares one"
    seq = layer_sequence(g["stack"])
    kinds = {k["id"]: k for k in g["layer_kinds"]}

    for i in range(first_dense):
        nodes = kinds[seq[i]]["nodes"]
        assert not any(n["op"] == "GroupedGEMM" for n in nodes), (
            f"layer {i} is within first_k_dense_replace {first_dense} and must run a "
            f"dense MLP, but it carries routed experts")
        widths = {n["n"] for n in nodes if n.get("role") == "mlp_gate_up"}
        assert widths == {2 * t["intermediate_size"]}, (
            f"layer {i}'s gate-and-up projection is {widths}; a dense KimiMLP is "
            f"2 * intermediate_size ({2 * t['intermediate_size']}) wide, and uses "
            f"intermediate_size rather than moe_intermediate_size")
    # And the layer just past the prologue IS routed, or the prologue swallowed too much.
    assert any(n["op"] == "GroupedGEMM"
               for n in kinds[seq[first_dense]]["nodes"]), (
        f"layer {first_dense} is the first routed layer and carries no experts")


def test_kimi_k3_routed_experts_run_at_the_latent_width():
    """routed_expert_hidden_size is the latent MoE width, and K is that, not hidden_size.

    KimiSparseMoeBlock projects hidden -> routed_expert_hidden_size before routing, runs
    each expert at that width, and projects back after
    (modeling_kimi_linear.py:776-832). The alias table recognized only the
    DeepSeek/Nemotron spelling, so every Kimi expert was priced at K=7168 instead of
    3584 -- double the expert FLOPs and dispatch volume -- and the two projections were
    missing entirely."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    latent = int(t["routed_expert_hidden_size"])
    hidden = int(t["hidden_size"])
    assert latent != hidden, "this test is about a config whose latent width differs"

    experts = [n for k in g["layer_kinds"] for n in k["nodes"]
               if n["op"] == "GroupedGEMM"]
    assert experts, "no routed expert nodes"
    for n in experts:
        assert n.get("latent_size") == latent, (
            f"expert node states latent_size {n.get('latent_size')}, config says "
            f"{latent}")

    # The projections are real work every token pays, so they are nodes.
    for k in g["layer_kinds"]:
        if not any(n["op"] == "GroupedGEMM" for n in k["nodes"]):
            continue
        down = [n for n in k["nodes"] if n.get("role") == "routed_expert_down_proj"]
        up = [n for n in k["nodes"] if n.get("role") == "routed_expert_up_proj"]
        assert len(down) == 1 and len(up) == 1, (
            f"{k['id']}: a latent MoE needs both projections, got "
            f"{len(down)} down and {len(up)} up")
        assert (down[0]["n"], down[0]["k"]) == (latent, hidden), (
            f"{k['id']}: down-projection is {down[0]['n']}x{down[0]['k']}, expected "
            f"{latent}x{hidden}")
        assert (up[0]["n"], up[0]["k"]) == (hidden, latent), (
            f"{k['id']}: up-projection is {up[0]['n']}x{up[0]['k']}, expected "
            f"{hidden}x{latent}")


def test_a_latent_moe_states_its_projections_wherever_it_appears():
    """The property across the catalog, not just for Kimi.

    Any node carrying latent_size describes a projection into and out of that width, so
    the layer must price both. Nemotron-3-Ultra states the same structure as
    moe_latent_size and had the same omission."""
    checked = 0
    for d in model_dirs():
        for k in graph_of(d)["layer_kinds"]:
            latent = {n.get("latent_size") for n in k["nodes"]
                      if n["op"] == "GroupedGEMM" and n.get("latent_size")}
            if not latent:
                continue
            assert len(latent) == 1, f"{d.name}/{k['id']}: mixed latent widths {latent}"
            width = latent.pop()
            roles = {n.get("role") for n in k["nodes"]}
            assert "routed_expert_down_proj" in roles, (
                f"{d.name}/{k['id']}: latent_size {width} with no projection into it")
            assert "routed_expert_up_proj" in roles, (
                f"{d.name}/{k['id']}: latent_size {width} with no projection out of it")
            checked += 1
    assert checked, "no latent MoE layers in the catalog to check"


def indexed_layer_rule(t: dict) -> bool:
    """Whether this config declares a rule that skips the indexer on some layers."""
    if t.get("index_topk_pattern"):
        return True
    # DeepSeek-V4.1 names its indexing layers outright. The rest read the top-k the last
    # indexer wrote into the shared buffer (deepseek_v41/attention.py:1208-1213), which
    # test_deepseek_v41_each_layer_pays_for_what_its_source_vectors_assign pins
    # layer by layer.
    if t.get("index_source_layer_ids"):
        return True
    return int(t.get("index_topk_freq") or 1) > 1


def test_a_sparse_latent_read_prices_the_selection_it_claims():
    """A graph recording index_topk must also charge for the indexer that selects.

    The representation gap behind two defects in opposite directions: DeepSeek-V4
    priced an indexer on 31 layers that have none, and the GLM-5 family recorded
    index_topk on all 78 layers and emitted no indexer at all. The indexer is the only
    context-proportional term in such a layer, so either way the dominant long-context
    cost is wrong.

    A property over the catalog rather than a per-model assertion, so a future family
    cannot reintroduce it."""
    checked = 0
    for d in model_dirs():
        g = graph_of(d)
        for k in g["layer_kinds"]:
            selects = [n for n in k["nodes"]
                       if n["op"] == "Attention" and n.get("index_topk")]
            if not selects:
                continue
            scorers = [n for n in k["nodes"] if n.get("role") == "block_index_scores"]
            if not scorers:
                # A DSA family may skip the indexer on some layers while every layer
                # still READS the shared top-k (deepseek_v32/attention.py:202-218,
                # mla_attention.py:564-566). Such a layer legitimately records
                # index_topk with no scorer, and which layers those are is pinned by
                # test_indexer_runs_only_on_the_layers_the_config_indexes.
                assert indexed_layer_rule(text_config(config_of(d))), (
                    f"{d.name}/{k['id']}: index_topk "
                    f"{selects[0]['index_topk']} asserts a top-k selection, but the "
                    f"layer has no block_index_scores node and the config declares no "
                    f"skip pattern that would explain it")
                continue
            # Two spellings, because the catalog holds two indexer shapes: the shared
            # lightning_indexer's wq_b (read from the latent rank, as both deepseek_v32
            # and glm5next build it) and DeepSeek-V4's own per-tensor wq_proj. All that
            # matters here is that SOME projection feeds the scorer.
            assert any(str(n.get("role", "")).startswith(
                ("index_wq_b", "index_wq_proj")) for n in k["nodes"]), (
                f"{d.name}/{k['id']}: an indexer with no projection to feed it")
            for n in scorers:
                assert "window" not in n, (
                    f"{d.name}/{k['id']}: the indexer scores the whole cache, so a "
                    f"window on it would make selection look bounded")
            checked += 1
    assert checked, "no selecting layers in the catalog to check"


def test_glm5_family_prices_its_dsa_indexer():
    """GLM-5 is model_type glm_moe_dsa and ships the full lightning-indexer spec.

    It is registered to handler_moe, which had no indexer path, so the selection work
    was unpriced on every layer of all four generations.

    The shapes are vLLM's: GlmMoeDsaForCausalLM is served by the deepseek_v32 module
    (its __init__ aliases GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM), whose indexer
    builds wq_b from q_lora_rank and one fused wk_weights_proj from hidden
    (models/deepseek_v32/attention.py:71-85). An earlier revision priced a single
    hidden-wide GEMM instead, which had the wrong k on the query stage.

    Only the layers the config indexes are checked, because the same config states a
    skip pattern -- see test_indexer_runs_only_on_the_layers_the_config_indexes."""
    seen = 0
    for name in ("glm-5", "glm-5.2", "glm-5.2-fp8", "glm-5.3"):
        d = ROOT / "models" / name
        if not d.is_dir():
            continue
        g, t = graph_of(d), text_config(config_of(d))
        assert t.get("index_n_heads") and t.get("index_topk"), (
            f"{name}: this test is about a config declaring an indexer")
        heads, dim = t["index_n_heads"], t["index_head_dim"]
        q_lora = int(t["q_lora_rank"])
        indexed_kinds = 0
        for k in g["layer_kinds"]:
            score = [n for n in k["nodes"] if n.get("role") == "block_index_scores"]
            if not score:
                continue
            indexed_kinds += 1
            gemms = {n.get("role"): (n["n"], n["k"]) for n in k["nodes"]
                     if n["op"] == "GEMM"}
            assert len(score) == 1, (
                f"{name}/{k['id']}: {len(score)} scorers in one layer kind")
            assert "index_qk_proj" not in gemms, (
                f"{name}/{k['id']}: still carries the fused index_qk_proj, which "
                f"sources the query stage from hidden instead of the latent rank")
            assert gemms["index_wq_b"] == (heads * dim, q_lora), (
                f"{name}/{k['id']}: index_wq_b is {gemms['index_wq_b']}; vLLM builds "
                f"{heads} * {dim} from the latent rank {q_lora}")
            assert gemms["index_wk_weights_proj"] == (dim + heads, t["hidden_size"]), (
                f"{name}/{k['id']}: index_wk_weights_proj is "
                f"{gemms['index_wk_weights_proj']}; vLLM fuses wk and weights_proj "
                f"into one GEMM of [{dim}, {heads}] from {t['hidden_size']}")
            # Shaped from the config's own indexer dimensions, not the model's heads.
            assert score[0]["n_q"] == heads
            assert score[0]["d_h"] == dim
            # The read it feeds is bounded; the scoring that selects it is not.
            assert "window" not in score[0]
        assert indexed_kinds, f"{name}: no layer kind carries an indexer"
        seen += 1
    if not seen:
        pytest.skip("no GLM-5 models in the catalog")


def test_gpt_oss_quantizes_only_its_experts():
    """modules_to_not_convert names what MXFP4 leaves alone, and it must be honored.

    gpt-oss declares quant_method mxfp4 with self_attn, the router, the embeddings and
    lm_head excluded, so MXFP4 is the expert weights only. Returning it as the global
    width priced every attention projection and the head at 4 bits instead of 16 -- a
    4x understatement on the whole non-expert parameter class, about 2.1 GiB here.

    The mirror of the expert_dtype case: there the experts are narrower than a global
    that is itself quantized; here they are the only narrow thing."""
    d = ROOT / "models" / "gpt-oss-120b"
    if not d.is_dir():
        pytest.skip("gpt-oss-120b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    quant = t.get("quantization_config") or {}
    excluded = quant.get("modules_to_not_convert")
    assert excluded, "this test is about a config that excludes modules from quantization"
    assert str(quant.get("quant_method")).lower() == "mxfp4"

    assert g["global"]["weight_dtype"] != "mxfp4", (
        "the global width must be the UNQUANTIZED base, since attention, the router, "
        "the embeddings and lm_head are all excluded from the MXFP4 scope")
    assert g["global"]["weight_dtype"] == "bf16"

    # The experts, which ARE in scope, carry the narrow width themselves.
    experts = [n for k in g["layer_kinds"] for n in k["nodes"]
               if n["op"] == "GroupedGEMM"]
    assert experts, "no expert nodes"
    for n in experts:
        assert n.get("weight_dtype") == "mxfp4", (
            f"expert node carries weight_dtype {n.get('weight_dtype')!r}; the MXFP4 "
            f"scope is exactly these weights")

    # And the excluded tensors are NOT narrowed: no override, so they take the global.
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "GEMM":
                assert n.get("weight_dtype") in (None, "bf16"), (
                    f"{k['id']}/{n.get('role')} is priced at "
                    f"{n.get('weight_dtype')!r}, but self_attn is excluded from the "
                    f"MXFP4 scope")
    for n in g["head"]:
        assert n.get("weight_dtype") in (None, "bf16"), (
            f"the head's {n.get('role')} is priced at {n.get('weight_dtype')!r}; "
            f"lm_head is excluded from the MXFP4 scope")


def test_llama4_chunked_attention_is_a_recorded_gap_not_a_silent_one():
    """The decision to price Llama-4 as unbounded must stay deliberate.

    attention_chunk_size 8192 bounds most layers in the implementation, and this
    checkpoint declares no layer_types, no moe_layers and an empty no_rope_layers, so
    the per-layer split is not derivable -- deriving one would mean inventing it. The
    handler says so. This test pins the two halves of that claim: the graph really is
    unbounded, and the config really does lack the fields that would let it not be. If a
    future checkpoint declares layer_types, this fails and the gap gets closed."""
    d = ROOT / "models" / "llama-4-scout-17b-16e-instruct-fp8-dynamic"
    if not d.is_dir():
        pytest.skip("llama-4-scout is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    assert t.get("attention_chunk_size"), "this test is about a chunked-attention config"
    assert not t.get("layer_types"), (
        "the config now declares layer_types, so the chunked/full split IS derivable "
        "and handler_moe's recorded gap should be closed rather than documented")
    assert not t.get("no_rope_layers"), (
        "no_rope_layers is now populated, so the full-attention layers are enumerated "
        "and the split should be derived")
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
    assert main
    for n in main:
        assert n["kind"] == "gqa" and "window" not in n, (
            "a bound appeared here without the config gaining the fields to derive it")
    # The gap must stay documented in the handler a reader would check.
    src = (ROOT / "scripts" / "derive_graph.py").read_text()
    assert "KNOWN GAP (Llama-4)" in src, (
        "the handler no longer records the chunked-attention gap; either close it or "
        "keep the note that explains why it is open")


BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
                   "nvfp4": 0.5, "mxfp4": 0.5, "int4": 0.5}


def weight_bytes(g: dict) -> float:
    """Total weight bytes the stack holds, honouring per-node dtype overrides."""
    base = g["global"]["weight_dtype"]
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    total = 0.0
    for lid in layer_sequence(g["stack"]):
        for n in kinds[lid]["nodes"]:
            w = BYTES_PER_PARAM[n.get("weight_dtype") or base]
            if n["op"] == "GEMM":
                total += n["n"] * n["k"] * w
            elif n["op"] == "GroupedGEMM":
                total += 3 * n["n"] * n["k"] * n["experts"] * w
    return total


def test_nemotron_nvfp4_keeps_its_declared_mixed_precision_layout():
    """An explicit quantized_layers map states a LAYOUT, not one width.

    nemotron-3-ultra-nvfp4 names 49,152 routed-expert matrices NVFP4, 192 Mamba and
    shared-expert matrices FP8, and ignores 243 more -- attention, the latent
    projections, embeddings and the head -- which stay at the declared bf16 base.
    Collapsing that to a single dominant width priced every ignored tensor at four
    bits, including the latent projections ModelOpt explicitly excludes."""
    d = ROOT / "models" / "nemotron-3-ultra-550b-a55b-nvfp4"
    if not d.is_dir():
        pytest.skip("nemotron-3-ultra-nvfp4 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    quant = t.get("quantization_config") or {}
    assert quant.get("quantized_layers"), "this test is about a config stating the map"

    assert g["global"]["weight_dtype"] == "bf16", (
        "the base dtype stays global; the quantized widths ride on their own nodes")

    by_role = {}
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] in ("GEMM", "GroupedGEMM"):
                by_role[n.get("role")] = n.get("weight_dtype")

    assert by_role.get("experts") == "nvfp4", "the routed experts are the NVFP4 tensors"
    assert by_role.get("shared_experts") == "fp8", (
        "the shared experts are stored at FP8, so they cannot ride on the routed node")
    assert by_role.get("mixer_out") == "fp8", "the Mamba projections are FP8"
    for role in ("qkv_proj", "o_proj", "routed_expert_down_proj",
                 "routed_expert_up_proj"):
        if role in by_role:
            assert by_role[role] is None, (
                f"{role} is in the checkpoint's ignore list and must keep the bf16 "
                f"base, not {by_role[role]!r}")


def test_nemotron_nvfp4_occupies_more_than_a_uniform_four_bit_reading():
    """The occupancy the layout decides, which is the cost of getting it wrong.

    Asserted through bytes rather than labels: pricing the whole checkpoint at NVFP4
    understates it, and pricing it all at bf16 overstates it, so the mixed reading must
    land strictly between."""
    d = ROOT / "models" / "nemotron-3-ultra-550b-a55b-nvfp4"
    if not d.is_dir():
        pytest.skip("nemotron-3-ultra-nvfp4 is not in the catalog")
    g = graph_of(d)
    mixed = weight_bytes(g)

    import copy
    all_four_bit = copy.deepcopy(g)
    all_four_bit["global"]["weight_dtype"] = "nvfp4"
    for k in all_four_bit["layer_kinds"]:
        for n in k["nodes"]:
            n.pop("weight_dtype", None)
    uniform = weight_bytes(all_four_bit)

    all_base = copy.deepcopy(g)
    for k in all_base["layer_kinds"]:
        for n in k["nodes"]:
            n.pop("weight_dtype", None)
    base = weight_bytes(all_base)

    assert uniform < mixed < base, (
        f"the mixed layout must sit between a uniform 4-bit reading "
        f"({uniform / 2**30:.1f} GiB) and a uniform bf16 one ({base / 2**30:.1f} GiB), "
        f"got {mixed / 2**30:.1f} GiB")


def test_v4_projections_are_low_rank_stages_not_dense_rectangles():
    """DeepSeek-V4's projection path is low-rank on both sides, with a norm between.

    The input is fused_wqa_wkv (hidden -> q_lora_rank + head_dim), then q_norm, then
    wq_b (q_lora_rank -> n_heads*head_dim). The output is a grouped wo_a then wo_b. A
    single hidden-width rectangle overstated the input side 4.1x and the output side
    2.5x -- about 39B parameters of phantom weight over 61 layers -- and the
    intervening norm means the input stages cannot be fused into one GEMM anyway."""
    g, t = _v4()
    hidden, nq = t["hidden_size"], t["num_attention_heads"]
    hd, q_lora = t["head_dim"], t["q_lora_rank"]
    o_lora, o_groups = t["o_lora_rank"], t["o_groups"]

    for k in g["layer_kinds"]:
        roles = {n.get("role"): n for n in k["nodes"] if n["op"] == "GEMM"}
        assert "qkv_proj" not in roles, (
            f"{k['id']}: a dense QKV rectangle is not this family's projection path")
        assert "o_proj" not in roles, (
            f"{k['id']}: a dense output rectangle is not this family's output path")

        fused, wq_b = roles["fused_wqa_wkv"], roles["wq_b"]
        assert (fused["n"], fused["k"]) == (q_lora + hd, hidden)
        assert (wq_b["n"], wq_b["k"]) == (nq * hd, q_lora)

        wo_a, wo_b = roles["wo_a"], roles["wo_b"]
        assert (wo_a["n"], wo_a["k"]) == (o_groups * o_lora, nq * hd // o_groups)
        assert (wo_b["n"], wo_b["k"]) == (hidden, o_groups * o_lora)

        # The norm between the two input stages, without which they would be one GEMM:
        # one launch norms both latents (attention.py:528-535), so it reads and writes
        # q_lora_rank + head_dim BF16 values, between fused_wqa_wkv and wq_b.
        order = [n.get("role") for n in k["nodes"]]
        between = k["nodes"][order.index("fused_wqa_wkv") + 1:order.index("wq_b")]
        assert [n.get("bytes_per_token") for n in between
                if n["op"] == "Elementwise"] == [2 * (q_lora + hd) * 2], (
            f"{k['id']}: no norm over both latents between the q-LoRA stages")

    # The totals, so a future refactor cannot drift while keeping the role names.
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    k0 = kinds[layer_sequence(g["stack"])[0]]
    roles = {n.get("role"): n for n in k0["nodes"] if n["op"] == "GEMM"}
    assert (roles["fused_wqa_wkv"]["n"] * roles["fused_wqa_wkv"]["k"]
            + roles["wq_b"]["n"] * roles["wq_b"]["k"]) == 115_343_360
    assert (roles["wo_a"]["n"] * roles["wo_a"]["k"]
            + roles["wo_b"]["n"] * roles["wo_b"]["k"]) == 184_549_376


def test_kimi_k3_kda_layer_prices_its_mixer_projections():
    """A RecurrentUpdate carries no N or K, so the projections must be their own nodes.

    The KDA mixer emitted only RecurrentUpdate and an AllReduce -- not even a producer
    GEMM for the reduction that followed it -- dropping 443,580,416 parameters per layer
    over 69 KDA layers, about 30.6B, plus their FLOPs.

    Shapes from the pinned KimiDeltaAttention: three separate q/k/v projections, a
    low-rank decay gate through head_dim, one beta scalar per head, a full-rank output
    gate because this config sets use_full_rank_gate, and an output projection."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    linear = t["linear_attn_config"]
    hidden = t["hidden_size"]
    heads, dim = linear["num_heads"], linear["head_dim"]
    inner = heads * dim

    kda = next(k for k in g["layer_kinds"] if k["id"].startswith("kda_"))
    roles = {n.get("role"): n for n in kda["nodes"] if n["op"] == "GEMM"}
    expected = {
        "kda_q_proj": (inner, hidden),
        "kda_k_proj": (inner, hidden),
        "kda_v_proj": (inner, hidden),
        "kda_f_a_proj": (dim, hidden),
        "kda_f_b_proj": (inner, dim),
        "kda_b_proj": (heads, hidden),
        "kda_o_proj": (hidden, inner),
    }
    if linear.get("use_full_rank_gate"):
        expected["kda_g_proj"] = (inner, hidden)
    else:
        expected["kda_g_a_proj"] = (dim, hidden)
        expected["kda_g_b_proj"] = (inner, dim)

    for role, (n, k) in expected.items():
        assert role in roles, f"the KDA mixer omits {role}"
        assert (roles[role]["n"], roles[role]["k"]) == (n, k), (
            f"{role} is {roles[role]['n']}x{roles[role]['k']}, expected {n}x{k}")

    total = sum(roles[r]["n"] * roles[r]["k"] for r in expected)
    assert total == 443_580_416, (
        f"the KDA mixer prices {total:,} projection parameters per layer, expected "
        f"443,580,416")

    # The reduction must have something producing the value it reduces.
    ops = [n["op"] for n in kda["nodes"]]
    assert ops.index("GEMM") < ops.index("AllReduce"), (
        "the mixer reduces an output no node in the layer produced")


# --- The hybrid handlers' specific structure ---------------------------------------
# The property tests above confirm the catalog HAS variety -- layer counts agree, dtypes
# differ, attention shapes come from the config. None of them asserts that a given model
# resolved to the RIGHT structure: which layers are attention and which are recurrent,
# and what state geometry the recurrent ones carry. For a handler that detects an index
# base or parses a character vector, that is the part most likely to be off by one.


def test_kimi_k3_resolves_its_full_attention_layers_one_based():
    """The off-by-one this handler detects rather than assumes.

    Kimi-K3 declares linear_attn_config.full_attn_layers as a 1-based list against a
    1-indexed layer numbering. Read as 0-based, every attention layer shifts by one and
    the model is still 93 layers of the right two kinds -- so the layer-count and
    attention-shape property tests both pass on a graph where every KDA and MLA layer
    sits one position off. The handler detects the base (a 0-based list would contain 0)
    and this pins the result: the declared index n must be the MLA layer at 0-based
    position n-1, and the position before it must be KDA."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    declared = [int(i) for i in t["linear_attn_config"]["full_attn_layers"]]
    assert 0 not in declared, (
        "this config's list is 1-based; a 0-based one would make the assertions below "
        "describe the wrong positions")
    seq = layer_sequence(g["stack"])
    assert len(seq) == t["num_hidden_layers"]

    mla = {i for i, k in enumerate(seq) if k.startswith("mla_")}
    expected = {n - 1 for n in declared}
    assert mla == expected, (
        f"MLA layers sit at 0-based {sorted(mla)[:6]}...; the config's 1-based list "
        f"{declared[:6]}... puts them at {sorted(expected)[:6]}...")
    # The concrete consequence, stated so a failure reads as the off-by-one it is:
    # index 4 declared means position 3 is attention and position 4 is not.
    assert seq[declared[0] - 1].startswith("mla_")
    assert seq[declared[0]].startswith("kda_"), (
        "the layer after the first declared attention layer is attention too, which is "
        "the signature of a 0-based read")
    assert set(seq) <= {"kda_moe", "mla_moe", "kda_dense", "mla_dense"}


def test_kimi_k3_kda_state_geometry_comes_from_the_config():
    """The recurrent node's shape, which decides its state bytes and its step cost."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    kda = [n for k in g["layer_kinds"] if k["id"] == "kda_moe"
           for n in k["nodes"] if n["op"] == "RecurrentUpdate"]
    assert len(kda) == 1, f"expected one recurrent node in a KDA layer, got {len(kda)}"
    n = kda[0]
    assert n["recurrent_kind"] == "kda"
    assert n["n_heads"] == t["num_attention_heads"]
    assert n["state_size"] == t["linear_attn_config"]["head_dim"]
    assert n["state_dtype"] == "fp32", (
        "a recurrent state carried at the weight dtype would understate its bytes")


def test_qwen3_5_resolves_full_attention_on_the_declared_vector():
    """Which layers are GDN and which are full attention, from layer_types.

    The handler reads the vector rather than full_attention_interval, so this asserts
    against the vector: a graph built from the interval would agree here only while the
    two agree, which is the point of preferring the vector."""
    d = ROOT / "models" / "qwen3.5-397b-a17b"
    if not d.is_dir():
        pytest.skip("qwen3.5-397b-a17b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    types = t["layer_types"]
    seq = layer_sequence(g["stack"])
    assert len(seq) == len(types)
    expected = ["attn_moe" if x == "full_attention" else "gdn_moe" for x in types]
    assert seq == expected, "the stack does not follow the declared layer_types vector"
    # The interval is a generator for the common case; the first full-attention layer
    # sits at the end of the first period, not the start of it.
    interval = int(t["full_attention_interval"])
    assert seq[interval - 1] == "attn_moe"
    assert seq[0] == "gdn_moe", (
        "layer 0 is full attention, which is what reading the interval as 1-based at "
        "the start of each period would produce")


def test_qwen3_5_gdn_state_geometry_follows_vllm_calculator():
    """The GDN state shape, which vLLM's own calculator defines.

    A convolutional state of width 2*k_heads*k_dim + v_heads*v_dim, and a temporal state
    whose per-head geometry is k_dim over v_heads heads. These are the numbers that decide
    the recurrent state's bytes per sequence, so a transposed head count or a key/value
    dim swap misprices every GDN layer and no other test would see it."""
    d = ROOT / "models" / "qwen3.5-397b-a17b"
    if not d.is_dir():
        pytest.skip("qwen3.5-397b-a17b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    gdn = [n for k in g["layer_kinds"] if k["id"] == "gdn_moe"
           for n in k["nodes"] if n["op"] == "RecurrentUpdate"]
    assert len(gdn) == 1
    n = gdn[0]
    k_heads, v_heads = t["linear_num_key_heads"], t["linear_num_value_heads"]
    k_dim, v_dim = t["linear_key_head_dim"], t["linear_value_head_dim"]
    assert n["recurrent_kind"] == "gdn"
    assert n["n_heads"] == v_heads, (
        f"n_heads is {n['n_heads']}; the temporal state is per VALUE head ({v_heads}), "
        f"not per key head ({k_heads})")
    assert n["state_size"] == k_dim, (
        f"state_size is {n['state_size']}; the temporal state's per-head width is the "
        f"KEY dim ({k_dim})")
    assert n["conv_kernel"] == t["linear_conv_kernel_dim"]
    assert n["intermediate_size"] == 2 * k_heads * k_dim + v_heads * v_dim, (
        "intermediate_size must be the convolution's width, "
        "2*k_heads*k_dim + v_heads*v_dim")


def test_qwen3_5_gdn_state_size_is_the_key_dim_not_the_value_dim():
    """The committed config cannot tell these two apart, so this uses one that can.

    Qwen3.5-397B declares linear_key_head_dim and linear_value_head_dim both 128, so an
    assertion against the real config passes whichever of the two the handler reads --
    verified by mutation: swapping them in the deriver leaves the whole suite green. The
    distinction is real (vLLM's gated_delta_net_state_shape takes the temporal state's
    per-head width from the KEY dim) and it decides the recurrent state's bytes, so it is
    pinned here on a config where the two differ."""
    dg = _dg()
    cfg = {
        "hidden_size": 4096, "vocab_size": 128000, "num_hidden_layers": 4,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 8192, "num_local_experts": 8, "num_experts_per_tok": 2,
        "moe_intermediate_size": 1024,
        "layer_types": ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"],
        "full_attention_interval": 4,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 64, "linear_value_head_dim": 256,
        "linear_conv_kernel_dim": 4,
    }
    kinds, _ = dg.handler_qwen3_5_moe(cfg, cfg, "synthetic")
    n = next(x for k in kinds if k["id"] == "gdn_moe"
             for x in k["nodes"] if x["op"] == "RecurrentUpdate")
    assert n["state_size"] == 64, (
        f"state_size is {n['state_size']}; the temporal state's per-head width is the "
        f"key dim (64), not the value dim (256)")
    assert n["n_heads"] == 32, (
        f"n_heads is {n['n_heads']}; the temporal state is per value head (32), not per "
        f"key head (16)")
    assert n["intermediate_size"] == 2 * 16 * 64 + 32 * 256


def test_nemotron_h_depth_and_kinds_come_from_the_layer_vector():
    """Depth from a vector, on a config that states no layer count at all.

    Nemotron-3-Ultra declares no num_hidden_layers: its depth IS the length of
    layers_block_type, and the MoE entries are MLP layers rather than separate blocks,
    which is why the vector is longer than a reader expecting a transformer depth would
    guess. This asserts both the depth and the per-position kind."""
    d = ROOT / "models" / "nemotron-3-ultra-550b-a55b-bf16"
    if not d.is_dir():
        pytest.skip("nemotron-3-ultra is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    vector = t["layers_block_type"]
    assert t.get("num_hidden_layers") is None, (
        "this test is about the config that states no layer count; it now states one")
    seq = layer_sequence(g["stack"])
    assert seq == list(vector), "the stack must be the declared vector, position by position"
    assert {k["id"] for k in g["layer_kinds"]} == set(vector)


def test_nemotron_h_parses_the_character_form_of_the_layer_vector():
    """The M/*/-/E character form, which no committed config exercises.

    handler_nemotron_h accepts either a list of names or a character string, and every
    config in the catalog today uses the list. The character branch is therefore live
    code that real data does not reach, so it is tested directly: M state-space,
    * attention, - MLP, E MoE, and an unknown character must raise rather than default."""
    dg = _dg()
    cfg = {
        "hidden_size": 4096, "vocab_size": 128000, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 8192,
        "num_local_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 1024,
        "mamba_num_heads": 64, "ssm_state_size": 128, "conv_kernel": 4,
        "mamba_head_dim": 64, "n_groups": 8,
        "hybrid_override_pattern": "M-*E",
    }
    kinds, stack = dg.handler_nemotron_h(cfg, cfg, "synthetic")
    assert layer_sequence(stack) == ["mamba", "mlp", "attention", "moe"], (
        "the character form must map M/-/*/E to state-space, MLP, attention and MoE")
    assert {k["id"] for k in kinds} == {"mamba", "mlp", "attention", "moe"}

    bad = dict(cfg, hybrid_override_pattern="M-*X")
    with pytest.raises(dg.DeriveError) as exc:
        dg.handler_nemotron_h(bad, bad, "synthetic")
    assert "X" in str(exc.value), (
        "an unknown layer character must be named in the error rather than defaulted")


# --- Where the collectives land ----------------------------------------------------
# A collective's `emit` condition is excluded from cost_signature on purpose: it selects
# whether the node exists in a given deployment, which is a layout property rather than a
# model one. That leaves it verified nowhere, which these cover structurally. This is the
# dense-vs-MoE distinction the behavioural signature deliberately cannot see.


# Architectures whose model code decides sequence parallelism itself rather than reading
# the engine-wide use_sequence_parallel_moe. Named here rather than read from the deriver,
# so the test does not move with the table it checks.
OWN_SEQUENCE_PARALLELISM = {"DeepseekV41ForCausalLM", "DeepseekV4ForCausalLM"}


def test_collectives_land_where_the_parallelism_needs_them():
    """Every layer reduces its MLP output; only a routed layer dispatches to experts.

    A dense layer under tensor parallelism reduces after its MLP. A routed layer also
    exchanges tokens with the ranks holding the experts, which is an All2All under expert
    parallelism. An All2All on a dense layer would price traffic that never moves; a
    missing one on a routed layer would drop the dominant collective of an MoE step."""
    checked_dense = checked_moe = 0
    for d in model_dirs():
        g = graph_of(d)
        # A family that decides sequence parallelism itself names its architecture, and
        # its MoE reduction stands down under that rule rather than the engine-wide one
        # (DeepSeek-V4.1: nvidia/model.py:239-247, 324-327; blis-schemas#55).
        own_sp = g["global"].get("architecture") in OWN_SEQUENCE_PARALLELISM
        mlp_reduction = ("tensor_parallel_unless_sequence_parallel" if own_sp
                         else "tensor_parallel_unless_sp_moe")
        # Context-stack kinds run once over the target's tokens, outside any decoder
        # layer, and launch no collective of their own.
        context_only = set()
        if g.get("speculator", {}).get("context_stack"):
            context_only = (set(layer_sequence(g["speculator"]["context_stack"]))
                            - set(layer_sequence(g["stack"]))
                            - set(layer_sequence(g["speculator"]["stack"])))
        for kind in g["layer_kinds"]:
            if kind["id"] in context_only:
                continue
            ops = [n for n in kind["nodes"] if n["op"] in ("AllReduce", "All2All")]
            all2all = [n for n in ops if n["op"] == "All2All"]
            allreduce = [n for n in ops if n["op"] == "AllReduce"]
            has_experts = any(n["op"] == "GroupedGEMM" for n in kind["nodes"])
            where = f"{d.name}/{kind['id']}"

            assert allreduce, f"{where}: a layer that reduces nothing"
            assert any(n["emit"] in ("tensor_parallel", mlp_reduction)
                       for n in allreduce), (
                f"{where}: no AllReduce conditioned on tensor parallelism")
            if own_sp:
                # Under the model's own sequence parallelism the attention reduction is
                # replaced, not dropped: a reduce-scatter takes its place.
                assert any(n["op"] == "ReduceScatter"
                           and n.get("emit") == "sequence_parallel"
                           for n in kind["nodes"]), (
                    f"{where}: attention reduces under TP but nothing replaces it under "
                    f"the model's own sequence parallelism")

            if has_experts:
                assert len(all2all) == 1, (
                    f"{where}: a routed layer with {len(all2all)} All2All nodes; an MoE "
                    f"step dispatches to the expert ranks exactly once")
                assert all2all[0]["emit"] == "expert_parallel", (
                    f"{where}: All2All emits on {all2all[0]['emit']!r}, but expert "
                    f"dispatch is what expert parallelism conditions")
                # A family whose MoE reduces a different tensor per mode states one
                # reduction per mode instead; neither fires under a dispatch, which every
                # sequence-parallel MoE has, so both stand down there too.
                per_mode = {n["emit"] for n in allreduce} >= {
                    "tensor_parallel_without_expert_parallel",
                    "expert_parallel_without_dispatch"}
                assert per_mode or any(n["emit"] == mlp_reduction for n in allreduce), (
                    f"{where}: a routed layer's MLP reduction must stand down under "
                    f"sequence parallelism, which is what {mlp_reduction!r} says")
                checked_moe += 1
            else:
                assert not all2all, (
                    f"{where}: a layer with no routed experts emits All2All "
                    f"{[n['role'] for n in all2all]}, pricing traffic that never moves")
                checked_dense += 1
    assert checked_dense and checked_moe, (
        f"the catalog must exercise both shapes; saw {checked_dense} dense and "
        f"{checked_moe} routed layer kinds")


def test_every_emit_condition_is_one_a_resolver_knows():
    """The closed set is the point: a graph cannot state a condition nothing evaluates.

    The deriver's EMIT_ constants are the whole vocabulary, so a committed graph naming
    anything else would reach a resolver with no branch for it. The two conditions the
    model's own sequence parallelism decides need the graph to name its architecture,
    since that is what a rules pack decides them by (blis-schemas#55)."""
    dg = _dg()
    known = {dg.EMIT_TENSOR_PARALLEL, dg.EMIT_EXPERT_PARALLEL,
             dg.EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE,
             dg.EMIT_TENSOR_PARALLEL_WITHOUT_EXPERT_PARALLEL,
             dg.EMIT_EXPERT_PARALLEL_WITHOUT_DISPATCH}
    own = {dg.EMIT_SEQUENCE_PARALLEL, dg.EMIT_TENSOR_PARALLEL_UNLESS_SEQUENCE_PARALLEL}
    for d in model_dirs():
        g = graph_of(d)
        allowed = known | (own if g["global"].get("architecture")
                           in OWN_SEQUENCE_PARALLELISM else set())
        nodes = [(kind["id"], n) for kind in g["layer_kinds"] for n in kind["nodes"]]
        nodes += [("head", n) for n in g["head"]]
        nodes += [("embedding", n) for n in g.get("embedding", [])]
        for where, n in nodes:
            if "emit" not in n:
                continue
            assert n["emit"] in allowed, (
                f"{d.name}/{where}: emit {n['emit']!r} is not in the closed set "
                f"{sorted(allowed)}")


def test_a_recurrent_layer_reduces_its_mixer_output():
    """A recurrent mixer is tensor-parallel the same way attention is.

    Its output projection is sharded, so it reduces unconditionally under TP rather than
    under the MoE-aware form an MLP output uses."""
    seen = 0
    for d in model_dirs():
        for kind in graph_of(d)["layer_kinds"]:
            if not any(n["op"] == "RecurrentUpdate" for n in kind["nodes"]):
                continue
            reduces = [n for n in kind["nodes"] if n["op"] == "AllReduce"]
            assert any(n["emit"] == "tensor_parallel" for n in reduces), (
                f"{d.name}/{kind['id']}: a recurrent layer with no unconditional "
                f"tensor-parallel reduction of its mixer output")
            seen += 1
    if not seen:
        pytest.skip("no recurrent layers in the catalog")


# --- The speculator block ----------------------------------------------------------
# A draft stack is work a deployment pays for on every step, so a wrong draft length or
# a draft stack mirrored from the wrong layer kind misprices the model. The source field
# differs by family and the candidates DISAGREE on a shipped config, so which one is read
# is a correctness question rather than a spelling one.


def test_minimax_m3_takes_its_draft_length_from_num_mtp_modules():
    """The case that makes the source field a correctness question.

    MiniMax-M3's config declares BOTH num_mtp_modules (7) and
    num_nextn_predict_layers (1). vLLM reads num_mtp_modules for this family
    (config/speculative.py, minimax_m3_mtp), so reading the other field would understate
    the draft stack sevenfold. Asserted against the config rather than against a literal,
    so the test still means something after a config update."""
    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    assert t.get("num_mtp_modules") != t.get("num_nextn_predict_layers"), (
        "this test is about a config whose two draft-length fields disagree; they now "
        "agree, so it no longer pins the distinction it was written for")
    spec = g["speculator"]
    assert spec["method"] == "minimax_m3_mtp"
    assert spec["num_spec_tokens"] == t["num_mtp_modules"], (
        f"draft length is {spec['num_spec_tokens']}, and num_mtp_modules is "
        f"{t['num_mtp_modules']}; num_nextn_predict_layers is "
        f"{t.get('num_nextn_predict_layers')} and reading it here would be the bug")


def test_deepseek_v3_takes_its_draft_length_from_num_nextn_predict_layers():
    """The other branch: the alias table's field, not num_mtp_modules.

    vLLM maps model_type deepseek_v3 to deepseek_mtp and reads n_predict from
    num_nextn_predict_layers. DeepseekV3ForCausalLM is deliberately NOT in
    MTP_MODULE_ARCHS, and this pins that."""
    d = ROOT / "models" / "deepseek-v3"
    if not d.is_dir():
        pytest.skip("deepseek-v3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    spec = g["speculator"]
    assert spec["method"] == "deepseek_mtp"
    assert spec["num_spec_tokens"] == t["num_nextn_predict_layers"]
    assert "num_mtp_modules" not in t, (
        "this config states num_mtp_modules too, so the assertion above no longer "
        "distinguishes the two sources")


def test_qwen3_5_takes_its_draft_length_from_mtp_num_hidden_layers():
    """The third spelling. Qwen3.5 states neither of the other two fields: its draft
    length rides in mtp_num_hidden_layers, resolved through the alias table."""
    d = ROOT / "models" / "qwen3.5-397b-a17b"
    if not d.is_dir():
        pytest.skip("qwen3.5-397b-a17b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    spec = g["speculator"]
    assert spec["method"] == "qwen3_5_mtp"
    assert spec["num_spec_tokens"] == t["mtp_num_hidden_layers"]


def test_a_declared_mtp_module_count_alone_does_not_make_a_speculator():
    """num_mtp_modules is read only for the families whose method consumes it.

    MiniMax-M2.5 declares num_mtp_modules 3 and vLLM registers no minimax_m2_mtp method
    at all -- its only draft path is a separate Eagle3 checkpoint -- so this config must
    derive NO speculator. Reading the field globally would invent a draft stack here,
    and that invention is what MTP_MODULE_ARCHS being a named set rather than a pattern
    prevents."""
    d = ROOT / "models" / "minimax-m2.5"
    if not d.is_dir():
        pytest.skip("minimax-m2.5 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    assert t.get("num_mtp_modules"), "this test is about a config that declares one"
    assert "speculator" not in g, (
        f"a speculator was derived from num_mtp_modules {t['num_mtp_modules']} for a "
        f"family with no registered MTP method")


def test_a_speculator_mirrors_a_layer_kind_the_graph_defines():
    """A draft stack naming a kind the graph does not define cannot be priced.

    Property test over the catalog: the draft stack is an MoE pass of its own for an MoE
    target, which a scalar draft length would hide, so its entries must resolve."""
    checked = 0
    for d in model_dirs():
        g = graph_of(d)
        spec = g.get("speculator")
        if not spec:
            continue
        defined = {k["id"] for k in g["layer_kinds"]}
        drafted = layer_sequence(spec["stack"])
        assert drafted, f"{d.name}: a speculator with an empty draft stack"
        unknown = set(drafted) - defined
        assert not unknown, f"{d.name}: draft stack names undefined kinds {unknown}"
        assert spec["num_spec_tokens"] > 0, f"{d.name}: non-positive draft length"
        assert spec.get("method"), f"{d.name}: a draft stack with no method"
        checked += 1
    assert checked >= 10, f"only {checked} speculators found; expected the catalog's 12"


def test_a_declared_mtp_vector_is_what_the_draft_stack_mirrors():
    """Where a config declares the draft module's layer composition, it is used.

    NemotronH states mtp_layers_block_type, and its draft stack must follow that vector
    rather than defaulting to the target's last layer kind."""
    d = ROOT / "models" / "nemotron-3.5-lightning-30b-a3b-bf16"
    if not d.is_dir():
        pytest.skip("nemotron-3.5-lightning is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    vector = t.get("mtp_layers_block_type")
    assert vector, "this test is about a config that declares one"
    assert layer_sequence(g["speculator"]["stack"]) == list(vector), (
        f"draft stack is {layer_sequence(g['speculator']['stack'])}, and the config "
        f"declares {vector}")


def test_glm5_3_flash_draft_module_is_sparse_mla_not_the_targets_last_layer():
    """The draft pass must be priced as sparse-MLA, which is NOT this target's last layer.

    GLM-5.3-Flash is the catalog's first family where "mirror the last declared kind" and
    "mirror the last layer in the sequence" disagree. Its layer_types vector ends on
    `linear_attention`, so the final base layer is KDA; the MTP module is sparse-latent.
    The checkpoint settles it -- layer 45 carries `self_attn.indexer.*`, `kv_a_layernorm`
    and `kv_b_proj` and none of KDA's `A_log`/`k_conv1d`/`dt_bias`, while layer 44 is the
    reverse.

    The handler now prices the module itself as DRAFT_KIND_ID, so the derived kind no
    longer depends on declaration order. This test outlives that change: it pins the
    PROPERTY -- the draft pass reads a latent cache, not a recurrent state -- which is
    what a future handler rewrite, or a reversion to derive()'s mirror-a-target-layer
    fallback, would break. Nothing else in the suite would notice such a reversion: the
    draft stack is one layer either way, the layer count is unchanged, and --check would
    re-derive the new wrong graph as current.

    Asserted on what the draft layer COSTS -- a latent cache read and projections, no
    convolutional state -- rather than on its identifier, so renaming a kind leaves this
    passing and repricing the draft pass fails it."""
    d = ROOT / "models" / "glm-5.3-flash"
    if not d.is_dir():
        pytest.skip("glm-5.3-flash is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))

    target = layer_sequence(g["stack"])
    kinds = {k["id"]: k for k in g["layer_kinds"]}

    def ops(layer_id: str) -> set:
        return {n["op"] for n in kinds[layer_id]["nodes"]}

    def mixer(layer_id: str):
        """The layer's sequence mixer, ignoring the indexer's own scoring pass."""
        return next((n for n in kinds[layer_id]["nodes"]
                     if n["op"] == "Attention" and n.get("role") != "block_index_scores"),
                    None)

    def is_kda(layer_id: str) -> bool:
        """The cost-bearing difference: KDA evolves a recurrent state and reads no KV
        cache, so it prices a RecurrentUpdate and has no Attention node. Latent
        attention is the reverse."""
        return "RecurrentUpdate" in ops(layer_id)

    # The premise: this test only means something while the two rules disagree here.
    assert t["layer_types"][-1] == "linear_attention", (
        f"this test is about a config whose last layer is KDA; it is now "
        f"{t['layer_types'][-1]!r}, so the divergence it pins is gone")
    assert is_kda(target[-1]), (
        f"the target's last layer {target[-1]!r} is no longer priced as KDA, so "
        f"mirroring it would no longer be the mistake this guards")

    drafted = layer_sequence(g["speculator"]["stack"])
    assert drafted, "no draft stack"
    assert len(drafted) == int(t["num_nextn_predict_layers"]), (
        f"draft stack is {len(drafted)} layer(s) and the config declares "
        f"{t['num_nextn_predict_layers']} MTP module(s)")

    for lid in drafted:
        # Priced as latent attention: the cache is the latent rank alone, because this
        # config declares mla_use_nope and so carries no rope dimensions beside it.
        assert not is_kda(lid), (
            f"the draft layer {lid!r} is priced as KDA -- the target's last layer kind. "
            f"The MTP module is sparse-MLA (checkpoint layer 45 has no KDA projections), "
            f"so every drafted token is being charged a convolutional state it never "
            f"reads instead of a latent cache it does")
        n = mixer(lid)
        assert n is not None, (
            f"the draft layer {lid!r} prices no Attention node at all, so it reads no "
            f"KV cache; the MTP module is latent attention and does")
        assert n.get("d_h") == int(t["kv_lora_rank"]), (
            f"the draft layer reads a {n.get('d_h')}-wide cache; the latent rank is "
            f"{t['kv_lora_rank']}")
        # The module is sparse-latent, so it selects with a lightning indexer and the
        # graph must charge for that selection.
        assert any(x.get("role") == "block_index_scores"
                   for x in kinds[lid]["nodes"]), (
            "the draft layer carries no indexer; the MTP module is "
            "deepseek_sparse_attention and its top-k selection is work")
        # Routed experts, not a dense MLP: the module ships mlp.gate and shared_experts.
        assert any(x["op"] == "GroupedGEMM" for x in kinds[lid]["nodes"]), (
            "the draft layer has no routed-expert GEMM; the MTP module is MoE")


def _flash():
    d = ROOT / "models" / "glm-5.3-flash"
    if not d.is_dir():
        pytest.skip("glm-5.3-flash is not in the catalog")
    return graph_of(d), text_config(config_of(d))


def test_glm5_3_flash_prices_unquantized_tensors_at_the_base_width():
    """An fp8 checkpoint's excluded tensors keep the base width, per node.

    The config names 1,509 modules in modules_to_not_convert and the pinned safetensors
    headers agree: every KDA projection and every indexer tensor is BF16 while the
    experts and the MLA q_a/q_b/kv_a/o_proj are F8_E4M3. Pricing the BF16 ones at one
    byte understates the weight traffic of all 34 KDA layers.

    The global width must stay fp8: the routed experts are fp8 and are 213x the rest of
    a layer by parameter volume, so flipping the global to bf16 to fix the exceptions
    would overstate far more than it corrects. Asserted as a width per node."""
    g, t = _flash()
    quant = config_of(ROOT / "models" / "glm-5.3-flash").get("quantization_config", {})
    excluded = quant.get("modules_to_not_convert")
    assert excluded, "this test is about a config that names its unquantized tensors"
    assert "fp8" in str(quant.get("quant_method", "")).lower()
    assert g["global"]["weight_dtype"] == "fp8", (
        "the routed experts dominate parameter volume and are fp8; the global width "
        "must stay fp8 with the exceptions carried per node")

    kinds = {k["id"]: k for k in g["layer_kinds"]}
    kda = [n for k in kinds.values() for n in k["nodes"]
           if str(n.get("role", "")).startswith("kda_")]
    assert kda, "no KDA projections found"
    for n in kda:
        assert n.get("weight_dtype") == "bf16", (
            f"KDA projection {n['role']!r} is priced at the global fp8 width; the "
            f"checkpoint stores it BF16 and vLLM builds KDA with quant_config=None")

    for k in kinds.values():
        for n in k["nodes"]:
            if n.get("role") == "index_qk_proj":
                assert n.get("weight_dtype") == "bf16", (
                    "the indexer projection is priced fp8; every indexer tensor is BF16")

    # The other side of the same list: a tensor the quantizer really did convert, and
    # that no loader converts back, keeps the global width. The routed experts are the
    # case -- they are neither in modules_to_not_convert nor in the loader's dequant map,
    # so they are fp8 in the checkpoint AND fp8 in HBM.
    #
    # The MLA projections are deliberately NOT in this list: they are stored fp8 and
    # dequantized on load, which
    # test_loader_dequantized_projections_carry_their_resident_bf16_width covers.
    for k in kinds.values():
        for n in k["nodes"]:
            if n.get("role") == "experts":
                assert "weight_dtype" not in n, (
                    "the routed experts carry an override; the checkpoint quantizes "
                    "them and no loader path converts them back, so they take the "
                    "global fp8 width")

    lm_head = next(n for n in g["head"] if n.get("role") == "lm_head")
    assert lm_head.get("weight_dtype") == "bf16", (
        "lm_head is priced fp8; the config excludes it and the header is BF16")


def test_loader_dequantized_projections_carry_their_resident_bf16_width():
    """A projection this model's loader dequantizes on load is resident at BF16.

    The checkpoint ships q_a_proj, kv_a_proj_with_mqa, q_b_proj and o_proj as block-FP8,
    each with a weight_scale_inv. That is NOT the end of the story: glm5next's
    load_weights routes every weight and scale through _try_load_fp8_attn_proj before the
    normal loader (common/model.py:876-913), whose map covers exactly those four
    (:1369-1374). Once both pieces arrive it calls _dequant_fp8_block, produces BF16, and
    loads that into the unquantized parameter (:1415-1441). Its docstring states the
    intent outright: "Dequantize FP8 q_a_proj / kv_a_proj_with_mqa / o_proj to BF16 on
    load ... the model holds them in BF16".

    So the resident width is BF16 and the stored width is FP8, and the graph prices what
    is resident -- the same rule the mHC matrices follow, where vLLM allocates fp32 over
    a bf16 checkpoint. An earlier revision of this file asserted the opposite on the
    grounds that an absent quant_config "cannot dequantize a tensor whose scales it never
    reads"; that was wrong, because this loader reads them explicitly.

    q_a and kv_a both feed the fused A-projection, so the graph's single qkv_a_proj node
    takes BF16 when either is dequantized -- here both are."""
    d = ROOT / "models" / "glm-5.3-flash"
    if not d.is_dir():
        pytest.skip("glm-5.3-flash is not in the catalog")
    g = graph_of(d)
    glob = g["global"]["weight_dtype"]
    assert glob != "bf16", (
        f"the global width is {glob!r}; with a bf16 global these overrides would be "
        f"unobservable and this test could not tell them from the default")

    kinds = {k["id"]: k for k in g["layer_kinds"]}
    dequantized = ("qkv_a_proj", "q_b_proj", "o_proj")
    seen = set()
    for kid, kind in kinds.items():
        for n in kind["nodes"]:
            role = n.get("role")
            if role in dequantized:
                assert n.get("weight_dtype") == "bf16", (
                    f"{kid}.{role} is priced {n.get('weight_dtype') or glob!r}; "
                    f"_try_load_fp8_attn_proj dequantizes it to BF16 on load, so BF16 is "
                    f"the width resident in HBM whatever the checkpoint stores")
                seen.add(role)
            if role == "kv_b_proj":
                # The control: BF16 in the checkpoint with no scale, so it needs no
                # loader conversion and reaches the same width by the other route.
                assert n.get("weight_dtype") == "bf16", (
                    f"{kid}.kv_b_proj is priced at the global width; the checkpoint "
                    f"ships it BF16 with no scale")
    assert seen == set(dequantized), (
        f"only {sorted(seen)} of {sorted(dequantized)} were found in the graph, so this "
        f"test does not cover what it claims")


def test_glm5_3_flash_prices_mhc_and_replaces_the_norms_it_fuses():
    """mhc: true adds a mixing projection per side, standing in for the norm it fuses.

    vLLM runs one mixing op on each of the attention and FFN sides and passes the
    layernorm weight into it (common/model.py:556-603), so the mixing REPLACES each
    norm rather than preceding it -- emitting both would charge a normalisation pass the
    layer does not run. Each projection is [(2+n)*n, n*hidden] at n = hc_mult.

    Asserted against the config's own hc_mult, so the test survives a config update."""
    g, t = _flash()
    assert t.get("mhc") is True, "this test is about a config that declares mhc"
    n = int(t["hc_mult"])
    hidden = int(t["hidden_size"])
    mix, width = (2 + n) * n, n * hidden

    kinds = {k["id"]: k for k in g["layer_kinds"]}
    base_ids = set(layer_sequence(g["stack"]))
    assert base_ids, "no base layers"

    for lid in sorted(base_ids):
        nodes = kinds[lid]["nodes"]
        roles = [x.get("role") for x in nodes]
        for side in ("hc_attn_fn", "hc_ffn_fn"):
            assert side in roles, f"{lid}: base layer carries no {side}"
            node = next(x for x in nodes if x.get("role") == side)
            assert node["op"] == "GEMM"
            assert (node["n"], node["k"]) == (mix, width), (
                f"{lid}.{side} is {node['n']}x{node['k']}; hc_mult {n} gives "
                f"{mix}x{width}")
            # FP32, which is the RUNTIME width rather than the stored one. vLLM
            # allocates both matrices as torch.float32
            # (glm5next/common/model.py:454-463) and the mixing kernel asserts it
            # (model_executor/kernels/mhc/torch.py:82), so a load-time upcast must occur.
            # The checkpoint stores them bf16; pricing that halves the bytes the kernel
            # reads, across both matrices of every layer.
            assert node.get("weight_dtype") == "fp32", (
                f"{lid}.{side} is priced {node.get('weight_dtype')!r}; the mixing kernel "
                f"asserts fp32 and vLLM allocates the parameter as fp32, so that is the "
                f"width resident in HBM whatever the checkpoint stores")
        # The mixing nodes must state their traffic. An Elementwise node with no
        # bytes_per_token is derivable from hidden_size, which is the ordinary norm
        # these replaced; nothing in a ModelGraph carries hc_mult, so without the field
        # the multi-stream mixing is priced as a single-stream norm.
        for side in ("hc_attn_mix", "hc_ffn_mix"):
            assert side in roles, f"{lid}: base layer carries no {side}"
            node = next(x for x in nodes if x.get("role") == side)
            assert node["op"] == "Elementwise"
            stated = node.get("bytes_per_token")
            assert stated, (
                f"{lid}.{side} states no bytes_per_token, so it prices as a "
                f"hidden-wide norm; the mixing moves {n} residual streams")
            # One fused post+pre over the bundle, per the kernel's documented shapes:
            # post reads x + bundle + the two mixes and writes the bundle; pre reads the
            # bundle and writes the mixes + layer_input.
            act, fp32 = 2, 4
            bundle = n * hidden * act
            mixes = n * fp32 + n * n * fp32
            expected = (hidden * act + bundle + mixes + bundle) + (
                bundle + mixes + hidden * act)
            assert stated == expected, (
                f"{lid}.{side} states {stated} B/token; one fused post+pre over "
                f"{n} streams of {hidden} is {expected}")
            assert stated > 2 * hidden * act, (
                f"{lid}.{side} states {stated} B/token, no more than the "
                f"{2 * hidden * act} an ordinary norm moves; the mixing would be free")

        # The fused norms must not also appear as separate passes.
        assert "input_norm" not in roles, (
            f"{lid}: both the mHC attention mixing and a separate input_norm are "
            f"priced; vLLM fuses the norm into the mixing")
        assert "post_attn_norm" not in roles, (
            f"{lid}: both the mHC FFN mixing and a separate post_attn_norm are priced")


def test_glm5_3_flash_mtp_module_carries_no_mhc():
    """vLLM gates mHC on `not is_mtp_layer`, and the checkpoint agrees.

    Layer 45 carries no hc_* tensors at all while every base layer carries six. A draft
    layer priced with the mixing would charge two GEMMs and two passes per speculative
    step that the module never runs."""
    g, _ = _flash()
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    drafted = layer_sequence(g["speculator"]["stack"])
    assert drafted, "no draft stack"
    for lid in drafted:
        hc = [n.get("role") for n in kinds[lid]["nodes"]
              if str(n.get("role", "")).startswith("hc_")]
        assert not hc, (
            f"the draft layer {lid!r} prices mHC nodes {hc}; the MTP module carries no "
            f"hc_* tensors and vLLM builds it with mHC off")


def test_glm5_3_flash_mtp_module_prices_its_own_prefix():
    """The draft pass pays enorm, hnorm, eh_proj and the shared-head norm.

    vLLM's Glm5NextMultiTokenPredictorLayer builds all four around the decoder block
    (common/mtp.py:51-88), and the pinned checkpoint's layer 45 carries them: an
    `eh_proj` of [4096, 8192] BF16 -- 33,554,432 parameters -- plus `enorm`, `hnorm` and
    `shared_head.norm`. Recording only the method and token count underprices every
    draft pass by that projection.

    eh_proj takes the concatenated [previous hidden, token embedding], so its k is twice
    hidden; asserted that way rather than against 8192."""
    g, t = _flash()
    hidden = int(t["hidden_size"])
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    drafted = layer_sequence(g["speculator"]["stack"])
    assert len(drafted) == int(t["num_nextn_predict_layers"])

    for lid in drafted:
        nodes = kinds[lid]["nodes"]
        roles = [n.get("role") for n in nodes]
        eh = next((n for n in nodes if n.get("role") == "mtp_eh_proj"), None)
        assert eh is not None, (
            f"the draft layer {lid!r} prices no eh_proj; the module concatenates the "
            f"previous hidden state with the token embedding and projects back")
        assert eh["op"] == "GEMM"
        assert (eh["n"], eh["k"]) == (hidden, 2 * hidden), (
            f"eh_proj is {eh['n']}x{eh['k']}; the concatenated input makes it "
            f"{hidden}x{2 * hidden}")
        assert eh.get("weight_dtype") == "bf16", (
            "eh_proj is priced fp8; the checkpoint stores it BF16")
        for role in ("mtp_enorm", "mtp_hnorm", "mtp_shared_head_norm"):
            assert role in roles, f"the draft layer prices no {role}"

    # The draft kind must be its own, not a reused target layer: it carries the prefix
    # and no target layer does.
    assert set(drafted).isdisjoint(set(layer_sequence(g["stack"]))), (
        "the draft stack reuses a target layer kind, so it cannot be carrying the "
        "MTP-specific prefix")


@pytest.mark.parametrize("mutate,description", [
    (lambda g: g["speculator"].update(num_spec_tokens=g["speculator"]["num_spec_tokens"] + 6),
     "lengthening the draft stack"),
    (lambda g: g["speculator"].update(method="some_other_mtp"),
     "changing the speculative method"),
    (lambda g: g["speculator"]["stack"].update(repeat=2),
     "running the draft stack twice"),
    (lambda g: g.pop("speculator"),
     "dropping the speculator entirely"),
])
def test_signature_notices_a_changed_speculator(mutate, description):
    """The gap this block was reported for: the signature must see a draft change.

    cost_signature iterated only the target stack, so a graph with the wrong draft
    length, the wrong method, or a mis-mirrored draft pattern collided with a correct one
    and passed test_distinct_architectures_cost_differently -- the strongest test here.
    These four mutations are the shapes that bug would take."""
    import copy

    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 is not in the catalog")
    g = graph_of(d)
    assert "speculator" in g, "the reference model must have a speculator"
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    mutate(variant)
    assert cost_signature(variant) != before, (
        f"{description} left the cost signature unchanged, so a wrong draft stack would "
        f"pass the collision tests")


def test_signature_separates_draft_work_from_target_work():
    """A layer moved between the target stack and the draft stack must be visible.

    Without the draft stack tagged as its own component, a counter over both would be
    blind to the move: the same layer bodies in the same quantity, priced differently
    because a draft pass runs per speculated token rather than once."""
    import copy

    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 is not in the catalog")
    g = graph_of(d)
    moved = copy.deepcopy(g)
    drafted = layer_sequence(g["speculator"]["stack"])
    # Append the draft layers to the target stack and leave the speculator's own stack
    # claiming them too: total layer bodies unchanged in kind, but the split differs.
    moved["stack"] = {"prologue": layer_sequence(g["stack"]) + drafted}
    assert cost_signature(moved) != cost_signature(g), (
        "moving layers into the target stack did not change the signature")


# --- compress(): the stack's spelling -----------------------------------------------
# compress decides how a layer sequence is stored, and the deriver runs it on every
# model. test_signature_ignores_how_the_stack_is_spelled proves the choice is
# cost-neutral, which is the property a cost model cares about; these tests cover the
# other half, that the stored form is CORRECT and says what the sequence actually is.
# A reader inspects the committed stack to see a hybrid's period, so a stack that
# expands correctly but misreports the period is still wrong for the thing compression
# exists to do.


def _dg():
    """The deriver as a module, for the functions that take no catalog on disk."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dg", DERIVER)
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)
    return dg


@pytest.mark.parametrize("sequence,expected,description", [
    (["a"] * 8, {"pattern": ["a"], "repeat": 8},
     "a uniform run collapses to a period of one"),
    (["a", "b"] * 5, {"pattern": ["a", "b"], "repeat": 5},
     "a two-layer period is stored once"),
    (["d", "d", "d"] + ["s"] * 9,
     {"prologue": ["d", "d", "d"], "pattern": ["s"], "repeat": 9},
     "a dense prologue then a uniform sparse tail"),
    ((["k", "k", "k", "m"] * 23) + ["m"],
     {"pattern": ["k", "k", "k", "m"], "repeat": 23, "epilogue": ["m"]},
     "a hybrid period with a trailing odd layer"),
    (["a", "b", "c", "d", "e"], {"prologue": ["a", "b", "c", "d", "e"]},
     "a sequence with no repeating unit is stored literally"),
    (["a"], {"prologue": ["a"]}, "a single layer"),
    ([], {}, "an empty sequence"),
])
def test_compress_stores_the_sequence_it_was_given(sequence, expected, description):
    assert _dg().compress(list(sequence)) == expected, description


def test_compress_prefers_the_true_period_over_a_padded_one():
    """A period-1 tail must be stored as one entry, not as a multiple of itself.

    This is a regression test. The search used to reject any candidate whose prologue
    was longer than its period, which rules out the optimal split on the commonest
    shape in this catalog -- a short dense prologue then a uniform sparse tail -- and
    left a worse one to win. Four committed graphs (glm-5.2, glm-5.2-fp8, glm-5.3,
    minimax-m3) stored `pattern: [attn_moe, attn_moe, attn_moe], repeat: 25` for a
    75-layer run whose period is 1, which reads as a three-layer repeating unit the
    model does not have. It expanded correctly, so every cost-based test passed.

    The GLM shape itself, at its real depth."""
    got = _dg().compress(["attn_dense"] * 3 + ["attn_moe"] * 75)
    assert got == {"prologue": ["attn_dense"] * 3,
                   "pattern": ["attn_moe"], "repeat": 75}, (
        f"stored {got}, which spells a period-1 run as a longer unit")


def test_compress_declines_a_split_that_stores_no_less_than_the_sequence():
    """The docstring's readability rule: past a point the literal form is clearer.

    A sequence whose only repeating unit is surrounded by material longer than the
    saving must come back literal rather than as a split that technically compresses."""
    seq = ["a", "b"] * 3 + ["x", "y", "z"]
    got = _dg().compress(list(seq))
    assert got == {"prologue": list(seq)}, (
        f"stored {got}; a split saving nothing should fall back to the literal form")
    assert "pattern" not in got


@pytest.mark.parametrize("sequence", [
    ["a"] * 8,
    ["a", "b"] * 5,
    ["d", "d", "d"] + ["s"] * 9,
    (["k", "k", "k", "m"] * 23) + ["m"],
    ["m", "e", "m", "e", "m", "a", "e", "m", "e"],
    ["a", "b", "c", "d", "e"],
    ["p", "q", "r", "s", "t"] + ["a", "b"] * 3,
    ["a", "b"] * 3 + ["x", "y", "z"],
    ["a"],
])
def test_compress_round_trips(sequence):
    """However it is spelled, it must expand to what went in.

    The property that makes compression safe at all: layer_sequence is what every
    consumer reads the stack through, so a compression it cannot invert would misprice
    the model however well the stored form reads."""
    assert layer_sequence(_dg().compress(list(sequence))) == list(sequence)


def test_compress_round_trips_every_committed_graph():
    """The same property against the real catalog rather than hand-built shapes."""
    dg = _dg()
    for d in model_dirs():
        seq = layer_sequence(graph_of(d)["stack"])
        assert layer_sequence(dg.compress(list(seq))) == seq, d.name


# --- MiniMax-M3: block-sparse attention ------------------------------------------


def _m3():
    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 not in the catalog")
    return graph_of(d), text_config(config_of(d))


def test_m3_bounded_read_is_cheaper_than_full_attention_at_long_context():
    """The point of block-sparse attention is a read that stops growing with context.

    Asserted as a cost difference rather than a field value: a graph that recorded
    the bound but left the kind full would read the whole context, and only a
    comparison catches that. The bound is in tokens, so at any context beyond it the
    sparse layer reads strictly fewer KV bytes than a full-attention layer of the
    same shape.
    """
    g, t = _m3()
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"
            and n["kind"] == "swa"]
    assert main, "no bounded attention node; a sparse layer would read the full context"
    sparse = t["sparse_attention_config"]
    bound = (sparse["sparse_topk_blocks"] + sparse["sparse_local_block"]
             + sparse["sparse_init_block"]) * sparse["sparse_block_size"]
    for n in main:
        per_token = 2 * n["n_kv"] * n["d_h"]
        for context in (8192, 131072, 1048576):
            bounded = min(context, n["window"]) * per_token
            full = context * per_token
            assert bounded < full, (
                f"at context {context} the bounded read is not cheaper than a full one"
            )
            assert bounded == bound * per_token, (
                f"the bound should saturate at {bound} tokens, not track context"
            )


def test_m3_indexer_read_is_not_bounded():
    """The indexer scores every block to choose the top k, so its read tracks context.

    If the indexer were given the same bound as the attention it feeds, the model would
    charge nothing for selection and a long-context step would come out too fast. The
    distinction is the whole reason it is a separate node."""
    g, _ = _m3()
    idx = [n for k in g["layer_kinds"] for n in k["nodes"]
           if n.get("role") == "block_index_scores"]
    assert idx, "no indexer node; block selection would be free"
    for n in idx:
        assert "window" not in n, (
            "the indexer carries a window, which would bound a read that must scan the "
            "whole context to rank blocks"
        )
        assert n["kind"] == "gqa"


def test_m3_dense_prologue_is_wider_than_a_routed_expert():
    """The first three layers are dense MLPs at dense_intermediate_size.

    Both widths are declared and they differ by 4x, so a handler that reached for
    intermediate_size would make the prologue a quarter of its true cost. Asserted
    as the inequality the weights show rather than as the literal number."""
    g, t = _m3()
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    dense_up = [n for n in kinds["dense"]["nodes"]
                if n.get("role") == "mlp_gate_up"][0]
    expert = [n for n in kinds["sparse_attn_moe"]["nodes"]
              if n["op"] == "GroupedGEMM"][0]
    assert dense_up["n"] == 2 * t["dense_intermediate_size"]
    assert expert["n"] == t["intermediate_size"]
    assert dense_up["n"] > 2 * expert["n"], (
        "the dense prologue should be wider than a routed expert on this config"
    )


def test_m3_prices_differently_from_its_gqa_sibling():
    """M2.5 and M3 are both MiniMax MoE models and must not price alike.

    M2.5 is dense-attention GQA throughout; M3 bounds its attention read and adds
    an indexer. A handler registered against the wrong architecture would collapse
    the two."""
    g3, _ = _m3()
    d2 = ROOT / "models" / "minimax-m2.5"
    if not d2.is_dir():
        pytest.skip("minimax-m2.5 not in the catalog")
    assert cost_signature(g3) != cost_signature(graph_of(d2))


# --- DeepSeek-V4-Pro: compressed sparse attention at two ratios -------------------


def _v4():
    d = ROOT / "models" / "deepseek-v4-pro"
    if not d.is_dir():
        pytest.skip("deepseek-v4-pro not in the catalog")
    return graph_of(d), text_config(config_of(d))


def test_v4_has_one_layer_kind_per_compression_ratio():
    """The config's compress_ratios vector names two ratios, and they cost differently.

    vLLM's compressor asserts compress_ratio in [4, 128] and sizes each layer's read
    from it, so a graph collapsing them to one kind would price 30 of 61 layers with
    the other one's bound. Asserted through the bound rather than the kind id: an id is
    a label, a window is a cost."""
    g, t = _v4()
    ratios = sorted(set(t["compress_ratios"][: t["num_hidden_layers"]]))
    assert len(ratios) == 2, f"expected two ratios, config states {ratios}"
    bounds = set()
    draft = set(layer_sequence(g["speculator"]["stack"]))
    for k in g["layer_kinds"]:
        if k["id"] in draft:
            continue  # the draft layer, which is not one of the target ratios
        for n in k["nodes"]:
            if n["op"] == "Attention" and n.get("role") != "block_index_scores":
                bounds.add((n.get("compress_ratio"), n.get("index_topk")))
    assert len(bounds) == len(ratios), (
        f"two compression ratios must give two distinct read bounds, got {bounds}"
    )


def test_v4_compression_ratios_cross_over_at_index_topk_entries():
    """The two ratios bound the read differently, and which reads less depends on the
    context.

    A ratio-4 layer reads its window plus at most index_topk compressed entries
    (nvidia/flashmla.py:193-203); a ratio-128 layer reads its window plus every
    compressed entry, context / 128 of them (sparse_mla.py:403-437). So the ratio-128
    layer reads less only while context / 128 < index_topk, below 131,072 tokens here,
    and MORE above it: at V4-Pro's 1M-token window it reads 8,192 entries against
    1,024. A graph giving both the same bound, or bounding the ratio-128 read by a
    constant, gets one side of that crossover wrong."""
    g, t = _v4()
    layers = t["num_hidden_layers"]
    by_ratio = {}
    seq = layer_sequence(g["stack"])
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    for idx, ratio in enumerate(t["compress_ratios"][:layers]):
        k = kinds[seq[idx]]
        for n in k["nodes"]:
            if n["op"] == "Attention" and n.get("role") != "block_index_scores":
                by_ratio.setdefault(ratio, set()).add(
                    (n["window"], n.get("compress_ratio"), n.get("index_topk")))
    assert all(len(v) == 1 for v in by_ratio.values()), (
        f"a ratio maps to more than one bound: {by_ratio}"
    )
    flat = {r: v.pop() for r, v in by_ratio.items()}
    lo, hi = min(flat), max(flat)
    # Every layer keeps the same sliding window; the ratio is what differs.
    assert flat[lo][0] == flat[hi][0] == t["sliding_window"], (
        f"both ratios retain the declared SWA window, got {flat}")
    assert flat[hi][1] == hi and flat[lo][1] == lo, (
        f"each layer must record its own compression ratio, got {flat}")
    # The read set is window + compressed positions, evaluated rather than compared as
    # a stored constant because the ratio-128 term is context-dependent and no constant
    # expresses it.
    def positions(bound, context):
        window, ratio, topk = bound
        compressed = context // ratio
        if topk:
            compressed = min(topk, compressed)
        return window + compressed
    crossover = hi * t["index_topk"]
    for context in (1024, 8192, 65536, crossover - hi):
        assert positions(flat[hi], context) < positions(flat[lo], context), (
            f"at context {context}, ratio {hi} should read less than ratio {lo}: "
            f"{positions(flat[hi], context)} vs {positions(flat[lo], context)}")
    for context in (crossover + hi, t["max_position_embeddings"]):
        assert positions(flat[hi], context) > positions(flat[lo], context), (
            f"at context {context}, ratio {hi} reads every compressed entry and should "
            f"read more than ratio {lo}'s top-{t['index_topk']}")
    # And the ordering INVERTS past a crossover, which is a real consequence of the two
    # bounds being different in kind rather than in degree: ratio 4 selects at most
    # index_topk compressed positions, so its read plateaus, while ratio 128 reads a
    # positional run that keeps growing. Stated here because a reader who assumed
    # "higher ratio is always cheaper" would mispredict long-context deployments, and
    # this config permits max_position_embeddings far beyond the crossover.
    crossover = flat[lo][2] * flat[hi][1]
    assert positions(flat[hi], crossover) == positions(flat[lo], crossover), (
        f"the two bounds should meet at context {crossover}")
    assert positions(flat[hi], 2 * crossover) > positions(flat[lo], 2 * crossover), (
        f"past context {crossover} the ratio-{hi} read must exceed the ratio-{lo} one, "
        f"whose top-k has plateaued")


def test_v4_head_dim_is_not_summed_with_rope():
    """head_dim is the inclusive per-token width, not the part outside RoPE.

    vLLM's compressor derives nope_head_dim = head_dim - rope_head_dim
    (models/deepseek_v4/compressor.py), so adding qk_rope_head_dim to head_dim would
    overstate this model's KV cache by an eighth. There is no kv_lora_rank to
    reconstruct it from, which is why the MLA branch cannot price this family."""
    g, t = _v4()
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
    assert main
    for n in main:
        assert n["d_h"] == t["head_dim"]
        assert n["d_h"] != t["head_dim"] + t["qk_rope_head_dim"]


def test_v4_indexer_is_unbounded_where_the_attention_is_bounded():
    """The indexer scans every compressed key, (pos + 1) // 4 of them
    (sparse_attn_indexer.py:726-736), where the attention it feeds reads a window plus
    at most index_topk: the measured split is 3.53x over a 12x context increase against
    1.18x. One node cannot carry both."""
    g, _ = _v4()
    indexed = 0
    for k in g["layer_kinds"]:
        idx = [n for n in k["nodes"] if n.get("role") == "block_index_scores"]
        main = [n for n in k["nodes"]
                if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
        assert len(main) == 1, f"{k['id']}: expected one attention node"
        assert main[0]["window"] > 0, "every layer retains its sliding window"
        if not idx:
            continue  # a layer with no indexer, checked by the test below
        assert len(idx) == 1, f"{k['id']}: expected at most one indexer"
        assert "window" not in idx[0], "a bounded indexer would make selection free"
        indexed += 1
    assert indexed, "no layer carries an indexer"


def test_v4_prices_differently_from_deepseek_v3():
    """V3 is dense MLA with kv_lora_rank; V4 is compressed sparse with an indexer.

    Both are DeepSeek MoE models, so a handler registered against the wrong architecture
    would be easy to miss -- and sending V4 through handler_moe is precisely what the
    registry comment warns derives kind=swa window=128 off sliding_window."""
    g4, _ = _v4()
    d3 = ROOT / "models" / "deepseek-v3"
    if not d3.is_dir():
        pytest.skip("deepseek-v3 not in the catalog")
    assert cost_signature(g4) != cost_signature(graph_of(d3))


def test_v4_experts_are_priced_at_their_own_declared_width():
    """A checkpoint storing experts at a different width from the rest must say so.

    DeepSeek-V4-Pro declares expert_dtype fp4 beside an fp8 quantization_config, and
    vLLM resolves that name to MXFP4 experts (models/deepseek_v4/quant_config.py).
    Pricing them at the global fp8 width doubles 384 experts from 720 GiB to 1,441 GiB,
    which puts 180 GiB per rank on a 141 GiB H200 and makes a deployment InferenceX ran
    at tp=8 on 8 GPUs look impossible. Asserted through the occupancy as well as the
    field, because the field is a label and the bytes are the cost."""
    g, t = _v4()
    assert t.get("expert_dtype") == "fp4", "this test is about a config that states one"
    experts = [n for k in g["layer_kinds"] for n in k["nodes"]
               if n["op"] == "GroupedGEMM" and n.get("role") != "shared_experts"]
    assert experts, "no expert node"
    # The shared expert is not one of them: it is a dense FP8 DeepseekV4MLP
    # (nvidia/model.py:900-914; checkpoint F8_E4M3 shared_experts.w1/w2/w3), stated as
    # its gate/up and down GEMMs at the global width.
    shared = [n for k in g["layer_kinds"] for n in k["nodes"]
              if n.get("role") in ("shared_gate_up", "shared_down")]
    assert shared and all(n["op"] == "GEMM" and "weight_dtype" not in n
                          and g["global"]["weight_dtype"] == "fp8" for n in shared)
    for n in experts:
        assert n.get("weight_dtype") == "mxfp4", (
            f"expert node carries weight_dtype {n.get('weight_dtype')!r}; vLLM "
            f"resolves expert_dtype fp4 to mxfp4"
        )
    # The occupancy that width decides: 0.5 bytes per parameter, against fp8's 1.
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    gib = 0.0
    for lid in layer_sequence(g["stack"]):
        for n in kinds[lid]["nodes"]:
            if n["op"] == "GroupedGEMM":
                gib += 3 * n["n"] * n["k"] * n["experts"] * 0.5 / 2**30
    per_rank = gib / 8
    assert per_rank < 141 * 0.9, (
        f"{per_rank:.1f} GiB per rank at tp=8 does not fit a 141 GiB H200, so this "
        f"deployment would be refused"
    )


def test_an_unmapped_expert_dtype_is_an_error_not_a_fallback():
    """Falling back to the global width is the bug this field exists to fix.

    A width the deriver does not recognize must stop the derivation rather than quietly
    pricing the experts at the checkpoint's other dtype."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dg", ROOT / "scripts" / "derive_graph.py")
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)
    cfg = {"hidden_size": 7168, "num_local_experts": 8, "num_experts_per_tok": 2,
           "intermediate_size": 1024, "expert_dtype": "fp6"}
    try:
        dg.moe(cfg, "synthetic", 7168)
    except dg.DeriveError as exc:
        assert "fp6" in str(exc)
        return
    raise AssertionError("an unmapped expert_dtype was accepted")


# --- DeepSeek-V4.1-Flash ------------------------------------------------------------
# Every assertion below is about what the graph CHARGES -- weight bytes, FLOPs, traffic,
# which bounded read a layer performs -- and never about kind ids or node order, so a
# handler that renames or regroups its kinds without changing a price keeps passing.
# The per-layer rules are checked three ways: against the committed config, as a
# property over generated source vectors, and as metamorphic relations between configs
# that differ in one declared layer.

V41 = MODELS / "deepseek-v4.1-flash"

# Bytes per stored element at the widths this graph uses. mxfp4 is payload only, which
# is also what blis-schemas' DType.Bytes() returns; the E8M0 scales are not counted on
# either side of the comparisons below.
WIDTH = {"fp32": 4, "bf16": 2, "fp8": 1, "mxfp4": 0.5}


def v41_graph() -> dict:
    # Committed, so its absence is a failure rather than a reason to skip the suite.
    assert V41.is_dir(), "models/deepseek-v4.1-flash is missing"
    return graph_of(V41)


def expanded_layers(g: dict, stack: dict | None = None) -> list[list[dict]]:
    kinds = {k["id"]: k["nodes"] for k in g["layer_kinds"]}
    return [kinds[k] for k in layer_sequence(stack or g["stack"])]


def weight_cost(n: dict, global_dtype: str) -> tuple[float, float]:
    """(stored elements, resident bytes) a weight-bearing node charges. A node that
    aliases another's weights runs compute on them but holds none of its own."""
    if n.get("aliases"):
        return 0, 0
    w = WIDTH[n.get("weight_dtype") or global_dtype]
    if n["op"] == "GEMM":
        return n["n"] * n["k"], n["n"] * n["k"] * w
    if n["op"] == "GroupedGEMM":
        e = 3 * n["n"] * n["k"] * n["experts"]
        return e, e * w
    return 0, 0


def caches_of(nodes: list[dict]) -> dict[str, dict]:
    """The caches a layer's reads state, by name. A reader states its owner's record, so
    the name is enough to find the record a launch writes."""
    return {c["name"]: c for n in nodes for c in n.get("caches", [])}


def per_token(n: dict, caches: dict[str, dict]) -> float:
    """What a launch moves per token: its own traffic plus, when it writes a cache, one
    record, the two together once every per_tokens tokens."""
    moved = n.get("bytes_per_token") or 0
    if "writes_cache" in n:
        moved += caches[n["writes_cache"]]["bytes_per_entry"]
    return moved / n.get("per_tokens", 1)


def traffic_of(nodes: list[dict], role_prefix: str) -> float:
    caches = caches_of(nodes)
    return sum(per_token(n, caches) for n in nodes
               if n["op"] == "Elementwise" and n.get("role", "").startswith(role_prefix))


def compressed_read(nodes: list[dict]) -> dict | None:
    """The layer's primary (roleless) attention: what it reads per decoded token."""
    return next(n for n in nodes if n["op"] == "Attention" and not n.get("role"))


def scoring_scans(nodes: list[dict]) -> list[dict]:
    return [n for n in nodes if n["op"] == "Attention" and n.get("role")]


def gemm_elements(nodes: list[dict], *roles: str) -> int:
    return sum(n["n"] * n["k"] for n in nodes
               if n["op"] == "GEMM" and n.get("role") in roles)


# Matrix shapes and stored dtypes read from the safetensors headers at the pinned
# revision 5254448103b6254fc646dacbca1cbbf20e9216c1 (ranged reads of all 48 shard
# headers; every layer's shapes are identical, the per-layer differences are which
# tensors exist). (out, in, stored dtype). The routed experts are packed two FP4 values
# per I8 byte, so [2304, 2560] stores a 2304 x 5120 matrix.
V41_CHECKPOINT = {
    "every_layer": [
        (1280, 5120, "fp8"),    # attn.wq_a
        (512, 5120, "fp8"),     # attn.wkv
        (32768, 1280, "fp8"),   # attn.wq_b
        (8192, 4096, "fp8"),    # attn.wo_a
        (5120, 8192, "fp8"),    # attn.wo_b
        (384, 5120, "bf16"),    # ffn.gate
        (2304, 5120, "fp8"),    # ffn.shared_experts.w1
        (2304, 5120, "fp8"),    # ffn.shared_experts.w3
        (5120, 2304, "fp8"),    # ffn.shared_experts.w2
        (24, 20480, "fp32"),    # hc_attn_fn
        (24, 20480, "fp32"),    # hc_ffn_fn
    ],
    "routed_expert": [(2304, 5120, "mxfp4"), (2304, 5120, "mxfp4"),
                      (5120, 2304, "mxfp4")],
    "compressor_gated": [(512, 5120, "bf16"), (512, 5120, "bf16")],   # wkv, wgate
    "compressor_ungated": [(512, 5120, "bf16")],                     # layer 20: no wgate
    "index_query": [(4096, 1280, "fp8"), (32, 5120, "bf16")],        # wq_b, weights_proj
    "index_key": [(128, 512, "bf16")],                               # wk
    "engram": [(25600, 6144, "fp8")],                                # engram.wkv
}
# What vLLM v0.31.0 holds resident at another width than the checkpoint stores, on SM90.
# wo_a is a batched MXFP8 matmul. Its kernels are DeepGemmMxfp8BmmLinearKernel and the
# emulation kernel (model_executor/kernels/linear/__init__.py:895-900); the first is
# Blackwell-only, and the second dequantizes it to BF16 at load (mxfp8/emulation.py:
# 31-50). Nothing else is converted.
V41_RESIDENT_OVERRIDES = {(8192, 4096): "bf16"}


def v41_checkpoint_layer(t: dict, i: int) -> list[tuple[int, int, str]]:
    kv = set(t["kv_source_layer_ids"])
    idx = set(t["index_source_layer_ids"])
    mats = list(V41_CHECKPOINT["every_layer"])
    mats += V41_CHECKPOINT["routed_expert"] * t["n_routed_experts"]
    if i in kv:
        mats += V41_CHECKPOINT["compressor_gated" if t["compress_ratios"][i] > 1
                               else "compressor_ungated"]
    if i in idx:
        mats += V41_CHECKPOINT["index_query"]
        if i in kv:
            mats += V41_CHECKPOINT["index_key"]
    if i in t["engram_layer_ids"]:
        mats += V41_CHECKPOINT["engram"]
    if i == 0:
        # The broadcast pre projects the embedding with hc_attn_fn summed over its
        # hc_mult copies (nvidia/model.py:1178-1191): a derived (24, 5120) FP32 matrix in
        # place of the (24, 20480) one, which layer 0 still loads but never multiplies.
        mats.remove((24, 20480, "fp32"))
        mats.append((24, 5120, "fp32"))
    return mats


def test_deepseek_v41_charges_the_weights_the_checkpoint_stores_at_their_resident_width():
    """Every matrix a target layer multiplies by is priced at the checkpoint's element
    count and at the width vLLM holds it in. Vectors (norms, attn_sink, gate biases) and
    gathered tables (the engram rows) are not GEMM weights and are not counted.

    The two sums are compared per layer, so a missing projection, a fused node that
    double-counts a stage, a node at the wrong width or a layer that pays for a module it
    does not build each move one side and not the other. The resident widths differ
    from the stored ones only where V41_RESIDENT_OVERRIDES says, which is where a loader
    converts."""
    g, t = v41_graph(), text_config(config_of(V41))
    glob = g["global"]["weight_dtype"]
    for i, nodes in enumerate(expanded_layers(g)):
        priced = [weight_cost(n, glob) for n in nodes]
        want = v41_checkpoint_layer(t, i)
        want_elems = sum(n * k for n, k, _ in want)
        want_bytes = sum(n * k * WIDTH[V41_RESIDENT_OVERRIDES.get((n, k), dt)]
                         for n, k, dt in want)
        assert sum(e for e, _ in priced) == want_elems, (
            f"layer {i}: graph prices {sum(e for e, _ in priced):,} weight elements, "
            f"the checkpoint stores {want_elems:,}")
        assert sum(b for _, b in priced) == want_bytes, (
            f"layer {i}: graph prices {sum(b for _, b in priced):,} resident bytes, "
            f"vLLM holds {want_bytes:,}")


def test_deepseek_v41_head_is_priced_at_the_unquantized_width():
    """The LM head is built with no quant_config (nvidia/model.py:1338-1342) and stored
    BF16, so it costs vocab x hidden x 2 bytes, not the global FP8 width."""
    g, t = v41_graph(), text_config(config_of(V41))
    head = [weight_cost(n, g["global"]["weight_dtype"]) for n in g["head"]]
    assert sum(b for _, b in head) == t["vocab_size"] * t["hidden_size"] * 2


def v41_kv_bytes_per_context_token(g: dict) -> float:
    """What every context token adds to the owned, unbounded caches of one sequence:
    each owner's record divided by the tokens one entry covers. Readers hold nothing,
    and the window and state caches are bounded per request, not per token."""
    total = 0.0
    for nodes in expanded_layers(g):
        for n in nodes:
            for c in n.get("caches", []):
                if c["owned"] and c["kind"] in ("kv", "index"):
                    total += c["bytes_per_entry"] / c.get("tokens_per_entry", 1)
    return total


def test_deepseek_v41_owned_caches_hold_what_vllm_allocates_per_context_token():
    """KV residency, from the cache records the graph states rather than from a
    restatement in this test.

    Per context token a sequence holds one 584 B record (attention.py:503-508) per
    compressed-stream owner, divided by its ratio, and one 132 B indexer key record
    (index_head_dim FP8 plus one FP32 scale per 128, attention.py:89-97) per key owner,
    divided by its ratio. At the published layout that is 3 x 584/2 + 584 + 3 x 132/2 +
    132 = 1,790 B. Each ratio-2 owner also holds an FP32 state ring of 8 rows of kv and
    score at head_dim (compressor.py:146-163, 230-238), and every target and draft layer
    its own 128-position window."""
    g, t = v41_graph(), text_config(config_of(V41))
    assert v41_kv_bytes_per_context_token(g) == 1_790
    kv, ratios = t["kv_source_layer_ids"], t["compress_ratios"]
    want = sum(584 / ratios[i] for i in kv) + sum(
        132 / ratios[i] for i in kv if i in t["index_source_layer_ids"])
    assert v41_kv_bytes_per_context_token(g) == want
    rings = [c for nodes in expanded_layers(g) for n in nodes for c in n.get("caches", [])
             if c["kind"] == "state"]
    assert [(c["entries"], c["bytes_per_entry"]) for c in rings] == [
        (8, 2 * t["head_dim"] * 4)] * sum(1 for i in kv if ratios[i] == 2)
    windows = [c for nodes in expanded_layers(g) + expanded_layers(
                   g, g["speculator"]["stack"])
               for n in nodes for c in n.get("caches", []) if c["kind"] == "window"]
    assert len(windows) == t["num_hidden_layers"] + t["num_nextn_predict_layers"]
    assert {(c["window"], c["bytes_per_entry"], c["owned"]) for c in windows} == {
        (t["sliding_window"], 584, True)}


def test_deepseek_v41_every_reader_states_the_record_of_the_cache_it_reads():
    """A reader holds nothing, but it prices its read at the record it reads. So each
    unowned cache must state exactly the record of the nearest earlier owner of that
    name, and every unowned cache must have one: a reader whose owner is missing, or
    whose record drifted from it, would read a cache no layer allocates."""
    g = v41_graph()
    owners: dict[str, dict] = {}
    readers = 0
    for i, nodes in enumerate(expanded_layers(g)):
        for n in nodes:
            for c in n.get("caches", []):
                if c["owned"]:
                    owners[c["name"]] = c
                    continue
                readers += 1
                assert c["name"] in owners, (i, c["name"])
                own = owners[c["name"]]
                for key in ("kind", "bytes_per_entry", "tokens_per_entry"):
                    assert c.get(key) == own.get(key), (i, c["name"], key)
    assert readers > 30, readers


def apply_capability_overlay(g: dict, family: int) -> dict:
    """The graph a family-`family` chip runs, built from arch_overrides the way the
    schema's ForCapability does: matching roles and cache names take the overlay's
    fields. Restated here so a check on the overlay does not run the code it checks."""
    import copy
    out = copy.deepcopy(g)
    over = next(o for o in g["arch_overrides"] if o["capability_family"] == family)
    nodes_by_role = {o["role"]: o for o in over.get("nodes", [])}
    caches_by_name = {o["name"]: o for o in over.get("caches", [])}
    for kind in out["layer_kinds"]:
        for n in kind["nodes"]:
            for key, v in nodes_by_role.get(n.get("role"), {}).items():
                if key != "role":
                    n[key] = v
            for c in n.get("caches", []):
                for key, v in caches_by_name.get(c["name"], {}).items():
                    if key != "name":
                        c[key] = v
    return out


def test_deepseek_v41_blackwell_overlay_runs_wo_a_natively_and_stores_the_mxfp8_record():
    """On SM100 the MXFP8 BMM runs natively, so wo_a stays FP8 (kernels/linear/
    __init__.py:895-900), and the latent record is 528 B with 512 B page alignment
    rather than SM90's 584 B and 576 (attention.py:503-508). The compressed cache adds
    a 288 B NVFP4 record under nvfp4_ds_mla. Only those change: the per-token residency
    follows the record, and every weight but wo_a's keeps its SM90 width."""
    g = v41_graph()
    assert g["base_capability_families"] == [9]
    sm100 = apply_capability_overlay(g, 10)
    glob = g["global"]["weight_dtype"]
    for base, over in zip(expanded_layers(g), expanded_layers(sm100)):
        for a, b in zip(base, over):
            if a.get("role") == "wo_a":
                assert (a["weight_dtype"], b["weight_dtype"]) == ("bf16", "fp8")
            else:
                assert weight_cost(a, glob) == weight_cost(b, glob), a.get("role")
    # 3 x 528/2 + 528 for the stream, and the indexer keys unchanged at 330.
    assert v41_kv_bytes_per_context_token(sm100) == 1_650
    compressed = {(c["bytes_per_entry"], c.get("page_alignment"),
                   frozen(c.get("bytes_by_cache_dtype")))
                  for nodes in expanded_layers(sm100) for n in nodes
                  for c in n.get("caches", []) if c["name"] == "compressed" and c["owned"]}
    assert compressed == {(528, 512, frozen({"nvfp4_ds_mla": 288}))}


def test_deepseek_v41_engram_tables_live_in_host_memory_at_their_stored_width():
    """Each engram layer gathers 24 rows a token -- (engram_max_ngram_size - 1) n-grams
    for each of engram_n_heads heads (common/engram.py:199, 925) -- from its own table of
    engram_num_embeddings rows. A row is engram_head_dim FP8 values and one E8M0 scale
    per 32, and the gathered rows reach the GPU as BF16. vLLM v0.31.0 keeps the tables
    in host memory by default (cpu_offload=True), which is why the node states host
    residency: priced on the device, the two tables alone are about 189 GiB."""
    g, t = v41_graph(), text_config(config_of(V41))
    layers = expanded_layers(g)
    hd = t["engram_head_dim"]
    rows_per_token = (t["engram_max_ngram_size"] - 1) * t["engram_n_heads"]
    gathered = []
    for i, nodes in enumerate(layers):
        for n in nodes:
            if n["op"] == "Gather":
                assert i in t["engram_layer_ids"], i
                gathered.append(n)
                assert (n["row_bytes"], n["rows_per_token"], n["out_bytes_per_token"],
                        n["residency"], n["shard"]) == (
                    hd + hd // 32, rows_per_token, rows_per_token * hd * 2, "host", "heads")
    assert [n["rows"] for n in gathered] == t["engram_num_embeddings"]
    assert 188 < sum(n["rows"] * n["row_bytes"] for n in gathered) / 2**30 < 190


def test_deepseek_v41_embedding_is_a_bf16_gather_over_the_vocabulary():
    """The token embedding is a row gather from a vocab x hidden BF16 table (the quant
    config has no method for it), split by rows across tensor parallelism and summed
    back. The draft reuses it rather than holding its own: its lookup is the
    speculator's embedding, outside every layer, because its reduction runs before the
    stream is sharded and so in both sequence-parallel modes (dspark.py:93-98,
    205-215)."""
    g, t = v41_graph(), text_config(config_of(V41))
    (embed,) = [n for n in g["embedding"] if n["op"] == "Gather"]
    assert (embed["rows"], embed["row_bytes"], embed["rows_per_token"],
            embed["shard"]) == (t["vocab_size"], t["hidden_size"] * 2, 1, "rows")
    assert [n["emit"] for n in g["embedding"] if n["op"] == "AllReduce"] == [
        "tensor_parallel"]
    spec = g["speculator"]
    assert not [n for nodes in expanded_layers(g, spec["stack"]) for n in nodes
                if n["op"] == "Gather"]
    assert [(n["op"], n.get("aliases"), n.get("emit")) for n in spec["embedding"]] == [
        ("Gather", "embedding.embed_tokens", None), ("AllReduce", None, "tensor_parallel")]
    (draft,) = [n for n in spec["embedding"] if n["op"] == "Gather"]
    assert {k: v for k, v in draft.items() if k not in ("role", "aliases")} == {
        k: v for k, v in embed.items() if k != "role"}


def v41_layer_facts(nodes: list[dict]) -> dict:
    """What a layer pays for and holds, read from its costs and caches rather than its
    labels."""
    read = compressed_read(nodes)
    scans = scoring_scans(nodes)
    held = {c["name"]: c["owned"] for c in read.get("caches", [])}
    roles = {n.get("role") for n in nodes}
    return {
        "owns_stream": traffic_of(nodes, "compressor") > 0,
        "holds_stream": held.get("compressed"),
        "holds_ring": held.get("compressor_ring"),
        "scores": len(scans),
        "scan_ratio": scans[0].get("compress_ratio") if scans else None,
        "owns_keys": traffic_of(nodes, "index_k") > 0,
        "holds_keys": scans[0]["caches"][0]["owned"] if scans else None,
        "selects": "candidate_select" in roles,
        "masks": "candidate_mask" in roles,
        "engram_rows": [n["rows"] for n in nodes if n["op"] == "Gather"],
        "read_ratio": read.get("compress_ratio"),
        "read_topk": read.get("index_topk"),
        "read_window": read.get("window"),
        "prefill_dequant": "prefill_dequant_gather" in roles,
    }


def v41_expected(t: dict, i: int) -> dict:
    r = t["compress_ratios"][i]
    kv, idx = t["kv_source_layer_ids"], t["index_source_layer_ids"]
    src = max((s for s in kv if s <= i), default=None) if r else None
    # vLLM's own defaults: no source, or no blocks, means no candidate selection
    # (nvidia/model.py:666-668).
    cand = t.get("candidate_source_layer_id", -1)
    if t.get("candidate_topk_blocks", 0) <= 0:
        cand = -1
    return {
        "owns_stream": i in kv,
        "holds_stream": (i in kv) if r else None,
        "holds_ring": True if i in kv and r > 1 else None,
        "scores": int(i in idx),
        "scan_ratio": t["compress_ratios"][src] if i in idx else None,
        "owns_keys": i in kv and i in idx,
        "holds_keys": (i in kv) if i in idx else None,
        "selects": i == cand,
        "masks": i in idx and cand >= 0 and i > cand,
        "engram_rows": [dict(zip(t["engram_layer_ids"], t["engram_num_embeddings"]))[i]]
                       if i in t["engram_layer_ids"] else [],
        # The read is the window plus at most index_topk positions of the source's
        # stream (nvidia/flashmla.py:230-246); the ratio names that stream, whose cache
        # the read holds or borrows.
        "read_ratio": r or None,
        "read_topk": t["index_topk"] if r else None,
        "read_window": t["sliding_window"],
        "prefill_dequant": bool(r),
    }


def test_deepseek_v41_each_layer_pays_for_what_its_source_vectors_assign():
    """Layer by layer, the committed graph charges exactly the work vLLM builds there
    and holds exactly the caches it allocates.

    Only the KV sources pay for a compressor and its cache insert and own the stream
    (attention.py:565-575, 1105-1106), and only ratio-2 sources hold a state ring; only
    the index sources pay for a scoring scan, and only those that are also KV sources
    pay for the key path and own the keys (:421-449, 1288-1298). The candidate source
    selects blocks and the later index sources mask to them (kernels/attention/dsa/
    candidate_blocks.py). Each engram layer gathers from its own table
    (common/engram.py:167-202). Every compressed layer reads its
    window plus at most a top-k of its source's stream; a ratio-0 layer reads its
    window alone."""
    g, t = v41_graph(), text_config(config_of(V41))
    layers = expanded_layers(g)
    assert len(layers) == t["num_hidden_layers"]
    for i, nodes in enumerate(layers):
        assert v41_layer_facts(nodes) == v41_expected(t, i), f"layer {i}"


def v41_config(**over) -> dict:
    """The published text config, with overrides, for direct calls to the handler."""
    t = dict(text_config(config_of(V41)))
    t.update(over)
    return t


def v41_derive(t: dict) -> dict:
    """Build a graph from a text config through the same path derive() takes."""
    dg = deriver()
    raw = dict(config_of(V41), text_config=t)
    kinds, stack, extras = dg.handler_deepseek_v41(t, raw, "synthetic")
    return {"layer_kinds": kinds, "stack": stack, "speculator": extras["speculator"],
            "global": {"weight_dtype": "fp8", "architecture": "DeepseekV41ForCausalLM"},
            "head": extras["head"],
            "embedding": extras["embedding"]}


def random_v41_vectors(rng, layers: int) -> dict:
    """A config whose source vectors are valid by vLLM's rules: each compressed span
    starts at a KV source of its own ratio, and index sources are compressed layers."""
    swa_head = rng.randint(0, 3)
    ratios, kv = [0] * swa_head, []
    i = swa_head
    while i < layers:
        span = rng.randint(1, max(1, layers - i))
        r = rng.choice((1, 2))
        kv.append(i)
        ratios += [r] * span
        i += span
    compressed = [j for j, r in enumerate(ratios) if r]
    # Any compressed layers may index, subject to vLLM's two rules: the first compressed
    # layer needs an index source at or below it, and an index source that is not a KV
    # source needs its KV source to index too (attention.py:326-338, 437-449). KV
    # sources that do not index are therefore drawn as well.
    idx = set(rng.sample(compressed, rng.randint(0, len(compressed)))) | {compressed[0]}
    for j in list(idx):
        if j not in kv:
            idx.add(max(s for s in kv if s <= j))
    idx = sorted(idx)
    # Unsorted as often as not: vLLM pairs each table with its layer by list position.
    engram = rng.sample(range(1, layers), rng.randint(0, 2))
    # Distinct table sizes, so a layer gathering from another layer's table shows.
    tables = rng.sample(range(1_000, 400_000_000), len(engram))
    targets = sorted(rng.sample(range(layers), rng.randint(1, min(3, layers))))
    # The candidate source is an index source, or disabled.
    return {"num_hidden_layers": layers, "compress_ratios": ratios + [0, 0, 0],
            "kv_source_layer_ids": kv, "index_source_layer_ids": idx,
            "candidate_source_layer_id": rng.choice(idx + [-1]),
            "candidate_topk_blocks": rng.choice((0, 8, 16)),
            "engram_layer_ids": engram, "engram_num_embeddings": tables,
            "dspark_target_layer_ids": targets}


def test_deepseek_v41_layer_rules_hold_for_any_valid_source_vectors():
    """The same per-layer rule, as a property over generated configs.

    The committed config exercises one arrangement. 300 seeded draws vary the depth,
    the number and placement of KV sources, which compressed layers index, which of
    those selects candidates and where the engram layers fall, and each graph must
    charge and hold at every layer what v41_expected says. The draws include index
    sources that are not KV sources at ratio 2, which the published config never has.
    Every draw's readers must also find an owner with their record."""
    import random
    rng = random.Random(4101)
    for draw in range(300):
        t = v41_config(**random_v41_vectors(rng, rng.randint(4, 24)))
        g = v41_derive(t)
        layers = expanded_layers(g)
        assert len(layers) == t["num_hidden_layers"], draw
        for i, nodes in enumerate(layers):
            assert v41_layer_facts(nodes) == v41_expected(t, i), (
                draw, i, t["compress_ratios"], t["kv_source_layer_ids"])
        owners: dict[str, dict] = {}
        for nodes in layers:
            for c in (c for n in nodes for c in n.get("caches", [])):
                if c["owned"]:
                    owners[c["name"]] = c
                else:
                    assert c["bytes_per_entry"] == owners[c["name"]]["bytes_per_entry"]


def work_by_layer(g: dict) -> list[collections.Counter]:
    return [collections.Counter(node_cost(n) for n in nodes)
            for nodes in expanded_layers(g)]


def test_deepseek_v41_adding_a_kv_source_moves_exactly_one_compressor():
    """Metamorphic: declaring one more KV source inside a compressed span adds exactly
    one compressor's work to that layer, makes it own the stream, and changes no other
    layer's price.

    The new source owns the stream from there on, but every reader still reads the same
    window and top-k of a stream at the same ratio and record, so nothing a reader pays
    changes -- which is the property that makes a per-layer KV-ownership field necessary
    rather than derivable from the reads. Residency is where it shows: one more ratio-2
    stream (292 B per context token) and one more state ring."""
    base = v41_config()
    added = v41_config(kv_source_layer_ids=sorted(base["kv_source_layer_ids"] + [5]))
    gb, ga = v41_derive(base), v41_derive(added)
    before, after = work_by_layer(gb), work_by_layer(ga)
    for i, (b, a) in enumerate(zip(before, after)):
        if i != 5:
            assert a == b, f"layer {i} changed price when layer 5 became a KV source"
    lb, la = expanded_layers(gb)[5], expanded_layers(ga)[5]
    assert ({node_cost(n) for n in lb if n["op"] != "Attention"}
            <= {node_cost(n) for n in la})
    roles = {n.get("role") for n in la} - {n.get("role") for n in lb}
    assert roles == {"compressor_fused_wkv_wgate", "compressor_pool_norm",
                     "compressor_rope_quant_insert"}, roles
    assert v41_kv_bytes_per_context_token(ga) - v41_kv_bytes_per_context_token(gb) == 292
    assert v41_layer_facts(la)["holds_ring"] and not v41_layer_facts(lb)["holds_ring"]


def test_deepseek_v41_halving_the_pool_doubles_the_scan_and_the_insert_traffic():
    """Metamorphic: at ratio 1 a compressed stream holds one entry per token where
    ratio 2 holds one per pair. So the scorer scans twice the positions (its ratio
    halves, 2 -> 1), the per-token cache-insert traffic doubles, an owned stream holds
    twice the bytes per context token, and the main read's top-k and window are
    unchanged, because they are counts of positions, not fractions of the context."""
    two = v41_config()
    one = v41_config(compress_ratios=[r if r != 2 else 1 for r in two["compress_ratios"]])
    l2, l1 = expanded_layers(v41_derive(two))[2], expanded_layers(v41_derive(one))[2]
    assert (scoring_scans(l1)[0]["compress_ratio"] * 2
            == scoring_scans(l2)[0]["compress_ratio"])
    for role in ("compressor_rope_quant_insert", "index_k_norm_rope_store"):
        assert traffic_of(l1, role) == 2 * traffic_of(l2, role), role
    for field in ("index_topk", "window"):
        assert compressed_read(l1)[field] == compressed_read(l2)[field]
    held = [{c["name"]: c["bytes_per_entry"] / c.get("tokens_per_entry", 1)
             for c in compressed_read(nodes)["caches"] if c["kind"] == "kv"}
            for nodes in (l1, l2)]
    assert held[0]["compressed"] == 2 * held[1]["compressed"]
    # The scorer's logits row covers one position per entry, so it doubles too.
    def logits(nodes):
        (c,) = [n["context"] for n in nodes if n.get("role") == "index_logits_topk"]
        return c["bytes_per_entry"] / c.get("tokens_per_entry", 1)
    assert logits(l1) == 2 * logits(l2)
    # And the compressor drops its gate: wkv and wgate merged at ratio 2, wkv alone at 1
    # (compressor.py:216-227), so the projection halves.
    assert (2 * gemm_elements(l1, "compressor_fused_wkv_wgate")
            == gemm_elements(l2, "compressor_fused_wkv_wgate"))


def test_deepseek_v41_engram_splits_the_mhc_boundary_without_changing_its_traffic():
    """Conservation: an engram layer runs the mHC post and pre as two launches around
    the injection (nvidia/model.py:522-552), where every other layer runs them fused.
    The split moves no bytes: post + pre on the engram layer equals the fused boundary
    on a plain layer (layer 3)."""
    g = v41_graph()
    layers = expanded_layers(g)
    t = text_config(config_of(V41))
    assert t["engram_layer_ids"], "the config declares no engram layer to check"
    plain = layers[3]
    assert 3 not in t["engram_layer_ids"] and traffic_of(plain, "engram") == 0
    for i in t["engram_layer_ids"]:
        assert (traffic_of(layers[i], "mhc_attn_post")
                + traffic_of(layers[i], "mhc_attn_pre")
                == traffic_of(plain, "mhc_attn_post_pre"))
        assert traffic_of(layers[i], "mhc") == traffic_of(plain, "mhc"), i


def test_deepseek_v41_mhc_runs_once_per_sublayer_at_fp32():
    """Every sublayer of every layer pays one mHC projection, at FP32 (nvidia/model.py:
    400-444), and pays no separate RMSNorm, which the TileLang epilogue fuses.

    Summed over the stack, the FP32 projection weights are 2 x layers x mix_hc x
    hc_mult x hidden x 4 bytes, less layer 0's attention side, which projects the
    broadcast embedding at hidden width. The launches that state no per-token traffic
    are the ones whose work is proportional to the context instead."""
    g, t = v41_graph(), text_config(config_of(V41))
    hc, hidden, layers = t["hc_mult"], t["hidden_size"], t["num_hidden_layers"]
    mix = (2 + hc) * hc
    fp32 = sum(weight_cost(n, "fp8")[1] for nodes in expanded_layers(g) for n in nodes
               if n.get("weight_dtype") == "fp32")
    assert fp32 == (2 * layers - 1) * mix * hc * hidden * 4 + mix * hidden * 4
    for nodes in expanded_layers(g):
        plain_norms = [n for n in nodes if n["op"] == "Elementwise"
                       and "bytes_per_token" not in n and "context" not in n]
        assert not plain_norms, [n.get("role") for n in plain_norms]


def test_deepseek_v41_draft_module_drafts_a_block_through_its_own_experts():
    """The DSpark draft: dspark_block_size tokens per step (config/speculative.py:
    1475-1482), drafted as one parallel block through num_nextn_predict_layers layers
    that read only their window (their compress_ratios entries are 0, attention.py:
    300-305) plus the block itself, non-causally -- 128 + 5 = 133 positions -- route over
    dspark_n_routed_experts at dspark_num_experts_per_tok, and run no engram and no
    indexer. Each draft layer still caches a 128-position window."""
    g, t = v41_graph(), text_config(config_of(V41))
    spec = g["speculator"]
    assert (spec["method"], spec["drafting"]) == ("dspark", "parallel_block")
    assert spec["num_spec_tokens"] == t["dspark_block_size"]
    draft = expanded_layers(g, spec["stack"])
    assert len(draft) == t["num_nextn_predict_layers"]
    for nodes in draft:
        read = compressed_read(nodes)
        assert not read.get("compress_ratio") and not read.get("index_topk")
        assert read["window"] == t["sliding_window"] + t["dspark_block_size"]
        assert [c["window"] for c in read["caches"]] == [t["sliding_window"]]
        assert not scoring_scans(nodes)
        assert not [n for n in nodes if n.get("residency") == "host"]
        routed = [n for n in nodes if n["op"] == "GroupedGEMM"
                  and n.get("role") != "shared_experts"]
        assert [(n["experts"], n["top_k"]) for n in routed] == [
            (t["dspark_n_routed_experts"], t["dspark_num_experts_per_tok"])]
    # The drafter materializes its own hc copies before its first layer
    # (dspark.py:226), so that layer's attention-side mHC costs what the target's
    # broadcast layer 0 costs, not a plain pre.
    assert traffic_of(draft[0], "mhc_attn") == traffic_of(expanded_layers(g)[0], "mhc_attn")
    # main_proj concatenates the three target layers' hidden states, once per target
    # token, so it runs in the context stack, not per drafted token.
    context = expanded_layers(g, spec["context_stack"])
    assert gemm_elements([n for nodes in context for n in nodes], "dspark_main_proj") == (
        t["hidden_size"] * t["hidden_size"] * len(t["dspark_target_layer_ids"]))
    assert not gemm_elements([n for nodes in draft for n in nodes], "dspark_main_proj")


def test_deepseek_v41_draft_context_projects_each_layer_through_its_own_weights():
    """Each draft layer's context KV comes from that layer's own fused_wqa_wkv
    (dspark.py:192), run over the target's tokens. So the context stack has one
    projection per draft layer, each aliasing a distinct draft layer's weights and
    matching their shape, and holds no weight of its own besides main_proj."""
    g, t = v41_graph(), text_config(config_of(V41))
    spec = g["speculator"]
    kinds = {k["id"]: k["nodes"] for k in g["layer_kinds"]}
    context = [n for nodes in expanded_layers(g, spec["context_stack"]) for n in nodes]
    aliased = [n for n in context if n.get("aliases")]
    assert len(aliased) == t["num_nextn_predict_layers"]
    targets = set()
    for n in aliased:
        kind, role = n["aliases"].split(".")
        (orig,) = [m for m in kinds[kind] if m.get("role") == role]
        assert (orig["n"], orig["k"]) == (n["n"], n["k"])
        targets.add(kind)
    assert targets == set(layer_sequence(spec["stack"]))
    held = [n.get("role") for n in context if weight_cost(n, "fp8")[0]]
    assert held == ["dspark_main_proj"]


@pytest.mark.parametrize("over, fragment", [
    ({"compress_ratios": [0, 0, 4] + [4] * 37 + [0, 0, 0]}, "accepts 0, 1 and 2"),
    ({"compress_ratios": [0, 0] + [2] * 4 + [1] * 2 + [2] * 12 + [1] * 20 + [0, 0, 0]},
     "reads the compressed"),
    ({"compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 2]}, "draft layer"),
    ({"kv_source_layer_ids": [1, 2, 8, 14, 20]}, "no compressed stream"),
    ({"head_dim": 576}, "head_dim 512"),
    ({"engram_layer_ids": [0, 14]}, "engram_layer_ids names a layer outside"),
    ({"engram_layer_ids": [1, 40]}, "engram_layer_ids names a layer outside"),
    ({"engram_layer_ids": [14, 14]}, "engram_layer_ids repeats"),
    ({"kv_source_layer_ids": [-1, 2, 8, 14, 20]}, "kv_source_layer_ids names a layer"),
    ({"index_source_layer_ids": [2, 8, 14, 20, 40]}, "index_source_layer_ids names"),
    ({"kv_source_layer_ids": [2.9, 8, 14, 20]}, "not a whole number"),
    ({"compress_ratios": [0, 0] + [2.5] * 18 + [1] * 20 + [0, 0, 0]}, "not a whole"),
    ({"compress_ratios": [0, 0] + [True] * 18 + [1] * 20 + [0, 0, 0]}, "not a whole"),
    ({"compress_ratios": [0, 2] + [2] * 18 + [1] * 20 + [0, 0, 0]},
     "reads the compressed cache of layer None"),
    ({"compress_ratios": [0, 0] + [2] * 18 + [1] * 21}, "compress_ratios has 41"),
    ({"sliding_window": 0}, "sliding_window is 0"),
    ({"hc_mult": 0}, "hc_mult is 0"),
    ({"o_groups": 7}, "o_groups 7 does not divide"),
    ({"o_groups": 0}, "o_groups is 0"),
    ({"qk_rope_head_dim": 32}, "head_dim 512 with rope"),
    ({"index_head_dim": 192}, "not a multiple of 128"),
    ({"engram_max_ngram_size": 1}, "engram_max_ngram_size is 1"),
    ({"n_shared_experts": 0}, "num_shared_experts is 0"),
    ({"dspark_target_layer_ids": []}, "dspark_target_layer_ids is empty"),
    ({"dspark_target_layer_ids": [37, 38, 99]}, "dspark_target_layer_ids names"),
    ({"dspark_block_size": 0}, "dspark_block_size is 0"),
    ({"index_source_layer_ids": [2, 9, 14, 20]}, "builds no indexer"),
    ({"index_source_layer_ids": [8, 14, 20]}, "no index source at or below"),
])
def test_deepseek_v41_refuses_layouts_vllm_would_not_serve(over, fragment):
    """Each refusal is a layout that vLLM v0.31.0 either rejects or would read at a
    width or ratio no node states, so pricing it would be a guess."""
    with pytest.raises(deriver().DeriveError, match=fragment):
        v41_derive(v41_config(**over))


def test_deepseek_v41_refuses_every_source_layout_vllm_rejects():
    """The negative property: perturb valid generated layouts into the two shapes vLLM
    v0.31.0 raises on, and require every one to be refused rather than priced.

      * a compressed layer with no index source at or below it (attention.py:326-338);
      * an index source whose KV source builds no indexer, so its keys do not exist
        (attention.py:437-449)."""
    import random
    rng = random.Random(4102)
    seen = collections.Counter()
    for _ in range(400):
        v = random_v41_vectors(rng, rng.randint(6, 24))
        kv, idx = v["kv_source_layer_ids"], v["index_source_layer_ids"]
        first = next(j for j, r in enumerate(v["compress_ratios"]) if r)
        borrowers = [j for j in idx if j not in kv]
        if borrowers and rng.random() < 0.6:
            j = rng.choice(borrowers)
            src = max(s for s in kv if s <= j)
            bad_idx = [s for s in idx if s != src]
        else:
            bad_idx = [s for s in idx if s > first]
        if not bad_idx:
            bad_idx = [v["num_hidden_layers"] - 1]
        with pytest.raises(deriver().DeriveError,
                           match="no index source at or below|builds no indexer") as exc:
            v41_derive(v41_config(**dict(v, index_source_layer_ids=bad_idx)))
        seen["builds no indexer" in str(exc.value)] += 1
    assert seen[True] > 50 and seen[False] > 50, seen


def v41_expected_traffic(t: dict, role: str, ratio: int) -> float | None:
    """What each launch moves per token, restated from its kernel rather than from the
    handler, cache records and the once-per-entry cadence included. None for an RMSNorm
    at hidden width, which states no traffic because a consumer derives it from
    hidden_size, and for a launch whose work is proportional to the context."""
    H, act, fp32, hc = t["hidden_size"], 2, 4, t["hc_mult"]
    D, qlora, heads = t["head_dim"], t["q_lora_rank"], t["num_attention_heads"]
    rec = 584                                   # SM90 record, attention.py:503-508
    ih, idim = t["index_n_heads"], t["index_head_dim"]
    irec = idim + idim // 128 * 4               # attention.py:89-97
    bundle, mixes = hc * H * act, (hc + hc * hc) * fp32
    post = H * act + 2 * bundle + mixes         # reads x, residual, mixes; writes residual
    pre = bundle + mixes + H * act              # reads residual; writes mixes, layer input
    role = re.sub(r"_\d+$", "", role)           # dspark_context_*_{d}: one per draft layer
    table = {
        "mhc_attn_post_pre": post + pre, "mhc_ffn_post_pre": post + pre,
        "mhc_attn_post": post, "mhc_final_post": post, "mhc_attn_pre": pre,
        # broadcast: the expand reads hidden and writes the bundle, then a pre
        "mhc_attn_broadcast_pre": H * act + bundle + pre,
        "hc_collapse": bundle + hc * fp32 + H * act,
        # fused_q_kv_rmsnorm reads and writes both latents at BF16
        "qkv_a_layernorm": 2 * (qlora + D) * act,
        # Q read + padded write, KV row read, one record written
        "q_rope_kv_insert": 2 * heads * D * act + D * act + rec,
        "o_inv_rope": 2 * heads * D * act,
        "compressor_rope_quant_insert": (D * act + rec) / max(ratio, 1),
        # BF16 q and weights in; FP8 q and FP32 weights out (no stored q scale)
        "index_q_rope_quant": ih * idim * act + ih * act + ih * idim + ih * fp32,
        "index_k_norm_rope_store": (idim * act + irec) / max(ratio, 1),
        # reads the bundle, the hc keys and the value; writes the bundle
        "engram_gate": 3 * bundle + H * act,
        "dspark_context_kv_norm": 2 * D * act,
        "dspark_context_kv_rope_insert": D * act + rec,
        "dspark_main_norm": None, "dspark_final_norm": None, "final_norm": None,
        "index_logits_topk": None, "candidate_select": None, "candidate_mask": None,
        "candidate_block_scores": None, "prefill_dequant_gather": None,
        "index_prefill_k_gather": None,
    }
    if role == "compressor_pool_norm":
        row = (2 * D if ratio > 1 else D) * fp32
        # decode: every ratio-2 token reads its raw row and writes it to the ring;
        # a closing token also reads the ring row and its raw row and writes a latent
        return 2 * row + (2 * row + D * act) / ratio if ratio > 1 else row + D * act
    return table[role]


def v41_expected_context(t: dict, role: str, ratio: int) -> dict | None:
    """The context-proportional work a launch states, restated from its kernel: bytes
    per context position (per entry of a cache it reads), and whether it runs per query
    token or once per request."""
    fp32, D, act = 4, t["head_dim"], 2
    scan = {"tokens_per_entry": ratio} if ratio > 1 else {}
    return {
        # an FP32 score per scanned entry, written by the scan and read by the top-k
        # (sparse_attn_indexer.py:583-626, 726-785)
        "index_logits_topk": {"bytes_per_entry": 2 * fp32, **scan, "per": "query_token"},
        # block scores read from the logits row (candidate_blocks.py:20-58, 392-445)
        "candidate_select": {"bytes_per_entry": fp32, **scan, "per": "query_token"},
        # an FP32 score per candidate block, written and read by top-k, across the
        # row's whole width: max_model_len in decode (candidate_blocks.py:400-445)
        "candidate_block_scores": {"bytes_per_entry": 2 * fp32,
                                   "tokens_per_entry": max(ratio, 1)
                                   * t["candidate_block_size"],
                                   "per": "query_token",
                                   "decode_extent": "max_model_len"},
        # prefill writes each key it reads into a workspace (sparse_attn_indexer.py:
        # 515-528)
        "index_prefill_k_gather": {"reads_cache": "index",
                                   "bytes_per_entry": t["index_head_dim"]
                                   + t["index_head_dim"] // 128 * 4,
                                   "per": "request", "phase": "prefill"},
        # -inf stores across the whole row, max_model_len wide in decode
        # (candidate_blocks.py:114-145, 448-480)
        "candidate_mask": {"bytes_per_entry": fp32, **scan, "per": "query_token",
                           "decode_extent": "max_model_len"},
        # prefill dequantizes the whole stream into a BF16 workspace once per step,
        # writing head_dim BF16 values per entry
        "prefill_dequant_gather": {"reads_cache": "compressed", "bytes_per_entry": D * act,
                                   "per": "request", "phase": "prefill"},
    }.get(role)


def v41_all_launch_sites(g: dict, t: dict) -> list[tuple[str, int, list[dict], int]]:
    """(where, index, nodes, compress ratio) for every target layer, draft layer, the
    draft's context stack, the head and the draft head."""
    spec = g["speculator"]
    ratios = t["compress_ratios"]
    return ([("target", i, n, ratios[i]) for i, n in enumerate(expanded_layers(g))]
            + [("draft", i, n, 0) for i, n in enumerate(expanded_layers(g, spec["stack"]))]
            + [("context", 0, n, 0) for n in expanded_layers(g, spec["context_stack"])]
            + [("head", 0, g["head"], 0), ("draft_head", 0, spec["head"], 0)])


def test_deepseek_v41_charges_each_launch_the_traffic_its_kernel_moves():
    """Every Elementwise launch in the target stack, the draft's stacks and both heads
    moves the bytes its kernel reads and writes, at the stored widths, its cache record
    included and amortized over the tokens one entry covers. A launch whose work grows
    with the context states that instead, at the width its kernel reads and writes.

    Checked per layer, so a ratio applied to the wrong layer, a launch priced twice or
    an unamortized per-entry write each show up where they happen."""
    g, t = v41_graph(), text_config(config_of(V41))
    checked = 0
    for where, i, nodes, ratio in v41_all_launch_sites(g, t):
        caches = caches_of(nodes)
        for n in nodes:
            if n["op"] != "Elementwise":
                continue
            want = v41_expected_traffic(t, n["role"], ratio)
            got = per_token(n, caches) if ("bytes_per_token" in n
                                           or "writes_cache" in n) else None
            assert got == want, (where, i, n["role"], got, want)
            assert n.get("context") == v41_expected_context(t, n["role"], ratio), (
                where, i, n["role"])
            checked += 1
    # 40 target layers, 3 draft layers, the context stack and two heads.
    assert checked > 250, checked


V41_CHECKPOINT_DRAFT = {
    # mtp.{0,1,2} at the pinned revision: the base attention set, the hc tensors, a
    # 128-expert gate and 128 routed experts, and the shared expert.
    "every_layer": [m for m in V41_CHECKPOINT["every_layer"] if m != (384, 5120, "bf16")]
                   + [(128, 5120, "bf16")],
    "main_proj": [(5120, 15360, "fp8")],                 # mtp.0.main_proj
    "markov_head": [(129280, 256, "bf16")],              # mtp.2.markov_head.head
}


def test_deepseek_v41_draft_layers_charge_the_mtp_weights_at_their_resident_width():
    """The DSpark module prices every matrix of the checkpoint's mtp.* tensors it
    multiplies by, at vLLM's resident widths, each exactly once: the layers their own
    attention and experts, the context stack main_proj, and the sequential head the
    Markov projection. The context projections and the draft lm_head reuse weights
    another node holds (dspark.py:192, 279-282) and so hold none; the Markov head's
    embed half is a gather (qwen3_dspark.py:48-93), and the confidence head runs only
    under adaptive verification."""
    g, t = v41_graph(), text_config(config_of(V41))
    spec, glob = g["speculator"], g["global"]["weight_dtype"]

    def check(nodes, want, where):
        priced = [weight_cost(n, glob) for n in nodes]
        assert sum(e for e, _ in priced) == sum(n * k for n, k, _ in want), where
        assert sum(b for _, b in priced) == sum(
            n * k * WIDTH[V41_RESIDENT_OVERRIDES.get((n, k), dt)]
            for n, k, dt in want), where

    for d, nodes in enumerate(expanded_layers(g, spec["stack"])):
        check(nodes, V41_CHECKPOINT_DRAFT["every_layer"]
              + V41_CHECKPOINT["routed_expert"] * t["dspark_n_routed_experts"], d)
    check([n for nodes in expanded_layers(g, spec["context_stack"]) for n in nodes],
          V41_CHECKPOINT_DRAFT["main_proj"], "context")
    check(spec["head"], [], "draft head")
    check(spec["sequential_head"], V41_CHECKPOINT_DRAFT["markov_head"], "markov head")
    # The Markov head's embed half: one rank-wide BF16 row per previous token.
    (w1,) = [n for n in spec["sequential_head"] if n["op"] == "Gather"]
    assert (w1["rows"], w1["row_bytes"]) == (t["vocab_size"], t["dspark_markov_rank"] * 2)
    (lm,) = [n for n in spec["head"] if n["op"] == "GEMM"]
    assert lm["aliases"] == "head.lm_head"


def v41_required_launches(t: dict, where: str, i: int) -> list[str]:
    """The Elementwise launches vLLM v0.31.0 runs on SM90 for one target layer, draft
    layer, the draft's context stack or a head, from the config's vectors alone (no
    graph is read)."""
    kv, idx = t["kv_source_layer_ids"], t["index_source_layer_ids"]
    cand = t.get("candidate_source_layer_id", -1)
    if t.get("candidate_topk_blocks", 0) <= 0:
        cand = -1
    attn = ["qkv_a_layernorm", "q_rope_kv_insert"]
    if where == "head":
        return ["mhc_final_post", "hc_collapse", "final_norm"]
    if where == "draft_head":
        return ["mhc_final_post", "hc_collapse", "dspark_final_norm"]
    if where == "context":
        return ["dspark_main_norm"] + [
            f"dspark_context_{s}_{d}" for d in range(t["num_nextn_predict_layers"])
            for s in ("kv_norm", "kv_rope_insert")]
    if where == "draft":
        return ((["mhc_attn_broadcast_pre"] if i == 0 else ["mhc_attn_post_pre"])
                + attn + ["o_inv_rope", "mhc_ffn_post_pre"])
    if i == 0:
        boundary = ["mhc_attn_broadcast_pre"]
    elif i in t["engram_layer_ids"]:
        boundary = ["mhc_attn_post", "engram_gate", "mhc_attn_pre"]
    else:
        boundary = ["mhc_attn_post_pre"]
    work = boundary + attn
    if i in kv:
        work += ["compressor_pool_norm", "compressor_rope_quant_insert"]
    if i in idx:
        work += (["index_q_rope_quant"] + (["index_k_norm_rope_store"] if i in kv else [])
                 + ["index_logits_topk", "index_prefill_k_gather"])
        if i == cand:
            work += ["candidate_select", "candidate_block_scores"]
        elif cand >= 0 and i > cand:
            work.append("candidate_mask")
    if t["compress_ratios"][i]:
        work.append("prefill_dequant_gather")
    return work + ["o_inv_rope", "mhc_ffn_post_pre"]


def test_deepseek_v41_every_layer_launches_exactly_the_work_vllm_runs():
    """Each layer's Elementwise launch count and total traffic equal what the config
    assigns it, so a launch the handler forgets (an engram gate, a norm, the inverse
    RoPE, a head mixer, a candidate mask) is missing from the sum rather than invisible.
    Comparing totals lets a handler regroup launches that move the same bytes."""
    g, t = v41_graph(), text_config(config_of(V41))
    for where, i, nodes, ratio in v41_all_launch_sites(g, t):
        launches = [n for n in nodes if n["op"] == "Elementwise"]
        want = v41_required_launches(t, where, i)
        caches = caches_of(nodes)
        assert len(launches) == len(want), (where, i, [n["role"] for n in launches], want)
        assert sum(per_token(n, caches) for n in launches) == sum(
            v41_expected_traffic(t, r, ratio) or 0 for r in want), (where, i)
        assert sum(1 for n in launches if "context" in n) == sum(
            1 for r in want if v41_expected_context(t, r, ratio)), (where, i)


def test_deepseek_v41_largest_launches_move_the_bytes_their_kernels_state():
    """Hand-computed figures for the five largest per-token launches, so a change to an
    assumption shared by the handler and v41_expected_traffic (a record size, a width)
    breaks a literal rather than moving both sides together."""
    g = v41_graph()
    layers = expanded_layers(g)
    caches = caches_of(layers[2])
    by_role = {n["role"]: per_token(n, caches) for n in layers[2]
               if n["op"] == "Elementwise" and "context" not in n}
    # residual 4 x 5120 BF16 = 40,960 per pass; post 10,240 + 2 x 40,960 + 80, pre
    # 40,960 + 80 + 10,240 (kernels/mhc/tilelang.py:296-358)
    assert by_role["mhc_attn_post_pre"] == 143_520
    # Q 64 x 512 BF16 read and written, KV 1,024 read, one 584 B record written
    # (attention.py:1034-1048, 503-508)
    assert by_role["q_rope_kv_insert"] == 132_680
    assert by_role["o_inv_rope"] == 131_072
    # decode at ratio 2: 2 x 4,096 + (2 x 4,096 + 1,024) / 2
    # (fused_compress_quant_cache.py:118-221)
    assert by_role["compressor_pool_norm"] == 12_800
    # at ratio 1 (layer 20): one FP32 row of head_dim read, one BF16 latent written
    l20 = {n["role"]: n["bytes_per_token"] for n in layers[20]
           if n.get("role") == "compressor_pool_norm"}
    assert l20["compressor_pool_norm"] == 3_072
    # one 1,024 B latent read and one 584 B record written per pair of tokens
    assert by_role["compressor_rope_quant_insert"] == 804
    engram = {n["role"]: n["bytes_per_token"] for n in layers[1]
              if n["op"] == "Elementwise" and "bytes_per_token" in n}
    # bundle, keys and the shared value read, bundle written (common/engram.py:755-843)
    assert engram["engram_gate"] == 133_120


def test_deepseek_v41_main_read_is_the_latent_record_every_layer_caches():
    """Every target and draft layer reads a one-head latent cache of head_dim, RoPE
    included, with qk_rope_head_dim of it rotated, stored as a 584 B record on SM90.
    head_dim is inclusive of the RoPE width (compressor.py:209-210), and the record is
    what a consumer multiplies by the selected positions to price a decode read."""
    g, t = v41_graph(), text_config(config_of(V41))
    for nodes in expanded_layers(g) + expanded_layers(g, g["speculator"]["stack"]):
        read = compressed_read(nodes)
        assert (read["n_kv"], read["d_h"], read["qk_rope_head_dim"], read["n_q"]) == (
            1, t["head_dim"], t["qk_rope_head_dim"], t["num_attention_heads"])
        assert {c["bytes_per_entry"] for c in read["caches"] if c["kind"] != "state"} == {
            584}
        for scan in scoring_scans(nodes):
            assert (scan["n_q"], scan["n_kv"], scan["d_h"], scan["tp"]) == (
                t["index_n_heads"], 1, t["index_head_dim"], "replicated")
            assert [c["bytes_per_entry"] for c in scan["caches"]] == [
                t["index_head_dim"] + t["index_head_dim"] // 128 * 4]


def test_deepseek_v41_splits_across_tensor_parallel_ranks_where_vllm_does():
    """Each node's tensor-parallel layout, which sets what one rank computes and holds.

    Split by heads: wq_b (column), the grouped wo_a (column, by group) and wo_b (row);
    the inverse RoPE over each rank's own heads (deepseek_v4/nvidia/ops/o_proj.py:61-77);
    the engram wkv (column, common/engram.py:928-937); the lm_head (ParallelLMHead).
    Everything else in attention is replicated: the merged wq_a/wkv (`disable_tp=True`,
    attention.py:354-370), its norm, the compressor, the indexer, and the Q RoPE + insert,
    whose Q write FlashMLA pads to 64 heads per rank (nvidia/flashmla.py:66-74). The
    experts state no tp, being split by expert parallelism instead."""
    g = v41_graph()
    split = {"wq_b": "column", "wo_a": "column", "wo_b": "row", "o_inv_rope": "sharded",
             "engram_wkv": "column", "lm_head": "column", "dspark_lm_head": "column",
             # the shared expert, a dense MLP (nvidia/model.py:159-175)
             "shared_gate_up": "column", "shared_down": "row"}
    spec = g["speculator"]
    nodes = [n for layers in (expanded_layers(g), expanded_layers(g, spec["stack"]),
                              expanded_layers(g, spec["context_stack"]),
                              [g["head"], spec["head"], spec["sequential_head"]])
             for ns in layers for n in ns]
    seen = set()
    for n in nodes:
        if n["op"] in ("AllReduce", "AllGather", "ReduceScatter", "All2All", "Gather"):
            continue
        role = n.get("role")
        if n["op"] == "GroupedGEMM":
            assert "tp" not in n, role
        elif n["op"] == "Elementwise" and "tp" not in n:
            # a hidden-width RMSNorm, sized from hidden_size by the consumer
            assert role.endswith("norm"), role
        elif n["op"] == "Attention" and not role:
            # The main read's per-rank heads are FlashMLA's padding to 64, a property of
            # the backend and the TP width, so its layout is left to the consumer
            # (inference-sim/blis-latency-kernel#39).
            assert "tp" not in n
        else:
            assert n["tp"] == split.get(role, "replicated"), (role, n["tp"])
            seen.add(role)
    assert set(split) <= seen


def test_deepseek_v41_residual_stream_work_runs_on_the_token_shard_under_its_sp():
    """Under V4.1's own sequence parallelism the residual stream is sharded by token
    (sp_shard, nvidia/model.py:898-905), and only attention runs on the gathered rows,
    between its all-gather and its reduce-scatter (:579-586). So exactly the work on the
    stream is token-sharded: every mHC launch and projection, the engram injection's
    wkv and gate (common/engram.py:999-1016), the router and the experts, and the head's
    post, collapse and final norm before it gathers (:976-990; dspark.py:229-248).

    Attention's own work, the compressor and the indexer are not: they run on every
    token. Nor is the engram lookup, which covers every token before the shard
    (nvidia/model.py:890-895), nor anything in the draft's context stack, which reads the
    gathered target states, nor any collective. Stated anywhere else, a node is priced
    at 1/tp of the rows it runs on, or tp times the rows it does."""
    g = v41_graph()
    spec = g["speculator"]
    stream = re.compile(r"mhc_|hc_|engram_gate$|engram_wkv$|router$|experts$|"
                        r"shared_gate_up$|shared_down$|final_norm$")
    def check(nodes, where, expect_head_norm=True):
        for n in nodes:
            sharded = n.get("under_sequence_parallel") == "token_sharded"
            role = n.get("role") or ""
            if n["op"] in ("AllReduce", "AllGather", "ReduceScatter", "All2All"):
                assert not sharded, (where, role)
            elif role == "dspark_final_norm":
                # applied by compute_logits, after the drafter gathers
                assert not sharded, where
            else:
                assert sharded == bool(stream.match(role)), (where, role, sharded)
    for i, nodes in enumerate(expanded_layers(g)):
        check(nodes, i)
        assert sum(n.get("under_sequence_parallel") == "token_sharded"
                   for n in nodes if n["op"] == "GroupedGEMM") == 1, i
    for d, nodes in enumerate(expanded_layers(g, spec["stack"])):
        check(nodes, f"draft {d}")
    check(g["head"], "head")
    check(spec["head"], "draft head")
    for nodes in expanded_layers(g, spec["context_stack"]):
        assert not [n for n in nodes if n.get("under_sequence_parallel")]
    for n in g["embedding"] + spec["embedding"] + spec["sequential_head"]:
        assert "under_sequence_parallel" not in n, n.get("role")


def v41_collectives(nodes: list[dict], sequence_parallel: bool) -> collections.Counter:
    """The collectives a layer launches under V4.1's own sequence parallelism or
    without it, at tensor and expert parallelism both above one."""
    live = {"tensor_parallel", "expert_parallel"} | (
        {"sequence_parallel"} if sequence_parallel
        else {"tensor_parallel_unless_sequence_parallel"})
    return collections.Counter((n["op"], n.get("role")) for n in nodes
                               if n.get("emit") in live
                               and n["op"] in ("AllReduce", "AllGather", "ReduceScatter",
                                               "All2All"))


def test_deepseek_v41_each_layer_reduces_and_gathers_what_vllm_does():
    """Per layer, in both of V4.1's tensor-parallel modes (nvidia/model.py
    _use_sequence_parallel: EP on, TP > 1, and mega-MoE or DP > 1).

    Without sequence parallelism the residual stream is replicated: attention's output
    is all-reduced after wo_b, the MoE output all-reduced, and an engram layer
    all-gathers its rows and its wkv output across heads (common/engram.py:928-937,
    996). With it the stream is sharded by token: attention's input is all-gathered and
    its output reduce-scattered, the MoE runs on the local tokens with no reduction, and
    the engram rows are gathered along tokens. Either way the experts dispatch over
    EP, and the hidden state is gathered before the head only under sequence
    parallelism. Each mode must move the residual stream back to the layout it started
    in, so every layer gathers once for each reduce-scatter."""
    g, t = v41_graph(), text_config(config_of(V41))
    sites = ([(i, n, i in t["engram_layer_ids"]) for i, n in enumerate(expanded_layers(g))]
             + [(-1 - d, n, False)
                for d, n in enumerate(expanded_layers(g, g["speculator"]["stack"]))])
    for i, nodes, engram in sites:
        replicated, sharded = v41_collectives(nodes, False), v41_collectives(nodes, True)
        want_rep = collections.Counter({("AllReduce", "attn_out"): 1,
                                        ("AllReduce", "mlp_out"): 1,
                                        ("All2All", "moe_dispatch_combine"): 1})
        want_sp = collections.Counter({("AllGather", "attn_in"): 1,
                                       ("ReduceScatter", "attn_out"): 1,
                                       ("All2All", "moe_dispatch_combine"): 1})
        if engram:
            want_rep += {("AllGather", "engram_rows"): 1, ("AllGather", "engram_wkv_out"): 1}
            want_sp += {("AllGather", "engram_rows_by_token"): 1}
        assert replicated == want_rep, (i, replicated)
        assert sharded == want_sp, (i, sharded)
    # The lookups reduce before the stream is sharded, so in both modes.
    for embedding in (g["embedding"], g["speculator"]["embedding"]):
        for sp in (False, True):
            assert [op for op, _ in v41_collectives(embedding, sp)] == ["AllReduce"]
    for head in (g["head"], g["speculator"]["head"]):
        assert not v41_collectives(head, False)
        assert sum(v41_collectives(head, True).values()) == 1
        assert [n["op"] for n in head if n.get("emit")] == ["AllGather"]


def test_deepseek_v41_engram_gathers_move_the_rows_and_the_projection_they_feed():
    """The engram collectives move what the lookup and wkv produce for every token: the
    24 gathered rows at BF16 head_dim, and wkv's BF16 output for every column of the
    column-split projection. The two modes gather the same rows, along heads or along
    tokens."""
    g, t = v41_graph(), text_config(config_of(V41))
    for i in t["engram_layer_ids"]:
        nodes = expanded_layers(g)[i]
        (lookup,) = [n for n in nodes if n["op"] == "Gather"]
        (wkv,) = [n for n in nodes if n.get("role") == "engram_wkv"]
        moved = {n["role"]: n["bytes_per_token"] for n in nodes if n["op"] == "AllGather"
                 and n.get("bytes_per_token")}
        assert moved["engram_rows"] == moved["engram_rows_by_token"] == (
            lookup["out_bytes_per_token"])
        # (hc_mult + 1) x hidden BF16 values (common/engram.py:928-937)
        assert moved["engram_wkv_out"] == (t["hc_mult"] + 1) * t["hidden_size"] * 2 == 51_200
        assert wkv["tp"] == "column"


def v41_derive_from_file(tmp_path, raw: dict) -> dict:
    """The real derive(), through a config.json on disk."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return deriver().derive(path, "synthetic")


def test_deepseek_v41_derive_applies_the_head_and_the_speculator_the_handler_states(
        tmp_path):
    """Through derive() itself: the head runs the mHC post and collapse before the final
    norm and prices lm_head at BF16; the global block carries the architecture, the
    resident scale and the 1M-token context window; and the speculator is the
    handler's DSpark module -- parallel block, context stack, aliased head, sequential
    Markov head -- rather than derive()'s generic MTP fallback. A checkpoint with no
    draft layers gets no speculator, although num_nextn_predict_layers is the field the
    fallback reads."""
    raw = config_of(V41)
    g = v41_derive_from_file(tmp_path, raw)
    t = text_config(raw)
    head_traffic = sum(n.get("bytes_per_token") or 0 for n in g["head"])
    assert head_traffic == (v41_expected_traffic(t, "mhc_final_post", 0)
                            + v41_expected_traffic(t, "hc_collapse", 0))
    lm = [n for n in g["head"] if n["op"] == "GEMM"]
    assert [(n["n"], n["k"], n["weight_dtype"]) for n in lm] == [
        (t["vocab_size"], t["hidden_size"], "bf16")]
    assert (g["global"]["architecture"], g["global"]["context_window"],
            g["global"]["weight_scale"]) == (
        "DeepseekV41ForCausalLM", t["max_position_embeddings"],
        {"elements": 32, "bytes": 1})
    spec = g["speculator"]
    assert (spec["method"], spec["num_spec_tokens"], spec["drafting"]) == (
        "dspark", t["dspark_block_size"], "parallel_block")
    assert spec["context_stack"] and spec["head"] and spec["sequential_head"]
    assert g["embedding"] and g["arch_overrides"]

    bare = dict(raw, text_config=dict(t, num_nextn_predict_layers=0,
                                      compress_ratios=t["compress_ratios"][:40]))
    assert "speculator" not in v41_derive_from_file(tmp_path, bare)


def test_deepseek_v41_omitted_draft_ratios_price_like_stated_zeros():
    """Metamorphic: vLLM gives a layer id past the end of compress_ratios ratio 0
    (attention.py:300-305), so a vector that omits the three draft entries prices
    exactly like one that states them as 0."""
    t = v41_config()
    short = v41_config(compress_ratios=t["compress_ratios"][:t["num_hidden_layers"]])
    a, b = v41_derive(t), v41_derive(short)
    assert cost_signature(a) == cost_signature(b)


def test_deepseek_v41_main_proj_reads_one_hidden_state_per_target_layer():
    """Metamorphic: dropping a DSpark target layer narrows main_proj's input by one
    hidden width (dspark.py:100-108) and changes nothing the draft layers pay."""
    full = v41_derive(v41_config())
    two = v41_derive(v41_config(dspark_target_layer_ids=[38, 39]))
    hidden = text_config(config_of(V41))["hidden_size"]
    def main_proj(g):
        return [n["k"] for nodes in expanded_layers(g, g["speculator"]["context_stack"])
                for n in nodes if n.get("role") == "dspark_main_proj"]
    assert main_proj(full) == [3 * hidden] and main_proj(two) == [2 * hidden]
    draft = [expanded_layers(g, g["speculator"]["stack"]) for g in (full, two)]
    assert [[node_cost(n) for n in nodes] for nodes in draft[0]] == [
        [node_cost(n) for n in nodes] for nodes in draft[1]]


def v4_graph() -> dict:
    d = MODELS / "deepseek-v4-pro"
    assert d.is_dir(), "models/deepseek-v4-pro is missing"
    return graph_of(d)


V4 = MODELS / "deepseek-v4-pro"
# Matrix shapes and stored dtypes from all 64 safetensors shard headers of
# deepseek-ai/DeepSeek-V4-Pro at b5968e9190ef611bbf34a7229255be88a0e937c1 (145,116
# tensors; their byte sizes sum to the index's total_size). (out, in, dtype). Routed
# experts are packed two FP4 values per I8 byte, so [3072, 3584] stores 3072 x 7168.
V4_CHECKPOINT = {
    "every_layer": [
        (1536, 7168, "fp8"),     # attn.wq_a
        (512, 7168, "fp8"),      # attn.wkv
        (65536, 1536, "fp8"),    # attn.wq_b
        (16384, 4096, "fp8"),    # attn.wo_a
        (7168, 16384, "fp8"),    # attn.wo_b
        (384, 7168, "bf16"),     # ffn.gate
        (3072, 7168, "fp8"),     # ffn.shared_experts.w1
        (3072, 7168, "fp8"),     # ffn.shared_experts.w3
        (7168, 3072, "fp8"),     # ffn.shared_experts.w2
        (24, 28672, "fp32"),     # hc_attn_fn
        (24, 28672, "fp32"),     # hc_ffn_fn
    ],
    "routed_expert": [(3072, 7168, "mxfp4"), (3072, 7168, "mxfp4"),
                      (7168, 3072, "mxfp4")],
    "compressor": {128: [(512, 7168, "bf16"), (512, 7168, "bf16")],      # wkv, wgate
                   4: [(1024, 7168, "bf16"), (1024, 7168, "bf16")]},
    "indexer": [(8192, 1536, "fp8"), (64, 7168, "bf16"),                 # wq_b, weights
                (256, 7168, "bf16"), (256, 7168, "bf16")],               # its compressor
    "mtp": [(7168, 7168, "fp8"), (7168, 7168, "fp8")],                   # e_proj, h_proj
    "hc_head": [(4, 28672, "fp32")],                                     # hc_head_fn
    "head": [(129280, 7168, "bf16")],
}


def v4_checkpoint_layer(t: dict, i: int) -> list[tuple[int, int, str]]:
    mats = list(V4_CHECKPOINT["every_layer"])
    mats += V4_CHECKPOINT["routed_expert"] * t["n_routed_experts"]
    r = t["compress_ratios"][i]
    mats += V4_CHECKPOINT["compressor"][r]
    if r == 4:
        mats += V4_CHECKPOINT["indexer"]
    if i == 0:
        # The broadcast pre projects the embedding with hc_attn_fn summed over its
        # hc_mult copies (nvidia/model.py:1821-1834): it multiplies by (24, 7168), not
        # (24, 28672). The (24, 28672) tensor stays resident, unmultiplied, which the
        # handler's docstring records rather than prices.
        mats.remove((24, 28672, "fp32"))
        mats.append((24, 7168, "fp32"))
    return mats


def check_weights(nodes, want, where, glob="fp8"):
    priced = [weight_cost(n, glob) for n in nodes]
    assert sum(e for e, _ in priced) == sum(n * k for n, k, _ in want), where
    assert sum(b for _, b in priced) == sum(n * k * WIDTH[dt] for n, k, dt in want), where


def test_deepseek_v4_charges_the_weights_the_checkpoint_stores():
    """Every matrix V4-Pro multiplies by, layer by layer, at the checkpoint's element
    count and stored width, which on V4 is also the resident width: the dense linears
    and the shared expert stay FP8 (wo_a included, as a DeepGEMM FP8 einsum,
    nvidia/ops/o_proj.py:61-95), the experts MXFP4, the router, compressors and indexer
    weights BF16, the mHC projections FP32. The MTP layer and both heads are checked
    against the mtp.0.* and top-level tensors; aliases hold nothing."""
    g, t = v4_graph(), text_config(config_of(V4))
    glob = g["global"]["weight_dtype"]
    assert glob == "fp8"
    layers = expanded_layers(g)
    assert len(layers) == t["num_hidden_layers"]
    for i, nodes in enumerate(layers):
        check_weights(nodes, v4_checkpoint_layer(t, i), i, glob)
    spec = g["speculator"]
    (draft,) = expanded_layers(g, spec["stack"])
    check_weights(draft, V4_CHECKPOINT["every_layer"] + V4_CHECKPOINT["mtp"]
                  + V4_CHECKPOINT["routed_expert"] * t["n_routed_experts"], "mtp", glob)
    check_weights(g["head"], V4_CHECKPOINT["head"] + V4_CHECKPOINT["hc_head"], "head", glob)
    check_weights(spec["head"], V4_CHECKPOINT["hc_head"], "mtp head", glob)
    # The draft's embedding and lm_head are the target's (nvidia/mtp.py; the checkpoint
    # has no mtp.0.embed or mtp.0.head).
    assert [n.get("aliases") for n in spec["embedding"] if n["op"] == "Gather"] == [
        "embedding.embed_tokens"]
    assert [n.get("aliases") for n in spec["head"] if n["op"] == "GEMM"
            and n["n"] == t["vocab_size"]] == ["head.lm_head"]


def test_deepseek_v4_hash_layers_route_through_a_token_table():
    """Layers below num_hash_layers pick their experts from tid2eid[token], an INT32
    [vocab, top_k] table on the default path (nvidia/model.py:866-880), and still run
    the gate GEMM, whose logits weight the chosen experts (moe_runner.py:903-910). The
    checkpoint carries tid2eid on layers 0-2 alone."""
    g, t = v4_graph(), text_config(config_of(V4))
    for i, nodes in enumerate(expanded_layers(g)):
        gathers = [n for n in nodes if n["op"] == "Gather"]
        routers = [n for n in nodes if n.get("role") == "router"]
        assert len(routers) == 1, i
        if i < t["num_hash_layers"]:
            assert [(n["rows"], n["row_bytes"], n["rows_per_token"],
                     n["out_bytes_per_token"]) for n in gathers] == [
                (t["vocab_size"], t["num_experts_per_tok"] * 4, 1,
                 t["num_experts_per_tok"] * 4)], i
        else:
            assert not gathers, i


def v4_layer_facts(nodes: list[dict]) -> dict:
    read = compressed_read(nodes)
    scans = scoring_scans(nodes)
    held = {c["name"]: c for n in nodes for c in n.get("caches", [])}
    gemms = {n.get("role"): n for n in nodes if n["op"] == "GEMM"}
    return {
        "read": (read["window"], read.get("compress_ratio"), read.get("index_topk")),
        "caches": sorted(held),
        "compressor_width": gemms.get("compressor_fused_wkv_wgate", {}).get("n"),
        "scores": [(s["n_q"], s["d_h"], s.get("compress_ratio")) for s in scans],
        "hash": any(n["op"] == "Gather" for n in nodes),
        "broadcast": "hc_attn_fn_broadcast" in gemms,
        "prefill_dequant": any(n.get("role") == "prefill_dequant_gather" for n in nodes),
    }


def v4_expected(t: dict, i: int) -> dict:
    r = t["compress_ratios"][i]
    return {
        "read": (t["sliding_window"], r, t["index_topk"] if r == 4 else None),
        "caches": sorted(["swa", "compressed", "compressor_state"]
                         + (["index", "index_compressor_state"] if r == 4 else [])),
        # wkv + wgate, each coff x head_dim (compressor.py:239-240, 271-279)
        "compressor_width": 2 * (2 if r == 4 else 1) * t["head_dim"],
        "scores": [(t["index_n_heads"], t["index_head_dim"], 4)] if r == 4 else [],
        "hash": i < t["num_hash_layers"],
        "broadcast": i == 0,
        "prefill_dequant": True,
    }


def test_deepseek_v4_each_layer_pays_for_what_its_ratio_assigns():
    """Per layer, from the config's compress_ratios alone: every layer reads its window;
    ratio 4 adds the indexer's top-k and ratio 128 every compressed entry; each owns its
    own compressor and caches (attention.py:356-374), the ratio-4 ones an indexer with
    its own key cache and state; the first num_hash_layers hash-route; layer 0 takes the
    broadcast mHC pre."""
    g, t = v4_graph(), text_config(config_of(V4))
    for i, nodes in enumerate(expanded_layers(g)):
        assert v4_layer_facts(nodes) == v4_expected(t, i), i
        for c in (c for n in nodes for c in n.get("caches", [])):
            assert c["owned"], (i, c["name"])     # V4 shares no cache


def v4_config(**over) -> dict:
    t = dict(text_config(config_of(V4)))
    t.update(over)
    return t


def v4_derive(t: dict) -> dict:
    kinds, stack, extras = deriver().handler_deepseek_v4(
        t, dict(config_of(V4), text_config=t), "synthetic")
    return {"layer_kinds": kinds, "stack": stack, "speculator": extras.get("speculator"),
            "head": extras["head"], "embedding": extras["embedding"],
            "global": {"weight_dtype": "fp8", **extras["global"]}}


def test_deepseek_v4_layer_rules_hold_for_any_ratio_vector():
    """The same rule as a property: 200 seeded draws of depth, ratio sequence and hash
    depth, each derived graph charging every layer what v4_expected says, and no
    two layers sharing a kind while building different work."""
    import random
    rng = random.Random(4001)
    for draw in range(200):
        layers = rng.randint(2, 16)
        ratios = [rng.choice((4, 128)) for _ in range(layers)]
        t = v4_config(num_hidden_layers=layers, compress_ratios=ratios + [0],
                      num_hash_layers=rng.randint(0, layers))
        g = v4_derive(t)
        built = expanded_layers(g)
        assert len(built) == layers, draw
        for i, nodes in enumerate(built):
            assert v4_layer_facts(nodes) == v4_expected(t, i), (draw, i, ratios)


def test_deepseek_v4_turning_a_layer_to_ratio_4_adds_exactly_an_indexer():
    """Metamorphic: changing one layer from ratio 128 to 4 adds that layer's indexer
    (its projections, compressor, keys and scorer), doubles its compressor's width,
    and changes no other layer's price. KV per context token moves by exactly
    584/4 + 132/4 - 584/128 bytes."""
    base = v4_config()
    i = next(j for j, r in enumerate(base["compress_ratios"][:base["num_hidden_layers"]])
             if r == 128 and j > 2)
    flipped = v4_config(compress_ratios=[4 if j == i else r
                                         for j, r in enumerate(base["compress_ratios"])])
    gb, gf = v4_derive(base), v4_derive(flipped)
    for j, (a, b) in enumerate(zip(work_by_layer(gb), work_by_layer(gf))):
        if j != i:
            assert a == b, j
    lb, lf = expanded_layers(gb)[i], expanded_layers(gf)[i]
    added = {n.get("role") for n in lf} - {n.get("role") for n in lb}
    assert added == {"index_wq_b", "index_weights_proj",
                     "index_compressor_fused_wkv_wgate", "index_compressor_save_state",
                     "index_compressor_compress_store", "index_q_rope_quant",
                     "block_index_scores", "index_logits_topk", "index_prefill_k_gather",
                     "compressor_compress_store"}, added
    assert gemm_elements(lf, "compressor_fused_wkv_wgate") == 2 * gemm_elements(
        lb, "compressor_fused_wkv_wgate")
    assert (v41_kv_bytes_per_context_token(gf) - v41_kv_bytes_per_context_token(gb)
            == 584 / 4 + 132 / 4 - 584 / 128)


def test_deepseek_v4_hash_depth_moves_exactly_the_token_table():
    """Metamorphic: one more hash layer adds exactly one tid2eid gather to that layer
    and changes nothing else; the gate GEMM stays, because the logits still weight the
    experts."""
    base = v4_config()
    deeper = v4_config(num_hash_layers=base["num_hash_layers"] + 1)
    for j, (a, b) in enumerate(zip(work_by_layer(v4_derive(base)),
                                   work_by_layer(v4_derive(deeper)))):
        diff = b - a
        if j == base["num_hash_layers"]:
            assert not a - b and [c[0] for c in diff.elements()] == ["Gather"], j
        else:
            assert a == b, j


def test_deepseek_v4_owned_caches_hold_what_vllm_allocates():
    """KV residency from the stated caches: per context token one 584 B record per 4
    tokens on each of the 30 ratio-4 layers plus a 132 B indexer key per 4, and one
    record per 128 tokens on each of the 31 ratio-128 layers: 30 x 179 + 31 x 4.5625 =
    5,511.4375 B. Every layer and the MTP layer also hold a 128-token window in 64-token
    pages, and each compressor a sliding FP32 state, stated as the rows a decoding
    request holds (v1/kv_cache_interface.py:840-866)."""
    g, t = v4_graph(), text_config(config_of(V4))
    assert v41_kv_bytes_per_context_token(g) == 30 * (584 / 4 + 132 / 4) + 31 * 584 / 128
    states = collections.Counter(
        (c["name"], c["entries"], c["bytes_per_entry"])
        for nodes in expanded_layers(g) for n in nodes for c in n.get("caches", [])
        if c["kind"] == "state")
    # ratio 4: window 8 in blocks of 4 -> 4 pages of 4 rows of 2 x 1024 FP32; ratio
    # 128: window 128 in blocks of 8 -> 18 pages of 8 rows of 2 x 512 FP32; the
    # indexer's: window 8 in blocks of 4 -> 16 rows of 2 x 256 FP32 (compressor.py:
    # 157-173, 282-287), at 1 + 1 tokens in flight.
    assert states == {("compressor_state", 16, 8192): 30,
                      ("compressor_state", 144, 4096): 31,
                      ("index_compressor_state", 16, 2048): 30}
    windows = [c for nodes in expanded_layers(g) + expanded_layers(
                   g, g["speculator"]["stack"])
               for n in nodes for c in n.get("caches", []) if c["kind"] == "window"]
    assert len(windows) == t["num_hidden_layers"] + 1
    assert {(c["window"], c["bytes_per_entry"], c["block_size"], c["page_alignment"])
            for c in windows} == {(128, 584, 64, 576)}


def v4_expected_traffic(t: dict, role: str, ratio: int) -> float | None:
    """What each launch moves per token, restated from the tensors its call site reads
    and writes, cache records and the once-per-entry cadence included."""
    H, act, fp32, hc = t["hidden_size"], 2, 4, t["hc_mult"]
    D, qlora, heads = t["head_dim"], t["q_lora_rank"], t["num_attention_heads"]
    ih, idim = t["index_n_heads"], t["index_head_dim"]
    rec, irec = 584, idim + 4
    bundle, mixes = hc * H * act, (hc + hc * hc) * fp32
    post = H * act + 2 * bundle + mixes
    pre = bundle + mixes + H * act
    coff = 2 if ratio == 4 else 1
    row = 2 * coff * D * fp32
    gathered = coff * ratio * 2 * D * fp32 if ratio > 1 else 0
    table = {
        "mhc_attn_post_pre": post + pre, "mhc_ffn_post_pre": post + pre,
        "mhc_final_post": post, "mtp_mhc_post": post, "mhc_attn_pre": pre,
        # reads the (T, H) embedding, writes the bundle, the mixes and the layer input
        "mhc_attn_broadcast_pre": H * act + bundle + mixes + H * act,
        "hc_head_mix": bundle + H * act, "mtp_hc_head_mix": bundle + H * act,
        "qkv_a_layernorm": 2 * (qlora + D) * act,
        "q_rope_kv_insert": 2 * heads * D * act + D * act + rec,
        # BF16 in; FP8 and one FP32 scale per 128 out
        "o_inv_rope_quant": heads * D * act + heads * D + heads * D // 128 * fp32,
        "compressor_save_state": row + coff * D * fp32 + row,
        "compressor_compress_store": (gathered + rec) / 4,
        "compressor_compress": (gathered + D * fp32) / 128,
        "compressor_store": (D * fp32 + rec) / 128,
        "index_compressor_save_state": 2 * 2 * idim * fp32 + 2 * idim * fp32
                                       + 2 * 2 * idim * fp32,
        "index_compressor_compress_store": (2 * 4 * 2 * idim * fp32 + irec) / 4,
        "index_q_rope_quant": ih * idim * act + ih * act + ih * idim + ih * fp32,
        "mtp_input_norm": 2 * (H * act + bundle),
        "mtp_input_add": 2 * bundle + H * act,
        "final_norm": None, "mtp_final_norm": None,
        "index_logits_topk": None, "prefill_dequant_gather": None,
        "index_prefill_k_gather": None,
    }
    return table[role]


def test_deepseek_v4_charges_each_launch_the_traffic_its_call_site_moves():
    """Every Elementwise launch in the target, the MTP layer and both heads moves what
    v4_expected_traffic restates, with each cache write's record added at its cadence;
    the two context-proportional launches state their context work instead."""
    g, t = v4_graph(), text_config(config_of(V4))
    spec = g["speculator"]
    sites = ([(n, t["compress_ratios"][i]) for i, n in enumerate(expanded_layers(g))]
             + [(n, 1) for n in expanded_layers(g, spec["stack"])]
             + [(g["head"], 1), (spec["head"], 1)])
    checked = 0
    for nodes, ratio in sites:
        caches = caches_of(nodes)
        for n in nodes:
            if n["op"] != "Elementwise":
                continue
            got = per_token(n, caches) if ("bytes_per_token" in n
                                           or "writes_cache" in n) else None
            assert got == v4_expected_traffic(t, n["role"], ratio), (n["role"], ratio)
            checked += 1
    assert checked > 61 * 8, checked
    by_role = {n.get("role"): n.get("context") for n in expanded_layers(g)[2]
               if "context" in n}
    assert by_role == {
        "index_logits_topk": {"bytes_per_entry": 8, "tokens_per_entry": 4,
                              "per": "query_token"},
        "prefill_dequant_gather": {"reads_cache": "compressed", "bytes_per_entry": 1024,
                                   "per": "request", "phase": "prefill"},
        "index_prefill_k_gather": {"reads_cache": "index", "bytes_per_entry": 132,
                                   "per": "request", "phase": "prefill"}}


def test_deepseek_v4_largest_launches_move_the_bytes_their_kernels_state():
    """Hand-computed literals, so a shared assumption cannot move both sides."""
    g = v4_graph()
    l4 = expanded_layers(g)[4]
    c = caches_of(l4)
    by_role = {n["role"]: per_token(n, c) for n in l4
               if n["op"] == "Elementwise" and "context" not in n}
    # residual 4 x 7168 BF16 = 57,344; post 14,336 + 2 x 57,344 + 80, pre 57,344 + 80 +
    # 14,336
    assert by_role["mhc_attn_post_pre"] == 200_864
    # Q 128 x 512 BF16 read and written, KV 1,024 read, one 584 B record written
    assert by_role["q_rope_kv_insert"] == 263_752
    # attention output 131,072 BF16 in, 65,536 FP8 + 512 x 4 B scales out
    assert by_role["o_inv_rope_quant"] == 198_656
    # 8 state rows x (1,024 kv + 1,024 score) FP32 per entry, plus a record, per 4 tokens
    assert by_role["compressor_compress_store"] == (32_768 + 584) / 4


def test_deepseek_v4_each_layer_reduces_and_gathers_what_vllm_does():
    """V4 decides its own sequence parallelism (nvidia/model.py:1180-1188: PP 1, EP on,
    TP > 1, and mega-MoE or DP > 1). Without it, wo_b all-reduces and FusedMoE
    all-reduces shared + routed once (moe_runner.py:506-512). With it, attention's
    input is all-gathered and its output reduce-scattered (nvidia/model.py:1385-1390),
    the MoE reduces nothing, and the whole hc stream -- not a hidden row -- is gathered
    before the learned hc_head (:1634-1635). The embedding all-reduces before the shard
    in both."""
    g = v4_graph()
    spec = g["speculator"]
    for i, nodes in enumerate(expanded_layers(g) + expanded_layers(g, spec["stack"])):
        rep, sp = v41_collectives(nodes, False), v41_collectives(nodes, True)
        want_rep = {("AllReduce", "attn_out"): 1, ("AllReduce", "mlp_out"): 1,
                    ("All2All", "moe_dispatch_combine"): 1}
        want_sp = {("AllGather", "attn_in"): 1, ("ReduceScatter", "attn_out"): 1,
                   ("All2All", "moe_dispatch_combine"): 1}
        if i == len(expanded_layers(g)):
            want_sp[("AllGather", "mtp_out")] = 1
        assert rep == collections.Counter(want_rep), (i, rep)
        assert sp == collections.Counter(want_sp), (i, sp)
    hidden, hc = g["global"]["hidden_size"], text_config(config_of(V4))["hc_mult"]
    (head_in,) = [n for n in g["head"] if n["op"] == "AllGather"]
    assert (head_in["emit"], head_in["bytes_per_token"]) == (
        "sequence_parallel", hc * hidden * 2)
    # the MTP layer gathers its whole hc stream back too (nvidia/mtp.py:206-209)
    (mtp_out,) = [n for nodes in expanded_layers(g, spec["stack"]) for n in nodes
                  if n.get("role") == "mtp_out"]
    assert mtp_out["bytes_per_token"] == hc * hidden * 2
    for embedding in (g["embedding"], spec["embedding"]):
        assert [(n["op"], n["emit"]) for n in embedding if n.get("emit")] == [
            ("AllReduce", "tensor_parallel")]


def test_deepseek_v4_residual_stream_work_runs_on_the_token_shard():
    """Under V4's sequence parallelism the stream is sharded before the first layer
    (nvidia/model.py:1571-1578) and attention alone runs on the gathered rows. So the
    mHC launches and projections, the router, the hash lookup and the experts are
    token-sharded; attention, its compressor and indexer, the head after its gather and
    the MTP input norm (before the draft's shard, nvidia/mtp.py:168-184) are not."""
    g = v4_graph()
    spec = g["speculator"]
    stream = re.compile(r"mhc_|hc_attn_fn|hc_ffn_fn|router$|hash_route$|experts$|"
                        r"shared_gate_up$|shared_down$|mtp_e_proj$|mtp_h_proj$|"
                        r"mtp_input_add$|mtp_mhc_post$")
    for where, nodes in ([(i, n) for i, n in enumerate(expanded_layers(g))]
                         + [("mtp", n) for n in expanded_layers(g, spec["stack"])]
                         + [("head", g["head"]), ("mtp head", spec["head"])]):
        for n in nodes:
            role = n.get("role") or ""
            sharded = n.get("under_sequence_parallel") == "token_sharded"
            if n["op"] in ("AllReduce", "AllGather", "ReduceScatter", "All2All"):
                assert not sharded, (where, role)
            else:
                assert sharded == bool(stream.match(role)), (where, role, sharded)


def test_deepseek_v4_splits_across_tensor_parallel_ranks_where_vllm_does():
    """Split by heads: wq_b, wo_a (column), wo_b (row), the Q insert and the inverse
    RoPE (sharded), lm_head (column). Replicated: the merged wq_a/wkv (`disable_tp`),
    the norms, the compressors, the indexer (ReplicatedLinear, attention.py:974-988),
    the router, mHC, the MTP projections (ReplicatedLinear, nvidia/mtp.py:105-120). The
    main read leaves tp to the consumer, for FlashMLA's head padding."""
    g = v4_graph()
    spec = g["speculator"]
    split = {"wq_b": "column", "wo_a": "column", "wo_b": "row", "lm_head": "column",
             "mtp_lm_head": "column", "q_rope_kv_insert": "sharded",
             "o_inv_rope_quant": "sharded",
             # the shared expert, a dense MLP (nvidia/model.py:159-175)
             "shared_gate_up": "column", "shared_down": "row"}
    for nodes in (expanded_layers(g) + expanded_layers(g, spec["stack"])
                  + [g["head"], spec["head"]]):
        for n in nodes:
            role = n.get("role")
            if n["op"] in ("AllReduce", "AllGather", "ReduceScatter", "All2All",
                           "Gather", "GroupedGEMM"):
                continue
            if n["op"] == "Attention" and not role:
                assert "tp" not in n
            elif n["op"] == "Elementwise" and "tp" not in n:
                assert role.endswith("norm"), role
            else:
                assert n["tp"] == split.get(role, "replicated"), (role, n["tp"])


def test_deepseek_v4_mtp_drafts_one_token_through_a_window_only_layer(tmp_path):
    """Through derive(): method "mtp" (config/speculative.py:1135-1139), one draft token
    by default from num_nextn_predict_layers (:696-707, 1483-1486), run once per drafted
    token; the draft layer reads its window alone, routes over the full 384 experts with
    a learned gate, and has its own e_proj, h_proj and hc_head. The target states
    DeepseekV4ForCausalLM, its 1M-token window and a BF16 lm_head after the hc_head."""
    raw = config_of(V4)
    t = text_config(raw)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    g = deriver().derive(path, "synthetic")
    assert (g["global"]["architecture"], g["global"]["context_window"]) == (
        "DeepseekV4ForCausalLM", t["max_position_embeddings"])
    spec = g["speculator"]
    assert (spec["method"], spec["num_spec_tokens"], spec.get("drafting")) == (
        "mtp", t["num_nextn_predict_layers"], None)
    (draft,) = expanded_layers(g, spec["stack"])
    read = compressed_read(draft)
    assert (read["window"], read.get("compress_ratio"), read.get("index_topk")) == (
        t["sliding_window"], None, None)
    assert not scoring_scans(draft)
    assert [c["name"] for c in read["caches"]] == ["swa"]
    routed = [n for n in draft if n["op"] == "GroupedGEMM" and n["role"] == "experts"]
    assert [(n["experts"], n["top_k"]) for n in routed] == [
        (t["n_routed_experts"], t["num_experts_per_tok"])]
    lm = [n for n in g["head"] if n["op"] == "GEMM" and n["n"] == t["vocab_size"]]
    assert [(n["k"], n["weight_dtype"]) for n in lm] == [(t["hidden_size"], "bf16")]

    bare = dict(raw, text_config=dict(t, num_nextn_predict_layers=0,
                                      compress_ratios=t["compress_ratios"][:61]))
    path.write_text(json.dumps(bare))
    assert "speculator" not in deriver().derive(path, "synthetic")


@pytest.mark.parametrize("over, fragment", [
    ({"compress_ratios": [0] + [4] * 60 + [0]}, "accepts 4 and 128"),
    ({"compress_ratios": [2] * 61 + [0]}, "accepts 4 and 128"),
    ({"compress_ratios": [4] * 60}, "compress_ratios has 60"),
    ({"compress_ratios": [4.0] * 61 + [0]}, "not a whole number"),
    ({"head_dim": 576}, "head_dim 512 with rope 64"),
    ({"index_head_dim": 256}, "index_head_dim 256"),
    ({"num_nextn_predict_layers": 2}, "one draft layer"),
    ({"num_hash_layers": 62}, "num_hash_layers 62"),
    ({"num_hash_layers": -1}, "num_hash_layers -1"),
    ({"o_groups": 7}, "o_groups 7 does not divide"),
    ({"num_experts_per_tok": 385}, "exceeds 384 experts"),
    ({"expert_dtype": "fp6"}, "expert_dtype 'fp6'"),
    ({"hc_mult": 0}, "hc_mult is 0"),
])
def test_deepseek_v4_refuses_layouts_vllm_would_not_serve(over, fragment):
    """Each is a layout vLLM v0.31.0 rejects or reads at a width no node states."""
    with pytest.raises(deriver().DeriveError, match=fragment):
        v4_derive(v4_config(**over))


def v4_required_launches(t: dict, where: str, i: int) -> list[str]:
    """The Elementwise launches vLLM v0.31.0 runs for one V4 target layer, the MTP
    layer or a head, from the config alone (no graph is read)."""
    if where == "head":
        return ["mhc_final_post", "hc_head_mix", "final_norm"]
    if where == "mtp_head":
        return ["mtp_hc_head_mix", "mtp_final_norm"]
    attn = ["qkv_a_layernorm", "q_rope_kv_insert"]
    tail = ["o_inv_rope_quant", "mhc_ffn_post_pre"]
    if where == "mtp":
        return (["mtp_input_norm", "mtp_input_add", "mhc_attn_pre"] + attn + tail
                + ["mtp_mhc_post"])
    r = t["compress_ratios"][i]
    work = (["mhc_attn_broadcast_pre"] if i == 0 else ["mhc_attn_post_pre"]) + attn
    work += ["compressor_save_state"] + (
        ["compressor_compress_store"] if r == 4 else ["compressor_compress",
                                                      "compressor_store"])
    if r == 4:
        work += ["index_compressor_save_state", "index_compressor_compress_store",
                 "index_q_rope_quant", "index_logits_topk", "index_prefill_k_gather"]
    return work + ["prefill_dequant_gather"] + tail


V4_CONTEXT_ROLES = {"index_logits_topk", "prefill_dequant_gather",
                    "index_prefill_k_gather"}


def v4_launch_sites(g: dict, t: dict):
    spec = g.get("speculator")
    sites = [("target", i, n, t["compress_ratios"][i])
             for i, n in enumerate(expanded_layers(g))] + [("head", 0, g["head"], 1)]
    if spec:
        sites += [("mtp", 0, n, 1) for n in expanded_layers(g, spec["stack"])]
        sites += [("mtp_head", 0, spec["head"], 1)]
    return sites


def check_v4_launches(g: dict, t: dict, where_tag=""):
    for where, i, nodes, ratio in v4_launch_sites(g, t):
        launches = [n for n in nodes if n["op"] == "Elementwise"]
        want = v4_required_launches(t, where, i)
        caches = caches_of(nodes)
        assert len(launches) == len(want), (where_tag, where, i,
                                            [n["role"] for n in launches], want)
        assert sum(per_token(n, caches) for n in launches) == sum(
            v4_expected_traffic(t, r, ratio) or 0 for r in want), (where_tag, where, i)
        assert sum(1 for n in launches if "context" in n) == sum(
            1 for r in want if r in V4_CONTEXT_ROLES), (where_tag, where, i)


def test_deepseek_v4_every_layer_launches_exactly_the_work_vllm_runs():
    """Each layer's Elementwise launch count, total traffic and context-work count
    equal what the config assigns it, so a launch the handler drops (a state save, the
    ratio-128 compress, the MTP input add, the head's final post) is missing from the
    sum rather than invisible. The MTP layer gathers no hash table and runs no prefill
    dequant: it has neither a hash table nor a compressed stream."""
    g, t = v4_graph(), text_config(config_of(V4))
    check_v4_launches(g, t)
    (draft,) = expanded_layers(g, g["speculator"]["stack"])
    assert not [n for n in draft if n["op"] == "Gather"]
    assert not [n for n in draft if "context" in n]


def test_deepseek_v4_launch_rules_hold_for_any_ratio_vector():
    """The launch rule as a property, over 100 generated depths, ratio sequences and
    hash depths, with and without the MTP layer."""
    import random
    rng = random.Random(4002)
    for draw in range(100):
        layers = rng.randint(2, 12)
        ratios = [rng.choice((4, 128)) for _ in range(layers)]
        spec = rng.choice((0, 1))
        t = v4_config(num_hidden_layers=layers, compress_ratios=ratios + [0] * spec,
                      num_hash_layers=rng.randint(0, layers),
                      num_nextn_predict_layers=spec)
        check_v4_launches(v4_derive(t), t, draw)


def test_deepseek_v4_states_the_pages_vllm_allocates():
    """Each owned cache's record, tokens per entry, page size and alignment, as
    literals: 584 B records in 64-token window pages and 256-token compressed pages,
    132 B indexer keys in 256-token pages, all padded to FlashMLA's 576 B stride
    (sparse_swa.py:82-84; sparse_mla.py:60-61; attention.py:865-879, 1010-1021)."""
    g = v4_graph()
    held = {(c["name"], c["bytes_per_entry"], c.get("tokens_per_entry", 1),
             c.get("block_size"), c.get("page_alignment"))
            for nodes in expanded_layers(g) + expanded_layers(g, g["speculator"]["stack"])
            for n in nodes for c in n.get("caches", []) if c["kind"] != "state"}
    assert held == {("swa", 584, 1, 64, 576), ("compressed", 584, 4, 256, 576),
                    ("compressed", 584, 128, 256, 576), ("index", 132, 4, 256, 576)}


def test_deepseek_v4_state_rows_follow_the_tokens_in_flight():
    """Without a draft, one token is in flight per decode step, so each state holds
    one page fewer: cdiv(window - 1 + 1, block) + 1 pages -- 12 rows at ratio 4 and
    for the indexer, 136 at ratio 128 -- where MTP's two tokens give 16, 144 and 16
    (v1/kv_cache_interface.py:840-866)."""
    t = v4_config(num_nextn_predict_layers=0,
                  compress_ratios=v4_config()["compress_ratios"][:61])
    g = v4_derive(t)
    states = {(c["name"], c["bytes_per_entry"], c["entries"])
              for nodes in expanded_layers(g) for n in nodes
              for c in n.get("caches", []) if c["kind"] == "state"}
    assert states == {("compressor_state", 8192, 12), ("compressor_state", 4096, 136),
                      ("index_compressor_state", 2048, 12)}


def test_deepseek_v41_states_the_pages_vllm_allocates():
    """V4.1's owned caches, as literals, on SM90 and under the SM100 overlay: 584 B
    records in 32-token window pages (attention.py:547-558) and 64-token compressed
    and indexer pages on SM90 (sparse_mla.py:89-90; indexer.py:277-278), padded to
    576 B; the 132 B key; the 8-row FP32 ring. SM100 stores the 528 B MXFP8 record,
    pads to 512 B, and adds the 288 B NVFP4 compressed record (attention.py:500-517)."""
    g = v41_graph()
    def held(graph):
        return {(c["name"], c["bytes_per_entry"], c.get("tokens_per_entry", 1),
                 c.get("block_size"), c.get("page_alignment"),
                 frozen(c.get("bytes_by_cache_dtype")), c.get("entries"))
                for nodes in expanded_layers(graph) + expanded_layers(
                    graph, graph["speculator"]["stack"])
                for n in nodes for c in n.get("caches", []) if c["owned"]}
    assert held(g) == {
        ("swa", 584, 1, 32, 576, None, None),
        ("compressed", 584, 2, 64, 576, None, None),
        ("compressed", 584, 1, 64, 576, None, None),
        ("index", 132, 2, 64, 576, None, None),
        ("index", 132, 1, 64, 576, None, None),
        ("compressor_ring", 4096, 1, None, None, None, 8)}
    assert held(apply_capability_overlay(g, 10)) == {
        ("swa", 528, 1, 32, 512, None, None),
        ("compressed", 528, 2, 64, 512, frozen({"nvfp4_ds_mla": 288}), None),
        ("compressed", 528, 1, 64, 512, frozen({"nvfp4_ds_mla": 288}), None),
        ("index", 132, 2, 64, 512, None, None),
        ("index", 132, 1, 64, 512, None, None),
        ("compressor_ring", 4096, 1, None, None, None, 8)}


def test_deepseek_v41_ring_grows_with_the_draft_block():
    """The compressor ring is max(8, the next power of two at or above num_spec_tokens
    + 2) rows (compressor.py:146-163): 8 at DSpark's 5, 16 at a block of 7."""
    g = v41_derive(v41_config(dspark_block_size=7))
    rings = {c["entries"] for nodes in expanded_layers(g) for n in nodes
             for c in n.get("caches", []) if c["kind"] == "state"}
    assert rings == {16}


def test_deepseek_v41_states_the_engram_and_draft_fields_a_consumer_prices():
    """Fields that change a price and that no traffic sum reaches: the engram lookup is
    under the engine's cpu_offload control, wo_a keeps its E8M0 scales resident at one
    byte per 32 (mxfp8/emulation.py:36-50), and the Markov head gathers one BF16 row
    of dspark_markov_rank per position."""
    g, t = v41_graph(), text_config(config_of(V41))
    lookups = [n for nodes in expanded_layers(g) for n in nodes if n["op"] == "Gather"]
    assert {n["residency_control"] for n in lookups} == {"engram_cpu_offload"}
    wo_a = [n for nodes in expanded_layers(g) for n in nodes if n.get("role") == "wo_a"]
    assert {frozen(n["weight_scale"]) for n in wo_a} == {frozen({"elements": 32,
                                                                 "bytes": 1})}
    (w1,) = [n for n in g["speculator"]["sequential_head"] if n["op"] == "Gather"]
    assert w1["out_bytes_per_token"] == t["dspark_markov_rank"] * 2


def test_deepseek_v41_layer_launches_hold_for_any_valid_source_vectors():
    """The launch and traffic rule as a property: over 150 generated source layouts,
    every target layer, draft layer, the context stack and both heads launch exactly
    v41_required_launches at v41_expected_traffic and v41_expected_context, at each
    layer's own ratio, and every reader states its owner's full record. The published
    config puts the candidate source, the masking layers and the borrowing indexers at
    ratio 1; the draws put them at ratio 2 too."""
    import random
    rng = random.Random(4103)
    for draw in range(150):
        t = v41_config(**random_v41_vectors(rng, rng.randint(4, 20)))
        g = v41_derive(t)
        for where, i, nodes, ratio in v41_all_launch_sites(g, t):
            caches = caches_of(nodes)
            launches = [n for n in nodes if n["op"] == "Elementwise"]
            want = v41_required_launches(t, where, i)
            assert len(launches) == len(want), (draw, where, i)
            assert sum(per_token(n, caches) for n in launches) == sum(
                v41_expected_traffic(t, r, ratio) or 0 for r in want), (draw, where, i)
            for n in launches:
                assert n.get("context") == v41_expected_context(t, n["role"], ratio), (
                    draw, where, i, n["role"])
        owners: dict[str, dict] = {}
        for nodes in expanded_layers(g):
            for c in (c for n in nodes for c in n.get("caches", [])):
                if c["owned"]:
                    owners[c["name"]] = c
                else:
                    own = owners[c["name"]]
                    for key in ("kind", "bytes_per_entry", "tokens_per_entry"):
                        assert c.get(key) == own.get(key), (draw, c["name"], key)


@pytest.mark.parametrize("over, fragment", [
    ({"torch_dtype": "float16"}, "BF16"),
    ({"num_nextn_predict_layers": -1}, "num_nextn_predict_layers is -1"),
    ({"candidate_source_layer_id": 21}, "is not an index source"),
    ({"dspark_num_experts_per_tok": 129}, "exceeds 128 draft experts"),
    ({"engram_num_embeddings": [384006168]}, "does not give one table per engram layer"),
    ({"engram_num_embeddings": [384006168, 0]}, "does not give one table per engram layer"),
])
def test_deepseek_v41_refuses_the_remaining_layouts_it_cannot_price(over, fragment):
    with pytest.raises(deriver().DeriveError, match=fragment):
        v41_derive(v41_config(**over))


def test_deepseek_v41_refuses_more_experts_per_token_than_it_routes():
    t = v41_config()
    with pytest.raises(deriver().DeriveError, match="exceeds"):
        v41_derive(v41_config(num_experts_per_tok=t["n_routed_experts"] + 1))


def test_deepseek_v4_refuses_a_non_bf16_base():
    with pytest.raises(deriver().DeriveError, match="BF16"):
        v4_derive(v4_config(torch_dtype="float16"))


@pytest.mark.parametrize("extras, fragment", [
    ({"global": {"hidden_size": 1}}, "restates global"),
    ({"head": [{"op": "Elementwise", "role": "final_norm"}]}, "does not state one"),
    ({"arch_overrides": [{"capability_family": 10}]}, "stated together"),
], ids=["global clash", "head without lm_head", "overlay without base family"])
def test_derive_refuses_extras_that_contradict_it(tmp_path, monkeypatch, extras,
                                                  fragment):
    """derive() computes the global block, prices one LM head and needs the base family
    an overlay applies over; a handler's extras that contradict any of these must stop
    the derivation rather than override it."""
    with pytest.raises(deriver().DeriveError, match=fragment):
        derive_through_stub(tmp_path, monkeypatch, (extras,))


def test_a_deriver_crash_is_reported_against_its_model(tmp_path):
    """An exception the deriver does not raise itself is still recorded against the
    model it came from, and the failures collected before it are still printed."""
    models = tmp_path / "models"
    (models / "a-unknown").mkdir(parents=True)
    (models / "a-unknown" / "config.json").write_text(
        json.dumps({"architectures": ["Qwen9ForCausalLM"]}))
    (models / "b-crash").mkdir()
    # A string layer count makes handler_dense's int() raise ValueError.
    (models / "b-crash" / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "hidden_size": 4096,
        "num_hidden_layers": "forty", "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 32000,
        "torch_dtype": "bfloat16"}))
    r = subprocess.run([sys.executable, str(DERIVER), str(tmp_path), "--check"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    lines = r.stderr.splitlines()
    assert any(ln.startswith("a-unknown: no handler") for ln in lines), r.stderr
    assert any(ln.startswith("b-crash: ValueError:") for ln in lines), r.stderr
    assert "Traceback" not in r.stderr


def derive_through_stub(tmp_path, monkeypatch, tail: tuple, **config) -> dict:
    """The real derive() over a one-kind stub handler returning ([layer], stack, *tail)."""
    dg = deriver()
    layer = {"id": "layer", "nodes": [dg.norm("n")], "edges": []}
    monkeypatch.setitem(dg.HANDLERS, "StubForCausalLM",
                        lambda cfg, raw, model: ([layer], {"prologue": ["layer"]}, *tail))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"architectures": ["StubForCausalLM"], "hidden_size": 64,
                                "vocab_size": 128, "torch_dtype": "bfloat16", **config}))
    return dg.derive(path, "stub")


@pytest.mark.parametrize("extras, fragment", [
    ({"lm_head_dtpye": "bf16"}, "extras name"),
    ({"speculator": {"method": "dspark", "num_spec_tokens": 5,
                     "stack": {"pattern": [], "repeat": 1}}}, "not a draft module"),
    ({"speculator": {"method": "dspark", "num_spec_tokens": 0,
                     "stack": {"pattern": ["layer"], "repeat": 1}}}, "not a draft module"),
    ({"speculator": {"method": "dspark", "num_spec_tokens": 5,
                     "stack": {"pattern": ["undefined"], "repeat": 1}}},
     "not a draft module"),
], ids=["misspelled key", "empty draft stack", "no draft tokens", "undefined draft kind"])
def test_derive_refuses_extras_it_cannot_honour(tmp_path, monkeypatch, extras, fragment):
    """A handler's third result is a contract with derive(). A key derive() does not
    read, or a speculator that drafts nothing or through a kind nobody defined, would
    otherwise leave a default standing or price a draft pass that does not exist."""
    with pytest.raises(deriver().DeriveError, match=fragment):
        derive_through_stub(tmp_path, monkeypatch, (extras,))


def test_derive_refuses_a_handler_result_of_the_wrong_length(tmp_path, monkeypatch):
    with pytest.raises(deriver().DeriveError, match="returned 4 results"):
        derive_through_stub(tmp_path, monkeypatch, ({}, {}))


def test_derive_refuses_a_head_width_that_contradicts_the_exclusion_list(
        tmp_path, monkeypatch):
    """Two sources for one width must agree; the later must not silently win. A
    handler's head that prices lm_head at FP32 contradicts a quant config that excludes
    lm_head, and so leaves it BF16."""
    dg = deriver()
    monkeypatch.setitem(dg.HEAD_UNQUANTIZED, "StubForCausalLM", {"lm_head": "lm_head"})
    head = [dg.norm("final_norm"),
            {"op": "GEMM", "role": "lm_head", "n": 128, "k": 64, "weight_dtype": "fp32"}]
    with pytest.raises(dg.DeriveError, match="exclusion list gives lm_head bf16"):
        derive_through_stub(
            tmp_path, monkeypatch, ({"head": head},),
            quantization_config={"quant_method": "fp8",
                                 "modules_to_not_convert": ["lm_head"]})


def test_m3_indexer_states_its_own_bf16_key_cache():
    """vLLM keeps a key-only indexer cache beside every sparse layer's full K/V.

    MiniMaxM3IndexerCache registers one head of sparse_index_dim per token, stored at
    bf16 unless indexer_kv_dtype says otherwise (models/minimax_m3/common/indexer.py,
    v0.31.0). Without it the graph holds 14,592 B a token fewer than the server and
    BLIS's pool came out 1.5-2.0x the logged "GPU KV cache size"."""
    g, t = _m3()
    idx = [n for k in g["layer_kinds"] for n in k["nodes"]
           if n.get("role") == "block_index_scores"]
    assert idx
    for n in idx:
        caches = n.get("caches") or []
        assert [(c["kind"], c["owned"], c["bytes_per_entry"]) for c in caches] == [
            ("index", True, 2 * t["sparse_attention_config"]["sparse_index_dim"])]


def test_w4a16_group_scales_are_resident():
    """compressed-tensors WNA16 keeps one params_dtype scale per group_size weights.

    Kimi-K2.5 is int4 at group 32 with bf16 scales (compressed_tensors_moe_wna16.py,
    v0.31.0): 1/16 B a parameter that an int4 format with no stated scale leaves out,
    which the KV pool then absorbs."""
    for name in ("kimi-k2.5", "kimi-k2.6"):
        d = ROOT / "models" / name
        if not d.is_dir():
            continue
        g = graph_of(d)
        assert g["global"]["weight_dtype"] == "int4"
        assert g["global"].get("weight_scale") == {"elements": 32, "bytes": 2}, name


# --- Muse Spark ------------------------------------------------------------------------

MUSE = MODELS / "muse-spark-open-816b-a42b"
MUSE_NVFP4 = MODELS / "muse-spark-open-816b-a42b-nvfp4"
MUSE_DSPARK = MODELS / "muse-spark-open-dspark"


def _derive_raw(raw: dict, tmp_path: Path, name: str = "muse-test") -> dict:
    sys.path.insert(0, str(DERIVER.parent))
    import derive_graph  # noqa: E402
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw))
    return derive_graph.derive(p, name)


def _kind_of(g: dict, layer: int) -> dict:
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    return kinds[layer_sequence(g["stack"])[layer]]


def _node(kind: dict, role: str) -> dict:
    found = [n for n in kind["nodes"] if n.get("role") == role]
    assert len(found) == 1, f"{kind['id']}: {len(found)} nodes with role {role!r}"
    return found[0]


def test_muse_spark_layers_follow_layer_types():
    """The 16 full-attention layers (1, 5, ..., 61) read the whole context; the other 46 a
    2048-token window. Read per layer from layer_types, never from a period."""
    raw = config_of(MUSE)
    t = text_config(raw)
    g = graph_of(MUSE)
    seq = layer_sequence(g["stack"])
    assert len(seq) == t["num_hidden_layers"] == 62
    for i, lt in enumerate(t["layer_types"]):
        attn = [
            n for n in _kind_of(g, i)["nodes"] if n["op"] == "Attention"][0]
        if lt == "full_attention":
            assert attn["kind"] == "gqa" and "window" not in attn, i
        else:
            assert attn["kind"] == "swa" and attn["window"] == t["sliding_window"], i
        assert (attn["n_q"], attn["n_kv"], attn["d_h"]) == (128, 16, 64)
    assert sum(lt == "full_attention" for lt in t["layer_types"]) == 16


def test_muse_spark_experts_run_at_the_latent_width():
    """Every layer is an MoE of 256 experts at top-8 whose GEMMs run at moe_hidden_size,
    between a pooler and an unpooler that run every token, with no shared expert."""
    g = graph_of(MUSE)
    for k in g["layer_kinds"]:
        e = _node(k, "experts")
        assert (e["experts"], e["top_k"], e["n"], e["k"], e["latent_size"]) == (
            256, 8, 4096, 8192, 4096)
        assert "shared_experts" not in e
        down, up = _node(k, "routed_expert_down_proj"), _node(k, "routed_expert_up_proj")
        assert (down["n"], down["k"]) == (4096, 8192)
        assert (up["n"], up["k"]) == (8192, 4096)
        assert down["tp"] == up["tp"] == "replicated"
        router = _node(k, "router")
        assert (router["n"], router["k"], router["weight_dtype"]) == (256, 8192, "fp32")


def test_muse_spark_parameters_match_the_checkpoint():
    """820.0 B parameters, the vLLM recipe's own count from this config (experts 798.9 B,
    attention 13.5 B, squeeze/expand 4.2 B, router + embed + lm_head 3.4 B), and 42.8 B
    active in the layers."""
    g = graph_of(MUSE)
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    total = active = 0
    for kid in layer_sequence(g["stack"]):
        for n in kinds[kid]["nodes"]:
            if n["op"] == "GEMM":
                total += n["n"] * n["k"]
                active += n["n"] * n["k"]
            elif n["op"] == "GroupedGEMM":
                per = 3 * n["n"] * n.get("latent_size", n["k"])
                total += per * n["experts"]
                active += per * n["top_k"]
    total += sum(n["n"] * n["k"] for n in g["head"] if n["op"] == "GEMM")
    total += sum(n["rows"] * n["row_bytes"] // 2 for n in g["embedding"] if n["op"] == "Gather")
    assert round(total / 1e9, 1) == 820.0
    assert round(active / 1e9, 1) == 42.8


def test_muse_spark_moe_reduces_one_tensor_per_mode():
    """Under TP the top_k per-route rows reduce before the post-expert norm; under EP with
    every token on every rank only the latent-width combine does; with a dispatch the
    all-to-all moves it. No hidden-width MoE all-reduce exists in any mode."""
    g = graph_of(MUSE)
    for k in g["layer_kinds"]:
        rows = _node(k, "moe_route_rows")
        combine = _node(k, "moe_combine")
        assert rows["emit"] == "tensor_parallel_without_expert_parallel"
        assert rows["bytes_per_token"] == 8 * 4096 * 2
        assert combine["emit"] == "expert_parallel_without_dispatch"
        assert combine["bytes_per_token"] == 4096 * 2
        assert _node(k, "moe_dispatch_combine")["emit"] == "expert_parallel"
        assert not any(n.get("role") == "mlp_out" for n in k["nodes"])


def test_muse_spark_hands_an_fp32_stream_between_pipeline_stages():
    g = graph_of(MUSE)
    assert g["global"]["pipeline_boundary_bytes_per_token"] == 8192 * 4
    assert g["global"]["architecture"] == "MuseSparkForConditionalGeneration"
    assert g["global"]["context_window"] == 1048576
    assert g["modality"] == "text_decoder_of_multimodal"


def test_muse_spark_nvfp4_quantizes_only_the_experts_it_names():
    """ModelOpt NVFP4 leaves attention, the pooler/unpooler, the router, lm_head and the
    experts of layers 0 and 61 at BF16: those two layers are kinds of their own, and every
    other expert node is NVFP4."""
    raw = config_of(MUSE_NVFP4)
    excluded = raw["quantization_config"]["modules_to_not_convert"]
    g = graph_of(MUSE_NVFP4)
    assert g["global"]["weight_dtype"] == "bf16"
    for i in range(62):
        e = _node(_kind_of(g, i), "experts")
        named = f"model.language_model.layers.{i}.mlp.experts" in excluded
        assert e.get("weight_dtype", "bf16") == ("bf16" if named else "nvfp4"), i
    assert {i for i in range(62) if "weight_dtype" not in _node(_kind_of(g, i), "experts")} == {0, 61}
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "GEMM":
                assert n.get("weight_dtype") in (None, "bf16", "fp32"), (k["id"], n["role"])


def test_muse_spark_refuses_what_it_does_not_price(tmp_path):
    raw = config_of(MUSE)
    for mutate, why in [
        (lambda t: t.update(num_shared_experts=1), "shared expert"),
        (lambda t: t["mlp_layer_types"].__setitem__(3, "dense"), "mlp_layer_types"),
        (lambda t: t["layer_types"].__setitem__(3, "chunked_attention"), "layer_types"),
        (lambda t: t.update(moe_hidden_size=16384), "moe_hidden_size"),
        (lambda t: t["layer_types"].pop(), "entries"),
    ]:
        bad = json.loads(json.dumps(raw))
        mutate(bad["text_config"])
        sys.path.insert(0, str(DERIVER.parent))
        import derive_graph  # noqa: E402
        with pytest.raises(derive_graph.DeriveError, match=why):
            _derive_raw(bad, tmp_path)


def test_modelopt_takes_the_base_width_only_where_the_handler_places_each_node(tmp_path):
    """ModelOpt quantizes every module its exclusion list does not name, so a single
    global width is wrong for a partial checkpoint and right only by accident for a whole
    one. Muse Spark's handler places each node's width itself, so the global width is the
    base there; any other architecture is refused, as before this handler existed."""
    D = deriver()
    raw = config_of(MUSE_NVFP4)
    assert D.weight_dtype(text_config(raw), raw, "x") == "bf16"
    whole = json.loads(json.dumps(raw))
    whole["quantization_config"]["quant_algo"] = "FP6"
    with pytest.raises(D.DeriveError, match="quant_algo"):
        D.weight_dtype(text_config(whole), whole, "x")
    # A non-Muse NVFP4 ModelOpt checkpoint, with each spelling of an exclusion list and
    # with none: refused, not priced at BF16 (modules_to_not_convert) or at NVFP4
    # throughout (exclude_modules, nothing).
    dense = config_of(MODELS / "llama-3.1-8b-instruct")
    for extra in ({"exclude_modules": ["lm_head"]}, {"ignore": ["lm_head"]},
                  {"modules_to_not_convert": ["lm_head"]}, {}):
        bad = dict(dense, quantization_config={"quant_method": "modelopt",
                                               "quant_algo": "NVFP4", "group_size": 16,
                                               **extra})
        with pytest.raises(D.DeriveError, match="ModelOpt"):
            _derive_raw(bad, tmp_path, "llama-nvfp4")


def test_modelopt_exclusions_match_as_vllm_matches_them():
    """vLLM's ModelOpt config reads `ignore`, else `exclude_modules`, and never
    modules_to_not_convert (modelopt.py:358-359@0f6b2619); a module is excluded when an
    entry equals it, is a substring of it, or matches it as a wildcard, and a trailing
    `*` also names the module itself (modelopt.py:187-220, 263-281@v0.32.0)."""
    D = deriver()
    experts = "model.layers.{}.feed_forward.experts".format
    pats = D.modelopt_exclusions({"exclude_modules": ["layers.1", "self_attn"],
                                  "modules_to_not_convert": ["lm_head"]})
    assert pats == ["layers.1", "self_attn"]
    # A substring, not a component match: "layers.1" also names layers 10-19.
    assert [i for i in range(62) if D.modelopt_excluded(experts(i), pats)] == [
        1, *range(10, 20)]
    assert D.modelopt_excluded("model.layers.3.self_attn.qkv_proj", pats)
    assert not D.modelopt_excluded("lm_head", pats)
    assert D.modelopt_exclusions({"ignore": ["a"], "exclude_modules": ["b"]}) == ["a"]
    assert D.modelopt_exclusions({"ignore": [], "exclude_modules": ["b"]}) == []
    assert [i for i in range(62) if D.modelopt_excluded(
        experts(i), D.modelopt_exclusions({"exclude_modules": ["*layers.2?.*experts"]}))] == [
        *range(20, 30)]
    assert D.modelopt_exclusions({"exclude_modules": ["model.layers.4*"]}) == [
        "model.layers.4", "model.layers.4.*"]


def test_muse_spark_nvfp4_reads_the_list_vllm_reads(tmp_path):
    """The BF16 expert layers are the ones exclude_modules names; modules_to_not_convert
    is not read. A checkpoint that leaves attention or lm_head quantized is refused,
    because the handler prices both at the base width."""
    D = deriver()
    raw = config_of(MUSE_NVFP4)
    committed = graph_of(MUSE_NVFP4)
    same = lambda a, b: {k: v for k, v in a.items() if k != "derived_from"} == {
        k: v for k, v in b.items() if k != "derived_from"}
    assert same(_derive_raw(raw, tmp_path, MUSE_NVFP4.name), committed)
    no_legacy = json.loads(json.dumps(raw))
    del no_legacy["quantization_config"]["modules_to_not_convert"]
    assert same(_derive_raw(no_legacy, tmp_path, MUSE_NVFP4.name), committed)
    moved = json.loads(json.dumps(raw))
    moved["quantization_config"]["exclude_modules"] = [
        "layers.1", "self_attn", "lm_head"]
    g = _derive_raw(moved, tmp_path, MUSE_NVFP4.name)
    bf16 = {i for i in range(62) if "weight_dtype" not in _node(_kind_of(g, i), "experts")}
    assert bf16 == {1, *range(10, 20)}
    for drop, why in (("self_attn", "attention"), ("lm_head", "lm_head")):
        bad = json.loads(json.dumps(raw))
        bad["quantization_config"]["exclude_modules"].remove(drop)
        with pytest.raises(D.DeriveError, match=why):
            _derive_raw(bad, tmp_path, MUSE_NVFP4.name)


def test_muse_spark_dspark_drafter_is_its_own_speculator():
    """Five dense sliding-window layers draft block_size tokens in one pass; the context
    stack projects the five target layers' hidden states; the Markov head runs per
    position over replicated weights."""
    raw = config_of(MUSE_DSPARK)
    g = graph_of(MUSE_DSPARK)
    sp = g["speculator"]
    assert (sp["method"], sp["num_spec_tokens"], sp["drafting"]) == ("dspark", 7, "parallel_block")
    assert sp["stack"] == {"pattern": ["dspark_layer"], "repeat": 5}
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    attn = [n for n in kinds["dspark_layer"]["nodes"] if n["op"] == "Attention"][0]
    assert (attn["kind"], attn["window"]) == ("swa", 2048)
    fc = _node(kinds["dspark_context"], "dspark_fc")
    assert (fc["n"], fc["k"]) == (8192, 8192 * len(raw["target_layer_ids"]))
    assert [n["role"] for n in sp["sequential_head"]] == ["markov_w1", "markov_w2"]
    assert _node({"id": "h", "nodes": sp["sequential_head"]}, "markov_w2")["tp"] == "replicated"


# --- Auxiliary hidden states and the pipeline boundary --------------------------------

def test_muse_spark_dspark_states_the_target_layers_it_reads():
    """vLLM resolves the drafter's capture points from dflash_config.target_layer_ids,
    adding one (get_eagle3_aux_layers_from_config, eagle3_utils.py:107-111@v0.32.0), and
    the target captures its fp32 residual stream there (model.py:1713-1722@0f6b2619)."""
    raw = config_of(MUSE_DSPARK)
    sp = graph_of(MUSE_DSPARK)["speculator"]
    assert raw["dflash_config"]["target_layer_ids"] == [13, 29, 45, 57, 61]
    assert sp["aux_hidden_state_layers"] == [14, 30, 46, 58, 62]
    assert sp["aux_hidden_state_bytes_per_token"] == 8192 * 4


def test_draft_aux_layers_follow_the_engine_resolution_order(tmp_path):
    """Each source vLLM consults, in its order, with its own offset; an id past the
    target's last layer, or a count the drafter's fc cannot take, is refused."""
    D = deriver()
    raw = config_of(MUSE_DSPARK)

    def aux(mutate):
        bad = json.loads(json.dumps(raw))
        mutate(bad)
        return _derive_raw(bad, tmp_path)["speculator"]["aux_hidden_state_layers"]

    # dflash_config wins over the top-level ids, and adds one.
    assert aux(lambda c: c["dflash_config"].update(target_layer_ids=[0, 1, 2, 3, 4])) == [
        1, 2, 3, 4, 5]
    # Without it, the dense-DSpark top-level ids, plus one.
    assert aux(lambda c: c.pop("dflash_config")) == [14, 30, 46, 58, 62]
    # An explicit eagle list is taken as-is, ahead of everything else.
    assert aux(lambda c: c.update(eagle_aux_hidden_state_layer_ids=[2, 4, 6, 8, 10])) == [
        2, 4, 6, 8, 10]
    for mutate, why in [
        # 61 + 1 is the last layer's output; 62 + 1 is past it.
        (lambda c: c["dflash_config"].update(target_layer_ids=[13, 29, 45, 57, 62]),
         "past the target's 62 layers"),
        (lambda c: c["dflash_config"].update(target_layer_ids=[13, 29, 45, 57]),
         "captures 4 states"),
        (lambda c: c["dflash_config"].update(target_layer_ids=[13, 13, 45, 57, 61]),
         "strictly increasing"),
    ]:
        bad = json.loads(json.dumps(raw))
        mutate(bad)
        with pytest.raises(D.DeriveError, match=why):
            _derive_raw(bad, tmp_path)


def test_v41_dspark_reads_its_target_layer_ids_as_is():
    """For model_type deepseek_v41 vLLM uses dspark_target_layer_ids without the +1
    (eagle3_utils.py:115-120@v0.32.0), and each captured state is the bf16 mean of the
    hc_mult streams, one hidden-width activation (tilelang.py:313-323, mhc.py:260-264), so
    the default width stands."""
    raw = config_of(V41)
    sp = v41_graph()["speculator"]
    assert raw["model_type"] == "deepseek_v41"
    assert sp["aux_hidden_state_layers"] == text_config(raw)["dspark_target_layer_ids"]
    assert sp["aux_hidden_state_layers"] == [37, 38, 39]
    assert "aux_hidden_state_bytes_per_token" not in sp


def test_an_mtp_speculator_reads_no_aux_layers():
    assert "aux_hidden_state_layers" not in v4_graph()["speculator"]


# What each architecture's stage boundary carries where it is not vLLM's standard
# hidden_states + residual pair (the schema's default, 2 x hidden x 2 bytes).
BOUNDARY_BYTES = {
    # One fp32 stream (_make_empty_intermediate_tensors, model.py:1639-1650@0f6b2619).
    "muse-spark-open-816b-a42b": 8192 * 4,
    "muse-spark-open-816b-a42b-nvfp4": 8192 * 4,
    # The (hc_mult, hidden) bundle at the activation dtype (deepseek_v4/nvidia/
    # model.py:1406-1424@v0.32.0).
    "deepseek-v4-pro": 4 * 7168 * 2,
    # That bundle plus the (hc_mult,) fp32 pre-mix (deepseek_v41/nvidia/
    # model.py:885-910@v0.32.0).
    "deepseek-v4.1-flash": 4 * 5120 * 2 + 4 * 4,
    # hidden_states alone (inkling/nvidia/model.py:294-296, 354-359@v0.32.0).
    "inkling": 6144 * 2,
}


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_pipeline_boundary_bytes_follow_the_intermediate_tensors(d: Path):
    """A graph states its boundary only where vLLM sends something other than the
    standard pair; every other committed architecture sends the pair and omits it.
    Kimi-K3's attention-residual blocks make its residual stage-dependent
    (kimi_k3/nvidia/model.py:1201-1222@v0.32.0), which one figure cannot state."""
    g = graph_of(d)
    assert g["global"].get("pipeline_boundary_bytes_per_token") == BOUNDARY_BYTES.get(d.name)


def test_muse_spark_post_expert_finalize_writes_the_activation_dtype():
    """The finalize reads each route's bf16 GEMM2 row and accumulates in fp32, but writes
    its output buffer's dtype (fused_finalize.py:142-143@0f6b2619), which is allocated in
    the GEMM2 rows' dtype (modular_kernel.py:543-548), the activation width:
    8 x 4096 x 2 read + 4096 x 2 written."""
    for d in (MUSE, MUSE_NVFP4):
        for k in graph_of(d)["layer_kinds"]:
            assert _node(k, "post_expert_norm")["bytes_per_token"] == 8 * 4096 * 2 + 4096 * 2
    assert 8 * 4096 * 2 + 4096 * 2 == 73728


def test_muse_spark_dspark_ffn_width_is_the_multiplier_times_hidden(tmp_path):
    """The draft layer's MLP is intermediate_size_mlp, which vLLM's config computes as
    int(ffn_dim_multiplier * hidden_size) and does not read from the file
    (configs/muse_spark.py:247@0f6b2619; model.py:1360-1366), not intermediate_size."""
    raw = config_of(MUSE_DSPARK)
    bad = json.loads(json.dumps(raw))
    bad.update(ffn_dim_multiplier=1.5, intermediate_size=12345, intermediate_size_mlp=999)
    kinds = {k["id"]: k for k in _derive_raw(bad, tmp_path, MUSE_DSPARK.name)["layer_kinds"]}
    layer = kinds["dspark_layer"]
    assert (_node(layer, "mlp_gate_up")["n"], _node(layer, "mlp_down")["k"]) == (
        2 * 12288, 12288)
    committed = {k["id"]: k for k in graph_of(MUSE_DSPARK)["layer_kinds"]}["dspark_layer"]
    assert _node(committed, "mlp_down")["k"] == int(raw["ffn_dim_multiplier"] * 8192) == 16384
    bad.pop("ffn_dim_multiplier")
    # vLLM's default multiplier is 1.0 (configs/muse_spark.py:116).
    kinds = {k["id"]: k for k in _derive_raw(bad, tmp_path, MUSE_DSPARK.name)["layer_kinds"]}
    assert _node(kinds["dspark_layer"], "mlp_down")["k"] == 8192


def test_muse_spark_attention_cannot_replicate_a_kv_head(tmp_path):
    """Muse Spark projects Q, K, V and the output gate through one fused linear that slices by
    head group and asserts total_num_kv_heads % tp_size == 0 (MuseSpark1xQKVOGateParallelLinear,
    vllm/models/muse_spark/common/layers.py:368-371@0f6b2619): the target always uses it
    (use_o_gate True off the drafter, nvidia/model.py:977), the drafter where its configuration
    turns the gate on. Every attention node says so, and a drafter without the gate is refused."""
    for m in (MUSE, MUSE_NVFP4, MUSE_DSPARK):
        for k in graph_of(m)["layer_kinds"]:
            for n in k["nodes"]:
                if n["op"] == "Attention":
                    assert n.get("kv_heads_not_replicated") is True, (m.name, k["id"])
    raw = config_of(MUSE_DSPARK)
    bad = json.loads(json.dumps(raw))
    bad["use_attn_o_gate"] = False
    with pytest.raises(deriver().DeriveError, match="use_attn_o_gate"):
        _derive_raw(bad, tmp_path, MUSE_DSPARK.name)


def test_muse_spark_pooler_and_unpooler_shard_at_decode_on_blackwell():
    """On SM100 and later with TP and no DP, a step of at most 256 tokens computes the MoE
    pooler's and unpooler's output columns a TP slice at a time and all-gathers them
    (shard_pooler / shard_unpooler, _SHARDED_PROJECTION_MAX_TOKENS,
    vllm/models/muse_spark/nvidia/model.py:151, 505-523, 635-660@0f6b2619); the unpooler only
    from 32 tokens when the TP group spans nodes (:154, 535-539). Both stay replicated GEMMs."""
    for m in (MUSE, MUSE_NVFP4):
        for k in graph_of(m)["layer_kinds"]:
            pool, unpool = _node(k, "routed_expert_down_proj"), _node(k, "routed_expert_up_proj")
            assert pool["tp"] == unpool["tp"] == "replicated"
            assert pool["sharded_at_decode"] == {"capability_family": 10, "max_tokens": 256}
            assert unpool["sharded_at_decode"] == {"capability_family": 10, "max_tokens": 256,
                                                   "min_tokens_across_nodes": 32}
