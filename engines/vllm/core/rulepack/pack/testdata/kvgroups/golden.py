# Golden KV cache groupings from vLLM's own code, for core/rulepack/pack/kvcache_test.go.
#
# It extracts _approximate_gcd, _get_packed_kv_cache_groups, _get_kv_cache_bytes_per_block
# and _get_per_layer_spec from a release's vllm/v1/core/kv_cache_utils.py, runs them over
# stub specs built from a *.layers.json file the way KVCacheGroups maps caches to specs,
# and prints the groups and bytes per block:
#
#   git -C vllm show v0.33.0:vllm/v1/core/kv_cache_planning.py > /tmp/kvp33.py
#   python -I golden.py /tmp/kvp33.py synth-state.layers.json 64 0 \
#     > synth-state.v0_33.block64.align0.json
#
# (v0.33.0 moved these functions from kv_cache_utils.py to kv_cache_planning.py.)
#
# At vLLM main head 0fda86a35419 the layer data model changed (KVCacheLayerConfig, KVCacheGroup,
# ManagerConfig; kv_cache_planning.py _bucket_layers / can_share_block_table). A source that
# defines _bucket_layers runs through the head stub below, which builds the same layer set with
# the manager each cache kind uses at the head (FullAttentionManager for kv and index,
# SlidingWindowManager for window, CircularBufferManager for state), so that a bucket is the
# layers sharing a manager policy and block size:
#
#   git -C vllm show 0fda86a35419:vllm/v1/core/kv_cache_planning.py > /tmp/kvp_head.py
#   python -I golden.py /tmp/kvp_head.py synth-state.layers.json 64 0 \
#     > synth-state.v0fda86a35419.block64.align0.json
#
# The next two arguments are the engine block size and a block_stride_alignment.
#
# This stub reproduces KVCacheGroups' own cache-to-spec mapping (kind, block size, page
# bytes), so a golden proves the packing and the block size given that mapping, not the
# mapping itself; MatchSource tests pin the mapping's engine facts separately.
import ast, json, math, sys
from collections import defaultdict, Counter

src_path, layers_path, block, align = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
# An optional fifth argument is the speculative K, which sizes the compressor ring as
# vllm/models/deepseek_v41/compressor.py:147-149 does at v0.31.0.
spec_k = int(sys.argv[5]) if len(sys.argv) > 5 else None
src = open(src_path).read()
tree = ast.parse(src)
HEAD = any(isinstance(n, ast.FunctionDef) and n.name == "_bucket_layers" for n in tree.body)
want = {"_approximate_gcd", "_get_packed_kv_cache_groups", "_get_kv_cache_bytes_per_block", "_get_per_layer_spec"}
if HEAD:
    want = {"_approximate_gcd", "_get_packed_kv_cache_groups", "_get_kv_cache_bytes_per_block", "_get_layer_block_stride", "_bucket_layers", "can_share_block_table"}
code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want)

def cdiv(a, b): return -(a // -b)
def round_up(x, m): return cdiv(x, m) * m

class KVCacheSpec:
    block_stride_alignment = None
    # v0.33's _get_kv_cache_bytes_per_block reads the alignment through this method
    # (vllm/v1/kv_cache_interface.py:224-234); this stub states plain byte alignments only.
    def get_block_stride_alignment(self): return self.block_stride_alignment or 1
class MambaSpec(KVCacheSpec): pass
class HiSparseHotSpec(KVCacheSpec): pass
class Spec(KVCacheSpec):
    def __init__(self, cls, key, block, page):
        self.cls, self.key, self.block_size, self.page_size_bytes = cls, key, block, page
class FullAttentionSpec(Spec): pass
class SlidingWindowSpec(Spec): pass
class CircularBufferSpec(Spec): pass

class UniformTypeKVCacheSpecs(KVCacheSpec):
    def __init__(self, specs): self.kv_cache_specs = specs; self.block_size = next(iter(specs.values())).block_size
    @property
    def first_spec(self): return next(iter(self.kv_cache_specs.values()))
    @property
    def page_size_bytes(self): return sum(s.page_size_bytes for s in self.kv_cache_specs.values())
    @classmethod
    def is_uniform_type(cls, specs): return len({s.key for s in specs.values()}) == 1
    @classmethod
    def from_specs(cls, specs): return cls(specs) if cls.is_uniform_type(specs) else None
    def get_max_layers_per_page_size(self):
        return Counter(s.page_size_bytes for s in self.kv_cache_specs.values()).most_common(1)[0][1]

class KVCacheGroupSpec:
    def __init__(self, names, spec): self.layer_names, self.kv_cache_spec = names, spec

class Layout: is_block_outermost = True
class CC:
    def get_resolved_kv_cache_layout(self): return Layout()
class Cfg: cache_config = CC()

if HEAD:
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class ManagerConfig:
        manager_cls: type
        params: tuple
    class _Manager:
        @classmethod
        def can_share_block_table(cls, specs): return True
    class FullAttentionManager(_Manager): pass
    class SlidingWindowManager(_Manager): pass
    class CircularBufferManager(_Manager): pass
    class MambaManager(_Manager): pass
    class HSpec:
        dcp_sharded = True
        block_stride_alignment = None
        def __init__(self, block, page): self.block_size, self.page_size_bytes = block, page
        def get_block_stride_alignment(self): return self.block_stride_alignment or 1
    class KVCacheLayerConfig:
        is_draft = False
        def __init__(self, manager_config, spec): self.manager_config, self.spec = manager_config, spec
        @property
        def manager_cls(self): return self.manager_config.manager_cls
    class KVCacheGroup:
        def __init__(self, specs, manager_config, block_size, dcp_sharded=False):
            self.kv_cache_specs, self.manager_config, self.block_size, self.dcp_sharded = specs, manager_config, block_size, dcp_sharded
        @classmethod
        def from_layers(cls, layer_configs):
            first = next(iter(layer_configs.values()))
            assert all(lc.manager_config == first.manager_config and lc.spec.block_size == first.spec.block_size
                       and lc.spec.dcp_sharded == first.spec.dcp_sharded for lc in layer_configs.values())
            return cls({n: lc.spec for n, lc in layer_configs.items()}, first.manager_config,
                       first.spec.block_size, first.spec.dcp_sharded)
        @property
        def layer_names(self): return list(self.kv_cache_specs)
        @property
        def manager_cls(self): return self.manager_config.manager_cls
        @property
        def page_size_bytes(self): return sum(s.page_size_bytes for s in self.kv_cache_specs.values())

ns = dict(math=math, defaultdict=defaultdict, cdiv=cdiv, round_up=round_up, KVCacheSpec=KVCacheSpec,
          MambaSpec=MambaSpec, CircularBufferSpec=CircularBufferSpec, SlidingWindowSpec=SlidingWindowSpec,
          HiSparseHotSpec=HiSparseHotSpec, UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
          KVCacheGroupSpec=KVCacheGroupSpec, VllmConfig=object, KVCacheGroupSpecT=None,
          _annotate_eagle_groups=lambda *a, **k: None, _warn_if_unannotated_eagle_mamba=lambda *a, **k: None,
          _uses_trailing_mtp_layers=lambda c: False, _is_deepseek_v4_eagle=lambda c: False,
          _glm5_next_tensor_layout=lambda g: None, Sequence=list, list=list, dict=dict, int=int,
          KVCacheSpecT=None)
if HEAD:
    class VC: cache_config = CC()
    ns.update(ManagerConfig=ManagerConfig, FullAttentionManager=FullAttentionManager,
              SlidingWindowManager=SlidingWindowManager, CircularBufferManager=CircularBufferManager,
              MambaManager=MambaManager, KVCacheGroup=KVCacheGroup, KVCacheLayerConfig=KVCacheLayerConfig,
              Counter=Counter, KVCacheLayout=Layout)
exec("from __future__ import annotations\n" + code, ns)

specs = {}
for l in json.load(open(layers_path)):
    kind = l["Kind"]
    if kind == "state":
        rows = l["Entries"] if spec_k is None else max(8, 1 << (spec_k + 1).bit_length())
        b, page, cls, key = rows, rows * l["Bytes"], CircularBufferSpec, ("state", rows)
    else:
        b = l["Block"] or block
        page = b // l["TPE"] * l["Bytes"]
        cls = SlidingWindowSpec if kind == "window" else FullAttentionSpec
        key = ("window", l["Window"], b) if kind == "window" else ("full", b)
    # Only the MLA specs pad (_apply_alignment_padding); a CircularBufferSpec never does.
    if l["Align"] > 1 and kind != "state": page = round_up(page, l["Align"])
    if HEAD:
        mgr = {"state": CircularBufferManager, "window": SlidingWindowManager}.get(kind, FullAttentionManager)
        params = ("window", l["Window"]) if kind == "window" else ()
        specs[l["Name"]] = KVCacheLayerConfig(ManagerConfig(mgr, params), HSpec(b, page))
        continue
    specs[l["Name"]] = cls(cls, key, b, page)
if HEAD:
    groups = ns["_get_packed_kv_cache_groups"](VC(), specs)
    if groups is None:
        print(json.dumps(None)); sys.exit()
    if align > 1:
        for g in groups[:1]:
            for n in g.layer_names[:1]:
                g.kv_cache_specs[n].block_stride_alignment = align
    bpb = ns["_get_kv_cache_bytes_per_block"](groups, Layout())
    print(json.dumps({"bytes": bpb, "groups": [[g.block_size, g.page_size_bytes, g.layer_names] for g in groups]}))
    sys.exit()
groups = ns["_get_packed_kv_cache_groups"](Cfg(), specs)
if groups is None:
    print(json.dumps(None)); sys.exit()
if align > 1:
    for g in groups[:1]:
        for n in g.layer_names[:1]:
            specs[n].block_stride_alignment = align
# v0.33 takes the resolved layout too (block-outermost, as _get_packed_kv_cache_groups saw it).
bpb = (ns["_get_kv_cache_bytes_per_block"](groups, Layout()) if "layout" in
       ns["_get_kv_cache_bytes_per_block"].__code__.co_varnames else ns["_get_kv_cache_bytes_per_block"](groups))
print(json.dumps({"bytes": bpb, "groups": [[g.kv_cache_spec.block_size, g.kv_cache_spec.page_size_bytes, g.layer_names] for g in groups]}))
