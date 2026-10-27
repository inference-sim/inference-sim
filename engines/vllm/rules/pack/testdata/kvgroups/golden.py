# Golden KV cache groupings from vLLM's own code, for rules/kvcache_test.go.
#
# It extracts _approximate_gcd, _get_packed_kv_cache_groups, _get_kv_cache_bytes_per_block
# and _get_per_layer_spec from a release's vllm/v1/core/kv_cache_utils.py, runs them over
# stub specs built from a *.layers.json file the way KVCacheGroups maps caches to specs,
# and prints the groups and bytes per block:
#
#   git -C vllm show v0.31.0:vllm/v1/core/kv_cache_utils.py > /tmp/kvu31.py
#   python -I golden.py /tmp/kvu31.py synth-state.layers.json 64 0 \
#     > synth-state.v0_31.block64.align0.json
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
want = {"_approximate_gcd", "_get_packed_kv_cache_groups", "_get_kv_cache_bytes_per_block", "_get_per_layer_spec"}
code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want)

def cdiv(a, b): return -(a // -b)
def round_up(x, m): return cdiv(x, m) * m

class KVCacheSpec:
    block_stride_alignment = None
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

ns = dict(math=math, defaultdict=defaultdict, cdiv=cdiv, round_up=round_up, KVCacheSpec=KVCacheSpec,
          MambaSpec=MambaSpec, CircularBufferSpec=CircularBufferSpec, SlidingWindowSpec=SlidingWindowSpec,
          HiSparseHotSpec=HiSparseHotSpec, UniformTypeKVCacheSpecs=UniformTypeKVCacheSpecs,
          KVCacheGroupSpec=KVCacheGroupSpec, VllmConfig=object, KVCacheGroupSpecT=None,
          _annotate_eagle_groups=lambda *a, **k: None, _warn_if_unannotated_eagle_mamba=lambda *a, **k: None,
          _uses_trailing_mtp_layers=lambda c: False, _is_deepseek_v4_eagle=lambda c: False,
          _glm5_next_tensor_layout=lambda g: None, Sequence=list, list=list, dict=dict, int=int,
          KVCacheSpecT=None)
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
    specs[l["Name"]] = cls(cls, key, b, page)
groups = ns["_get_packed_kv_cache_groups"](Cfg(), specs)
if groups is None:
    print(json.dumps(None)); sys.exit()
if align > 1:
    for g in groups[:1]:
        for n in g.layer_names[:1]:
            specs[n].block_stride_alignment = align
bpb = ns["_get_kv_cache_bytes_per_block"](groups)
print(json.dumps({"bytes": bpb, "groups": [[g.kv_cache_spec.block_size, g.kv_cache_spec.page_size_bytes, g.layer_names] for g in groups]}))
