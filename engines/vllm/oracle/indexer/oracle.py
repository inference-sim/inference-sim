# Original to blis-vllm.
"""Oracle for attention_config.indexer_kv_dtype on DeepSeek V4 and V4.1: vLLM v0.32.0's own code, run unmodified.

Which dtype the DeepSeek sparse indexer's key cache runs in, and how many bytes one key record
takes, is decided in four places that sit in modules importing torch and CUDA wheels. This script
parses each module with `ast`, takes the named definitions out as written, and runs them in a
namespace that holds only what they read (as oracle/lower/extract.py does). Nothing is rewritten
or copied (FID-8):

  vllm/config/attention.py                      IndexerKVDType, AttentionConfig.resolve_indexer_kv_dtype
  vllm/v1/attention/backends/mla/indexer.py     DSA_INDEXER_KV_DTYPES, dsa_indexer_uses_fp4
  vllm/models/deepseek_v41/attention.py         _indexer_k_cache_head_dim
  vllm/models/deepseek_v4/attention.py          quant_block_size = 128 and the `if self.use_fp4_kv` statement
                                                of DeepseekV4Indexer.__init__ (the record width of V4)

The replacements (FID-1) are listed in oracle/indexer/replacements.json: the platform, which
answers is_rocm() False and is_device_capability_family(100) from the case, and the configuration
object, a namespace holding the one field the functions read. A value of IndexerKVDType that
pydantic refuses is run through pydantic's own validation of vLLM's literal, as the engine's
argument parser would.

Usage:
    python -I oracle/indexer/oracle.py <vllm source root> print            the table, as JSON
    python -I oracle/indexer/oracle.py <vllm source root> check <file>     fail unless <file> is what print gives

<vllm source root> is a directory holding vllm/ at commit 5df63a7203cdb9328d824a27b682e77c7b9928be. The Python needs
pydantic only (the oracle environment, oracle/requirements.lock, has it).
"""
import ast
import json
import os
import sys
import types
from typing import Any, Literal

COMMIT = "5df63a7203cdb9328d824a27b682e77c7b9928be"
# DeepSeek V4's and V4.1's index_head_dim (the catalog graphs state d_h 128 for the indexer).
HEAD_DIM = 128
REQUESTED = ["auto", "bf16", "fp8", "mxfp4", "nvfp4", "fp8_e4m3"]
# is_device_capability_family's argument: 90 is Hopper (SM9x), 100 is Blackwell datacenter (SM10x).
FAMILIES = [90, 100]


def parse(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return ast.parse(f.read(), rel)


def assign(tree, name):
    """The top-level assignment to name, as written."""
    out = [n for n in tree.body
           if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == name for t in n.targets))
           or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == name)]
    if len(out) != 1:
        raise SystemExit(f"assignment to {name} not found exactly once")
    return out[0]


def func(tree, name):
    out = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(out) != 1:
        raise SystemExit(f"function {name} not found exactly once")
    return out[0]


def method(tree, cls, name):
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    out = [n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(out) != 1:
        raise SystemExit(f"{cls}.{name} not found exactly once")
    return out[0]


def run(nodes, ns, rel):
    mod = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, rel, "exec"), ns)


def v4_record_statement(tree):
    """The `if self.use_fp4_kv:` statement of DeepseekV4Indexer.__init__ that sets k_cache_head_dim, and the
    `self.quant_block_size = 128` assignment it reads."""
    init = method(tree, "DeepseekV4Indexer", "__init__")
    stmts = [n for n in init.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.use_fp4_kv"
             and any(isinstance(m, ast.Assign) and ast.unparse(m.targets[0]) == "k_cache_head_dim"
                     for m in ast.walk(n))]
    quant = [n for n in init.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.quant_block_size"]
    if len(stmts) != 1 or len(quant) != 1:
        raise SystemExit("DeepseekV4Indexer.__init__: the record-width statement or quant_block_size moved")
    return quant[0], stmts[0]


def load(root):
    att = parse(root, "vllm/config/attention.py")
    idx = parse(root, "vllm/v1/attention/backends/mla/indexer.py")
    v41 = parse(root, "vllm/models/deepseek_v41/attention.py")
    v4 = parse(root, "vllm/models/deepseek_v4/attention.py")
    q = parse(root, "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py")
    ns = {"__builtins__": __builtins__, "Literal": Literal, "Any": Any}
    run([assign(att, "IndexerKVDType")], ns, "vllm/config/attention.py")
    # AttentionConfig is a pydantic dataclass whose module imports the attention backends; only the
    # method that resolves "auto" is taken, into a class of its own.
    cls = ast.ClassDef(name="AttentionConfig", bases=[], keywords=[], decorator_list=[], type_params=[],
                       body=[method(att, "AttentionConfig", "resolve_indexer_kv_dtype")])
    run([cls], ns, "vllm/config/attention.py")
    # Annotations the next definitions mention are evaluated when they run.
    ns.update(VllmConfig=object)
    run([assign(idx, "DSA_INDEXER_KV_DTYPES"), func(idx, "dsa_indexer_uses_fp4")], ns,
        "vllm/v1/attention/backends/mla/indexer.py")
    run([assign(q, "MXFP4_BLOCK_SIZE")], ns, "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py")
    run([func(v41, "_indexer_k_cache_head_dim")], ns, "vllm/models/deepseek_v41/attention.py")
    quant, stmt = v4_record_statement(v4)
    record = ast.Module(body=[quant, stmt], type_ignores=[])
    ast.fix_missing_locations(record)
    return ns, record


def v4_width(ns, record, use_fp4):
    """DeepseekV4Indexer's k_cache_head_dim, by running its statement over a namespace holding the fields
    it reads."""
    me = types.SimpleNamespace(use_fp4_kv=use_fp4, head_dim=HEAD_DIM)
    scope = {"self": me, "MXFP4_BLOCK_SIZE": ns["MXFP4_BLOCK_SIZE"]}
    exec(compile(record, "vllm/models/deepseek_v4/attention.py", "exec"), scope)
    return scope["k_cache_head_dim"]


def case(ns, record, requested, family):
    from pydantic import TypeAdapter, ValidationError
    out = {"requested": requested, "family": family}
    try:
        TypeAdapter(ns["IndexerKVDType"]).validate_python(requested)
    except ValidationError as e:
        first = e.errors()[0]
        out["error"] = {"class": "ValidationError", "type": first["type"], "message": first["msg"]}
        return out
    # The platform: CUDA, on a device of the case's capability family.
    platform = types.SimpleNamespace(is_rocm=lambda: False, is_device_capability_family=lambda f: f == family)
    ns["current_platform"] = platform
    cfg = types.SimpleNamespace(attention_config=ns["AttentionConfig"]())
    cfg.attention_config.indexer_kv_dtype = requested
    try:
        fp4 = ns["dsa_indexer_uses_fp4"](cfg)
    except ValueError as e:
        out["error"] = {"class": "ValueError", "message": str(e)}
        return out
    out["resolved"] = "mxfp4" if fp4 else "fp8"
    out["width_v4"] = v4_width(ns, record, fp4)
    out["width_v41"] = ns["_indexer_k_cache_head_dim"](HEAD_DIM, fp4)
    return out


def build(root):
    ns, record = load(root)
    return {
        "commit": COMMIT,
        "index_head_dim": HEAD_DIM,
        "cases": [case(ns, record, r, f) for f in FAMILIES for r in REQUESTED],
    }


def main(argv):
    if len(argv) < 3 or argv[2] not in ("print", "check"):
        raise SystemExit(__doc__)
    out = json.dumps(build(argv[1]), indent=1, sort_keys=True) + "\n"
    if argv[2] == "print":
        sys.stdout.write(out)
        return
    with open(argv[3], encoding="utf-8") as f:
        if f.read() != out:
            raise SystemExit(f"{argv[3]} is not what vLLM at {COMMIT} gives; regenerate it with `print`")
    print("ok:", argv[3])


if __name__ == "__main__":
    main(sys.argv)
