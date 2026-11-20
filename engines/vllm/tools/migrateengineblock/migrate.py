# Original to blis-vllm.
"""Rewrite engine blocks written in the earlier key names into vLLM's own (blis-vllm#93).

The engine block takes vLLM's EngineArgs names, so a name in fields.json is a name in the block.
This renames the keys that differed, in the YAML a repository holds, and changes nothing else: it
edits only the lines of an `engine:` mapping, keeps comments and layout, and writes a file only
when a key changed. It is idempotent.

  cudagraph_mode, decoder_replay_cudagraph_capture_sizes, custom_ops  -> compilation_config.<key>
  use_trtllm_attention, mla_prefill_backend, indexer_kv_dtype          -> attention_config.<key>
  cache_dtype                                                          -> kv_cache_dtype
  dbo: {enable_dbo, dbo_*_token_threshold}                             -> enable_dbo, dbo_*_token_threshold
  eplb: {enable_eplb, num_redundant_experts, ...}                      -> enable_eplb, eplb_config.<key>
  kv_events: {...}                                                     -> kv_events_config: {...}
  speculative: {method, num_spec_tokens, draft_model, ...}             -> speculative_config: {method,
                                                                          num_speculative_tokens, model, ...}
  frontend: rust                                                       -> env: {VLLM_USE_RUST_FRONTEND: "1"}
  allreduce_backend                                                    -> left alone (the block keeps it)

Usage: python3 -I tools/migrateengineblock/migrate.py [--check] <file-or-dir>...
--check writes nothing and exits 1 if a file would change.
"""
import os
import re
import sys

SIMPLE = {
    "cudagraph_mode": ("compilation_config", "cudagraph_mode"),
    "decoder_replay_cudagraph_capture_sizes": ("compilation_config", "decoder_replay_cudagraph_capture_sizes"),
    "custom_ops": ("compilation_config", "custom_ops"),
    "use_trtllm_attention": ("attention_config", "use_trtllm_attention"),
    "mla_prefill_backend": ("attention_config", "mla_prefill_backend"),
    "indexer_kv_dtype": ("attention_config", "indexer_kv_dtype"),
}
RENAME = {"cache_dtype": "kv_cache_dtype"}
# group key -> (new group key or None for flattening, {old leaf: new leaf or (None, flat name)})
GROUPS = {
    "dbo": (None, {"enable_dbo": "enable_dbo", "dbo_decode_token_threshold": "dbo_decode_token_threshold",
                   "dbo_prefill_token_threshold": "dbo_prefill_token_threshold"}),
    "eplb": ("eplb_config", {"enable_eplb": "FLAT:enable_eplb", "num_redundant_experts": "num_redundant_experts",
                             "window_size": "window_size", "step_interval": "step_interval"}),
    "kv_events": ("kv_events_config", {"enable_kv_cache_events": "enable_kv_cache_events",
                                      "publisher": "publisher", "topic": "topic"}),
    "speculative": ("speculative_config", {"method": "method", "num_spec_tokens": "num_speculative_tokens",
                                           "draft_model": "model", "kv_cache_dtype": "kv_cache_dtype",
                                           "draft_sample_method": "draft_sample_method",
                                           "rejection_sample_method": "rejection_sample_method",
                                           "synthetic_acceptance_length": "synthetic_acceptance_length",
                                           "enable_adaptive_verification": "enable_adaptive_verification"}),
}
KEY = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*):(.*)$")


def inline_items(text):
    """Split the inside of a flow mapping at top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def convert_group(name, items):
    """items: list of 'leaf: value' strings. Returns (flat items, grouped items, new group name)."""
    new_group, leaves = GROUPS[name]
    flat, grouped = [], []
    for it in items:
        leaf, _, val = it.partition(":")
        leaf, val = leaf.strip(), val.strip()
        if leaf not in leaves:
            raise ValueError(f"{name}.{leaf}: not a key of the earlier {name} group")
        to = leaves[leaf]
        if new_group is None or to.startswith("FLAT:"):
            flat.append(f"{to.removeprefix('FLAT:')}: {val}")
        else:
            grouped.append(f"{to}: {val}")
    return flat, grouped, new_group


def migrate_lines(lines):
    """Rewrite the lines of one file; returns the new lines."""
    out, i = [], 0
    while i < len(lines):
        m = re.match(r"^(\s*(?:-\s+)?)engine:\s*(#.*)?$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        base = len(m.group(1)) - (len(m.group(1)) - len(m.group(1).replace("-", " ")) and 0)
        base = len(m.group(1).replace("-", " "))
        out.append(lines[i])
        i += 1
        block = []
        while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")
                                  or len(lines[i]) - len(lines[i].lstrip()) > base):
            block.append(lines[i])
            i += 1
        # trailing blank/comment lines belong to what follows
        tail = []
        while block and (not block[-1].strip() or len(block[-1]) - len(block[-1].lstrip()) <= base):
            tail.insert(0, block.pop())
        out.extend(migrate_block(block))
        out.extend(tail)
    return out


def frontend_value(rest):
    """The value of a `frontend:` line: `rust` or `python`, after a trailing comment and quotes."""
    v = rest.strip()
    if v[:1] in ("'", '"'):
        end = v.find(v[0], 1)
        if end < 0:
            raise ValueError(f"frontend: unterminated quote in {rest.strip()!r}")
        tail = v[end + 1:].strip()
        v = v[1:end]
    else:
        v, _, tail = v.partition(" #")
        v = v.strip()
        tail = ("#" + tail) if _ else ""
    if tail and not tail.startswith("#"):
        raise ValueError(f"frontend: cannot read the value in {rest.strip()!r}; write rust or python")
    if v not in ("rust", "python"):
        raise ValueError(f"frontend: {rest.strip()!r} is neither rust nor python; "
                         "rust is env: {VLLM_USE_RUST_FRONTEND: \"1\"}, python is no setting")
    return v


def migrate_block(block):
    if not block:
        return block
    indent = None
    for ln in block:
        if ln.strip() and not ln.lstrip().startswith("#"):
            indent = len(ln) - len(ln.lstrip())
            break
    if indent is None:
        return block
    pad = " " * indent
    out, extra_groups, i = [], {}, 0
    frontend_env = None
    flat_after = []
    while i < len(block):
        ln = block[i]
        m = KEY.match(ln)
        if not m or len(m.group(1)) != indent:
            out.append(ln)
            i += 1
            continue
        key, rest = m.group(2), m.group(3)
        if key == "frontend":
            frontend_env = "1" if frontend_value(rest) == "rust" else "0"
            i += 1
            continue
        if key in SIMPLE:
            g, leaf = SIMPLE[key]
            if not rest.strip():
                raise ValueError(f"{key} is written as a block; write it on one line, [a, b], and run again")
            extra_groups.setdefault(g, []).append(f"{leaf}:{rest}")
            i += 1
            continue
        if key in RENAME:
            out.append(f"{pad}{RENAME[key]}:{rest}\n" if ln.endswith("\n") else f"{pad}{RENAME[key]}:{rest}")
            i += 1
            continue
        if key in GROUPS:
            items, j = [], i + 1
            val = rest.strip()
            if val.startswith("{"):
                inner = val[1:val.rindex("}")]
                items = inline_items(inner)
            else:
                while j < len(block) and (not block[j].strip() or len(block[j]) - len(block[j].lstrip()) > indent):
                    if block[j].strip() and not block[j].lstrip().startswith("#"):
                        items.append(block[j].strip())
                    j += 1
            flat, grouped, new_group = convert_group(key, items)
            for f in flat:
                out.append(f"{pad}{f}\n")
            if grouped:
                out.append(f"{pad}{new_group}: {{{', '.join(grouped)}}}\n")
            i = j
            continue
        out.append(ln)
        i += 1
    for g, items in extra_groups.items():
        out.append(f"{pad}{g}: {{{', '.join(x.strip() for x in items)}}}\n")
    if frontend_env == "1":
        for k, ln in enumerate(out):
            m = KEY.match(ln)
            if m and len(m.group(1)) == indent and m.group(2) == "env" and m.group(3).strip().startswith("{"):
                body = m.group(3).strip()
                inner = body[1:body.rindex("}")].strip()
                out[k] = f'{pad}env: {{{inner + ", " if inner else ""}VLLM_USE_RUST_FRONTEND: "1"}}\n'
                break
            if m and len(m.group(1)) == indent and m.group(2) == "env":
                raise ValueError("a block-style env beside frontend: rewrite by hand")
        else:
            out.append(f'{pad}env: {{VLLM_USE_RUST_FRONTEND: "1"}}\n')
    return out


def migrate_text(text):
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
        added = True
    else:
        added = False
    new = "".join(migrate_lines(lines))
    return new[:-1] if added and new.endswith("\n") else new


def files(paths):
    for p in paths:
        if os.path.isdir(p):
            for d, _, fs in os.walk(p):
                for f in sorted(fs):
                    if f.endswith((".yaml", ".yml")):
                        yield os.path.join(d, f)
        else:
            yield p


def main(argv):
    check = "--check" in argv
    paths = [a for a in argv[1:] if a != "--check"]
    changed = 0
    for f in files(paths):
        with open(f, encoding="utf-8") as fh:
            old = fh.read()
        try:
            new = migrate_text(old)
        except ValueError as e:
            sys.exit(f"{f}: {e}")
        if new != old:
            changed += 1
            if not check:
                with open(f, "w", encoding="utf-8") as fh:
                    fh.write(new)
    print(f"{changed} file(s) {'would change' if check else 'rewritten'}")
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
