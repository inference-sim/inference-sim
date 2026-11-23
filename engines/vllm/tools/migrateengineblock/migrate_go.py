# Original to blis-vllm.
"""Rewrite engine-block keys inside the Go string literals of tests (blis-vllm#93).

A Go test states an engine block as text in a string. This rewrites, inside string literals only
(never struct tags, identifiers or comments), the one-line forms the tests use:

  speculative: {method: x, num_spec_tokens: n, draft_model: m}  -> speculative_config: {method: x, num_speculative_tokens: n, model: m}
  dbo: {enable_dbo: true, ...}                                   -> enable_dbo: true, ...
  eplb: {enable_eplb: true, ...}                                 -> enable_eplb: true, eplb_config: {...}
  kv_events: {...}                                               -> kv_events_config: {...}
  cudagraph_mode: X, cache_dtype: X, custom_ops: [..], ...      -> compilation_config: {...}, kv_cache_dtype: X, ...

Usage: python3 -I tools/migrateengineblock/migrate_go.py [--check] <file.go>...
"""
import re
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import migrate  # noqa: E402

STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')


def fix_literal(lit):
    body = lit[1:-1]
    if not re.search(r"(^|\\n|[ ,{])(speculative|dbo|eplb|kv_events|cudagraph_mode|cache_dtype|custom_ops|"
                     r"use_trtllm_attention|mla_prefill_backend|indexer_kv_dtype|"
                     r"decoder_replay_cudagraph_capture_sizes|frontend):", body):
        return lit
    # Treat the literal as lines of an engine block: the Go text uses \n for newlines.
    lines = body.split("\\n")
    out = []
    migrated = migrate.migrate_block([ln + "\n" for ln in lines if ln != ""] or [""])
    # keep a trailing \n if the original had one
    text = "\\n".join(m.rstrip("\n") for m in migrated)
    if body.endswith("\\n"):
        text += "\\n"
    # a literal that is not a block of top-level keys (a sentence, a fragment) is left alone
    return '"' + text + '"'


def migrate_text(src):
    out, i = [], 0
    for m in STRING.finditer(src):
        out.append(src[i:m.start()])
        # skip struct tags: a backquoted tag is not matched; a quoted yaml:"..." inside backquotes is
        # excluded by checking the preceding char sequence
        before = src[max(0, m.start() - 6):m.start()]
        if before.endswith("yaml:") or before.endswith("json:"):
            out.append(m.group(0))
        else:
            try:
                out.append(fix_literal(m.group(0)))
            except ValueError:
                out.append(m.group(0))
        i = m.end()
    out.append(src[i:])
    return "".join(out)


def main(argv):
    check = "--check" in argv
    n = 0
    for f in [a for a in argv[1:] if a != "--check"]:
        old = open(f, encoding="utf-8").read()
        new = migrate_text(old)
        if new != old:
            n += 1
            if not check:
                open(f, "w", encoding="utf-8").write(new)
    print(f"{n} file(s) {'would change' if check else 'rewritten'}")
    return 1 if (check and n) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
