# Original to blis-vllm.
"""Write recorded_block_sizes.json: the KV block size each recorded InferenceX run used.

The runs are InferenceX benchmark rows (https://inferencex.semianalysis.com, public API). Each row
has a vLLM server log, https://inferencex.semianalysis.com/api/v1/server-log?id=<row>. This script
reads the log of each row, records where the run got its block size, and keeps no log text.

Usage:
  python3 -I fetch.py --models gptoss120b=gpt-oss-120b,minimaxm2.5=MiniMax-M2.5 \\
        --rows rows.json --out recorded_block_sizes.json
      walks /api/v1/benchmarks?model=<display name> for each slug=display pair, writes the derived
      input list rows.json, then fetches the logs and writes the output;
  python3 -I fetch.py --from-rows rows.json --out recorded_block_sizes.json
      fetches the logs of the rows already listed in rows.json (the set stays as committed);
  python3 -I fetch.py --check recorded_block_sizes.json
      fetches each row's log again and exits 1, naming the row, when a log hash differs.

What a log says about the block size (vllm/platforms/interface.py:757-781@5df63a72 sets it; the
line exists from vLLM v0.18.0, and is logged only when the size picked is not the default 16):

  engine_line  "Setting kv cache block size to N for <backends>": vLLM picked N itself;
  args         the log's non-default arguments state block_size N: the operator's flag, which says
               nothing about what vLLM picks when none is passed;
  silent       vLLM v0.18.0 or later, no block_size among the arguments and no engine line: vLLM
               kept the default, 16. This is an inference from the code, not a line in the log.

A row whose log has none of these (no server log, an older or unidentified vLLM) is listed under
"unobtainable" with its reason. The output holds the row's settings, the vLLM version and KV cache
dtype the log's engine configuration states, the size and its kind, the sha256 and byte count of
the decoded UTF-8 `serverLog` string the API returns (not of the gzip stream or the JSON wrapper),
and the date they were fetched; --from-rows keeps the date of a row whose hash is
unchanged, so a re-run reproduces the file byte for byte.

Only NVIDIA hardware is walked (the block-size table covers compute capabilities 9.0 to 10.3), and
only vLLM rows that are not disaggregated or multi-node. One row for each distinct vLLM image of
each (model, GPU, precision, tensor-parallel width, speculative method), the latest first, up to
PER_KEY.

A nightly job that runs --check is blis-vllm issue 91, not built (it is deferred beyond M4).
"""
import argparse
import concurrent.futures as cf
import datetime
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://inferencex.semianalysis.com/api/v1"
NVIDIA = {"h100", "h200", "b200", "b300", "gb200", "gb300"}
PER_KEY = 4
NOTE = ("Derived values only: each InferenceX row's block size and where the run got it, read from the "
        "row's vLLM server log, whose sha256 is recorded. No log text is kept. fetch.py writes this file.")
ENGINE_LINE = re.compile(r"Setting kv cache block size to (\d+) for")
ARGS = re.compile(r"'block_size': (\d+)")
VERSION = re.compile(r"Initializing a V1 LLM engine \(v([0-9][^)\s]*)\)")
KV_DTYPE = re.compile(r"Initializing a V1 LLM engine.{0,6000}?\bkv_cache_dtype=([A-Za-z0-9_]+)", re.S)
TP = re.compile(r"Initializing a V1 LLM engine.{0,6000}?\btensor_parallel_size=(\d+)", re.S)


def get(url):
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # network errors and HTTP errors alike are retried
            last = e
            time.sleep(2 * (attempt + 1))
    raise SystemExit("fetch.py: %s: %s" % (url, last))


def server_log(row):
    """(text, sha256 hex) of the `serverLog` string the API returns, or (None, None). The digest is
    of that string encoded as UTF-8 (the exact bytes classify reads), not of the gzip stream or the
    JSON wrapper."""
    try:
        body = get("%s/server-log?id=%s" % (BASE, row))
    except SystemExit:
        return None, None
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    text = json.loads(body).get("serverLog") or ""
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_bytes(text):
    return len(text.encode("utf-8"))


def minor(version):
    m = re.match(r"(\d+)\.(\d+)\.", version or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def classify(text):
    """(evidence, block_size, version, cache_dtype, tp) or (None, reason, ...)."""
    v = VERSION.search(text)
    version = v.group(1) if v else None
    kv = KV_DTYPE.search(text)
    tp = TP.search(text)
    kv, tp = (kv.group(1) if kv else None), (int(tp.group(1)) if tp else None)
    lines, args = set(ENGINE_LINE.findall(text)), set(ARGS.findall(text))
    if len(lines) > 1 or len(args) > 1 or (args and lines and lines != args):
        return None, "the log names several block sizes (engine lines %s, args %s)" % (sorted(lines), sorted(args)), version, kv, tp
    if args:
        return "args", int(args.pop()), version, kv, tp
    if lines:
        return "engine_line", int(lines.pop()), version, kv, tp
    if not text:
        return None, "the API serves no server log for the row", version, kv, tp
    mv = minor(version)
    if mv is None:
        return None, "the log states no vLLM version in an engine configuration line", version, kv, tp
    if mv < (0, 18):
        return None, "vLLM %s logs no block-size line (it exists from v0.18.0)" % version, version, kv, tp
    if kv is None:
        return None, "the log states no kv_cache_dtype", version, kv, tp
    return "silent", 16, version, kv, tp


def walk(models):
    rows = []
    for slug, display in models.items():
        body = get("%s/benchmarks?model=%s" % (BASE, urllib.parse.quote(display)))
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        bykey = {}
        for r in json.loads(body):
            if (r["model"] != slug or r["framework"] != "vllm" or r.get("disagg") or r.get("is_multinode")
                    or r["hardware"] not in NVIDIA):
                continue
            key = (r["hardware"], r["precision"], r["decode_tp"], r["spec_method"])
            bykey.setdefault(key, []).append(r)
        for key, rs in sorted(bykey.items()):
            seen = set()
            for r in sorted(rs, key=lambda r: -int(r["id"])):
                if r["image"] in seen or len(seen) >= PER_KEY:
                    continue
                seen.add(r["image"])
                rows.append({"row": int(r["id"]), "model": slug, "hardware": key[0], "precision": key[1],
                             "tp": key[2], "speculative_method": key[3], "benchmark_type": r["benchmark_type"],
                             "image": r["image"]})
    return sorted(rows, key=lambda r: r["row"])


def one(r):
    text, digest = server_log(r["row"])
    if text is None:
        return r, None, "the API serves no server log for the row"
    evidence, size, version, kv, tp = classify(text)
    if evidence is None:
        return r, None, size
    return r, dict(r, evidence=evidence, block_size=size, engine_version=version, cache_dtype=kv, log_tp=tp,
                   log_sha256=digest, log_bytes=log_bytes(text)), None


def build(rows, previous, models):
    today = datetime.date.today().isoformat()
    runs, unobtainable, changed = [], [], []
    with cf.ThreadPoolExecutor(8) as ex:
        for r, rec, why in ex.map(one, rows):
            if rec is None:
                unobtainable.append({"row": r["row"], "model": r["model"], "reason": why})
                continue
            old = previous.get(r["row"])
            rec["fetched"] = old["fetched"] if old and old["log_sha256"] == rec["log_sha256"] else today
            if old and old["log_sha256"] != rec["log_sha256"]:
                changed.append(r["row"])
            runs.append(rec)
    for row in changed:
        print("fetch.py: the log of row %d differs from the recorded hash" % row, file=sys.stderr)
    return {"source": "InferenceX server logs, %s/server-log?id=<row>" % BASE, "note": NOTE, "models": models,
            "unobtainable": sorted(unobtainable, key=lambda u: u["row"]),
            "runs": sorted(runs, key=lambda r: r["row"])}


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def check(path):
    recorded = json.load(open(path))
    bad = []
    with cf.ThreadPoolExecutor(8) as ex:
        for r, digest in ex.map(lambda r: (r, server_log(r["row"])[1]), recorded["runs"]):
            if digest != r["log_sha256"]:
                bad.append(r["row"])
                print("row %d: log sha256 is %s, recorded %s" % (r["row"], digest, r["log_sha256"]))
    print("%d rows checked, %d differ" % (len(recorded["runs"]), len(bad)))
    return 1 if bad else 0


# Known answers from the API, stable across repeat fetches: row -> (evidence, bytes, sha256).
KNOWN = {
    411031: ("args", 126004, "388dfb966c896b83cf47956614a78007fa0941675892173b6a77e330c65e900e"),
    420106: ("engine_line", 100832, "7718bd95e60c95bc4279a1cc6760b32e0ab7413060b7a706cd184ef1f09a88cd"),
}


def self_test():
    bad = 0
    for row, want in sorted(KNOWN.items()):
        text, digest = server_log(row)
        got = (classify(text)[0], log_bytes(text), digest) if text is not None else None
        print("row %d: %s" % (row, "ok" if got == want else "DIFFERS: got %s, want %s" % (got, want)))
        bad += got != want
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="slug=display name,... (display names are the API's /benchmarks?model= values)")
    ap.add_argument("--rows", help="rows.json to write (with --models)")
    ap.add_argument("--from-rows", help="rows.json to fetch")
    ap.add_argument("--out", help="recorded_block_sizes.json to write")
    ap.add_argument("--self-test", action="store_true", help="fetch the rows with known digests and compare (needs the network)")
    ap.add_argument("--check", help="recorded_block_sizes.json to verify against the logs now served")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.check:
        return check(a.check)
    if a.models:
        models = dict(p.split("=", 1) for p in a.models.split(","))
        rows = walk(models)
        if a.rows:
            dump({"models": models, "rows": rows}, a.rows)
    elif a.from_rows:
        spec = json.load(open(a.from_rows))
        models, rows = spec["models"], spec["rows"]
    else:
        ap.error("give --models, --from-rows or --check")
    previous = {}
    if a.out:
        try:
            previous = {r["row"]: r for r in json.load(open(a.out))["runs"]}
        except (OSError, ValueError, KeyError):
            pass
    out = build(rows, previous, models)
    if a.out:
        dump(out, a.out)
    print("%d rows: %d with a block size, %d unobtainable" % (len(rows), len(out["runs"]), len(out["unobtainable"])), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
