# Original to blis-vllm.
"""Write recorded_block_sizes.json: the block size each recorded InferenceX run used.

Usage: python3 -I fetch.py <rows.json> > recorded_block_sizes.json

rows.json lists one InferenceX benchmark row for each run, with the run's model, GPU,
tensor-parallel width, speculative method and KV cache dtype: [{"row": 434552, "model": ...}].
For each row the script reads the vLLM server log the public InferenceX API serves for it,
https://inferencex.semianalysis.com/api/v1/server-log?id=<row>, and records where the run got its
block size:

  engine_line  the log has "Setting kv cache block size to N for <backends>". vLLM prints that
               when it picks the size itself (vllm/platforms/interface.py:757-781@5df63a72), so
               N is vLLM's choice and can validate the resolution.
  args         the log's passed arguments state block_size N: the operator's flag, which says
               nothing about what vLLM picks when none is passed.

Only derived values are written (the row id, the run's settings, the size and its kind). No log
text is kept. A row whose log states neither (an agentic row's log is the client's, not the server's) is
listed under "unobtainable" with its id, and written nowhere else. A run whose log names more than
one size, or states both an engine line and an argument that differ, stops the script.
"""
import gzip
import json
import re
import sys
import time
import urllib.request

API = "https://inferencex.semianalysis.com/api/v1/server-log?id=%d"
ENGINE_LINE = re.compile(r"Setting kv cache block size to (\d+) for")
ARGS = re.compile(r"'block_size': (\d+)")


def read(row):
    row = int(row)
    req = urllib.request.Request(API % row, headers={"Accept-Encoding": "gzip"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return (json.loads(body).get("serverLog") or "")


def classify(row, log):
    lines = set(ENGINE_LINE.findall(log))
    args = set(ARGS.findall(log))
    if len(lines) > 1 or len(args) > 1:
        raise SystemExit("row %d: the log names several sizes (engine lines %s, args %s)" % (row, sorted(lines), sorted(args)))
    if args:
        if lines and lines != args:
            raise SystemExit("row %d: the run passed %s and the engine logged %s" % (row, sorted(args), sorted(lines)))
        return "args", int(args.pop())
    if lines:
        return "engine_line", int(lines.pop())
    return None, None


def main(path):
    out, unobtainable = [], []
    for r in json.load(open(path)):
        row = int(r["row"])
        evidence, size = classify(row, read(row))
        if evidence is None:
            unobtainable.append(row)
            continue
        out.append(dict(r, row=row, evidence=evidence, block_size=size))
    out.sort(key=lambda r: int(r["row"]))
    json.dump({"source": "InferenceX server logs, https://inferencex.semianalysis.com/api/v1/server-log?id=<row>",
               "unobtainable": sorted(unobtainable), "runs": out}, sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
