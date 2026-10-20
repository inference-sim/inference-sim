"""Build <work>/kv_units.json for cov.kv_units_differ: per server log, whether the logged "GPU KV cache size: N
tokens" agrees (within 5%) with the same log's "Maximum concurrency for L tokens per request: Xx" line
(implied = X * L). A hybrid/compressed cache whose logged size is not token capacity fails the check.

Logs: the server_log_id of every result.json under <work>/points, read from the restored DB with logfacts.py
(UNTRUSTED text, parsed with regexes only). A log missing either line gets {"consistent": null}.
Usage: uv run --no-project python -I kv_units.py <work dir> [--ids FILE]
"""
import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logfacts as LF  # noqa: E402

TOL = 0.05


def entry(f: dict) -> dict:
    kv, mc = (f["kv_tokens"] or [None])[0], f["max_conc"]
    if not kv or not mc:
        return {"consistent": None}
    implied = mc[0] * mc[1]
    return {"kv": kv, "implied": implied, "consistent": abs(kv / implied - 1) <= TOL}


def main():
    work = Path(sys.argv[1])
    if "--ids" in sys.argv:
        ids = sorted({x.strip() for x in open(sys.argv[sys.argv.index("--ids") + 1]) if x.strip()}, key=int)
    else:
        ids = set()
        for p in work.glob("points/*/*/result.json"):
            i = (json.loads(p.read_text()).get("source") or {}).get("server_log_id")
            if i:
                ids.add(str(i))
        ids = sorted(ids, key=int)
    with cf.ThreadPoolExecutor(4) as ex:
        out = {i: entry(f) for i, f in ex.map(LF.one, ids)}
    (work / "kv_units.json").write_text(json.dumps(out))
    print(len(out), "logs;", sum(v["consistent"] is False for v in out.values()), "inconsistent")


if __name__ == "__main__":
    main()
