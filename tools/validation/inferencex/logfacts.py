"""Per server log, the facts the parity rules need (PARITY_FIXES.md): served checkpoint id, quantization, MoE backend
line, num_gpu_blocks_override (logged override lines and whether the run's own args set it), the profiled
"Available KV cache memory", "GPU KV cache size" and "Maximum concurrency" lines.
Usage: uv run --no-project python -I logfacts.py <ids file> <out.json>"""
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import sys as _s
from pathlib import Path
_s.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402


def facts(text: str) -> dict:
    nd = re.search(r"non-default args: (\{.*\})", text)
    ndt = nd.group(1) if nd else ""
    mdl = re.search(r"Initializing a V1 LLM engine \([^)]*\) with config: model='([^']+)'", text)
    ndm = re.search(r"'model': '([^']+)'", ndt)
    conc = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
    return {
        "model": (mdl.group(1) if mdl else None) or (ndm.group(1) if ndm else None),
        "quantization": (re.search(r"quantization=([a-z0-9_-]+)", text) or [None, None])[1],
        "moe_backend": (re.search(r"Using '?([A-Za-z_0-9]+)'? (\w+) MoE backend", text) or [None])[0],
        "override_lines": sorted({f"{a}>{b}" for a, b in re.findall(
            r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)", text)}),
        "override_in_args": "num_gpu_blocks_override" in ndt,
        "max_capture": (re.search(r"'max_cudagraph_capture_size': (\d+)", text) or [None, None])[1],
        "avail_gib": [float(x) for x in re.findall(r"Available KV cache memory: ([\d.]+) GiB", text)][:1],
        "kv_tokens": [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)],
        "max_conc": [float(conc.group(2)), int(conc.group(1).replace(",", ""))] if conc else None,
        "block_size": (re.search(r"'block_size': (\d+)", text) or [None, None])[1],
    }


def one(i):
    t = subprocess.run(["psql", "-d", PX.PGDATABASE, "-At", "-c", f"select server_log from server_logs where id={int(i)}"],
                       capture_output=True, text=True).stdout
    return i, facts(t)


if __name__ == "__main__":
    ids = [x.strip() for x in open(sys.argv[1]) if x.strip()]
    with cf.ThreadPoolExecutor(4) as ex:
        out = dict(ex.map(one, ids))
    json.dump(out, open(sys.argv[2], "w"))
    print(len(out))
