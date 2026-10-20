"""Reconstruct the vLLM serve configuration of InferenceX rows that have no server log, from the
InferenceX recipe at the run's commit (provenance "config from recipe @<commit>").

Inputs: the inventory (g_nolog runs) and the recipe files fetched from SemiAnalysisAI/InferenceX at each
run's head_sha into <recipes>/<sha>/ (benchmarks/*.sh, .github/workflows/*.yml, .github/configs/*.yaml).
For a run (model, precision, GPU, framework vllm):
  image   : .github/configs/nvidia-master.yaml entry <prefix>-<precision>-<gpu>-vllm `image:` if present,
            else the `image:` of the job with `runner: <gpu>` in the model's workflow template
            (.github/workflows/<prefix>-tmpl.yml); the db row's image when the row states one (must agree).
  serve   : every benchmarks/<prefix>_<precision>_<gpu>_{docker,slurm}.sh (the launcher the runner picks);
            the `vllm serve` flags and the inline config.yaml keys; $TP/$CONC/$MAX_MODEL_LEN bound from
            the row (max-model-len from the template input or the generator, isl+osl(+200)).
  EXACT   : one image, and every launcher variant states the same serve flags (so which launcher ran
            does not matter), and no flag depends on an unbound variable.
  PARTIAL : resolvable, but launchers disagree or a value is unbound (listed).
Lost versus a logged run: the logged KV pool size (the check of BLIS's pool), the resolved defaults
the engine printed, the exact vLLM build inside a floating tag, and the HF snapshot sha.
Usage: python -I reconstruct.py --inventory inv.jsonl --recipes DIR --out recon.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re

PREFIX = {"gptoss120b": "gptoss", "llama70b": "70b", "dsr1": "dsr1"}
FLAG = re.compile(r"--([a-z0-9-]+)(?:[= ]+('[^']*'|\"[^\"]*\"|[^\s\\]+))?")


def serve_args(text: str):
    """The vllm serve flags and the heredoc config.yaml keys of a launcher script."""
    cfg = {}
    m = re.search(r"cat > \$?\{?[A-Za-z_]*\}?config\.yaml << ?'?EOF'?\n(.*?)\nEOF", text, re.S) or \
        re.search(r"cat > [^\n]*\.yaml << ?'?EOF'?\n(.*?)\nEOF", text, re.S)
    if m:
        for ln in m.group(1).splitlines():
            if ":" in ln and not ln.strip().startswith("#"):
                k, v = ln.split(":", 1)
                cfg[k.strip()] = v.strip()
    flags = {}
    i = text.find("vllm serve ")
    if i >= 0:
        lines, out = text[i:].splitlines(), []
        for ln in lines:  # the command runs to the first line without a trailing backslash
            out.append(ln.rstrip("\\").strip())
            if not ln.rstrip().endswith("\\"):
                break
        body = " ".join(out).split(" > ")[0]
        for k, v in FLAG.findall(body):
            if k in ("host", "port", "disable-log-requests", "config"):
                continue
            flags[k] = (v or "true").strip("'\"")
    return {"flags": flags, "config_yaml": cfg}


def image_for(root: str, prefix: str, prec: str, gpu: str):
    out = set()
    nm = os.path.join(root, ".github/configs/nvidia-master.yaml")
    if os.path.exists(nm):
        txt = open(nm).read()
        for blk in re.split(r"\n(?=[a-z0-9][a-z0-9.\-]*:\n)", txt):
            head = blk.split(":", 1)[0].strip()
            if head == f"{prefix}-{prec}-{gpu}-vllm":
                m = re.search(r"\n\s*image:\s*['\"]?([^'\"\n]+)", blk)
                if m:
                    out.add(m.group(1).strip())
    tm = os.path.join(root, ".github/workflows", f"{prefix}-tmpl.yml")
    if not out and os.path.exists(tm):
        txt = open(tm).read()
        for job in re.split(r"\n  [a-z0-9-]+:\n", txt):
            if re.search(rf"\n\s*runner:\s*{gpu}\s*\n", job):
                m = re.search(r"\n\s*image:\s*['\"]?([^'\"\n]+)", job)
                if m and "vllm" in m.group(1):
                    out.add(m.group(1).strip())
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--recipes", required=True)
    ap.add_argument("--heads", required=True, help="tsv github_run_id<TAB>head_sha")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    heads = dict(ln.rstrip("\n").split("\t") for ln in open(a.heads) if "\t" in ln)
    stat = collections.Counter()
    why = collections.Counter()
    with open(a.out, "w") as fo:
        for ln in open(a.inventory):
            r = json.loads(ln)
            if r["blocker"] != "g_nolog":
                continue
            ghid = r["source_url"].split("/runs/")[1].split("/")[0]
            sha = heads.get(ghid)
            prefix = PREFIX.get(r["model_slug"])
            res = {"run_id": r["run_id"], "head_sha": sha, "points": len(r["points"])}
            root = os.path.join(a.recipes, sha or "-")
            if not sha or not os.path.isdir(root) or not prefix:
                res["status"] = "UNRESOLVED"; res["why"] = "no recipe at commit" if prefix else "model prefix unknown"
            else:
                imgs = image_for(root, prefix, r["precision"], r["gpu"])
                if r.get("engine_image"):
                    if imgs and r["engine_image"] not in imgs:
                        res["image_conflict"] = {"db": r["engine_image"], "recipe": imgs}
                    imgs = [r["engine_image"]]
                vars_ = []
                for launcher in ("docker", "slurm"):
                    p = os.path.join(root, "benchmarks", f"{prefix}_{r['precision']}_{r['gpu']}_{launcher}.sh")
                    if os.path.exists(p):
                        vars_.append((launcher, serve_args(open(p).read())))
                res["image"] = imgs
                res["launchers"] = {k: v for k, v in vars_}
                same = len({json.dumps(v, sort_keys=True) for _, v in vars_}) == 1
                unbound = sorted({k for _, v in vars_ for k, x in v["flags"].items() if "$" in x and not re.fullmatch(r"\$\{?(TP|CONC|MAX_MODEL_LEN|EP_SIZE|PORT|MODEL)\}?", x)})
                res["unbound_flags"] = unbound
                if len(imgs) == 1 and vars_ and same and not unbound and "image_conflict" not in res:
                    res["status"] = "EXACT"
                elif imgs and vars_:
                    res["status"] = "PARTIAL"
                    res["why"] = "; ".join(x for x in (
                        "launcher variants differ" if not same else "", f"unbound {unbound}" if unbound else "",
                        "image conflict" if "image_conflict" in res else "", f"{len(imgs)} images" if len(imgs) != 1 else "") if x)
                else:
                    res["status"] = "UNRESOLVED"; res["why"] = "no image" if not imgs else "no launcher script"
                res["provenance"] = f"config from recipe @{sha}"
            stat[res["status"]] += res["points"]
            if res.get("why"):
                why[(res["status"], res["why"])] += res["points"]
            fo.write(json.dumps(res, sort_keys=True) + "\n")
    print(dict(stat))
    for k, v in why.most_common(15):
        print(v, k)


if __name__ == "__main__":
    main()
