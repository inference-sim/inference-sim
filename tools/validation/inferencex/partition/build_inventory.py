"""Build the InferenceX end-to-end run inventory (inventory.jsonl, written to --out).

Sources (all public; nothing here is redistributed, only identifiers and counts):
  - InferenceX db-dump/2026-09-28 (SemiAnalysisAI/InferenceX-app release), restored locally:
    benchmark_results x configs x workflow_runs (rows.tsv) and server_logs (logs*.tsv);
  - GitHub Actions bmk_* artifacts of SemiAnalysisAI/InferenceX "Run Sweep" runs created on or
    after 2026-09-27 (post-dump), downloaded to data/external/inferencex-gh-bmk-postdump;
  - the request-level bindings of validation/runs/ix_dsv41/artifact_binding.json.

A RUN is one (GitHub run, config, benchmark_type, isl/osl, image, offload) sweep; its levels are
the concurrencies measured. A POINT is run x concurrency: point_id = sha1 of the run key and the
concurrency, stable under re-extraction, which the partition hashes.

Blocker classes (rules, fixed before any scoring):
  AMD          GPU is AMD (out of scope by decision; recorded, never scored)
  d_engine     engine is not vLLM (sglang, trt, atom, tilert, dynamo-sglang/trt, mori, mooncake-atom)
  b_hardware   NVIDIA GPU with no blis-catalog hardware file
  a_model      model with no blis-catalog entry
  c_coeff      model family whose attention / indexer family has no registry coefficients on that GPU
  e_config     configuration BLIS cannot state (connector not modelled, multinode DP>16, ...)
  f_workload   workload BLIS cannot replay as InferenceX ran it
  g_nolog      no server log (engine settings unknown), so not apples to apples
  ok           scoreable today
Usage: uv run --no-project --with pyyaml python -I build_inventory.py --rows R --logs L [--logs L2]
       --bmk-dir D --binding B --catalog C --registry G --out inventory.jsonl
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re

AMD = ("mi300x", "mi325x", "mi355x")
DB_URL = "https://github.com/SemiAnalysisAI/InferenceX-app/releases/tag/db-dump/2026-09-28"
GH = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/"
# InferenceX model slug -> (HF repo as InferenceX serves it, blis-catalog name or None)
MODELS = {
    "gptoss120b": ("openai/gpt-oss-120b", "gpt-oss-120b"),
    "dsr1": ("deepseek-ai/DeepSeek-R1-0528", "deepseek-v3"),
    "llama70b": ("meta-llama/Llama-3.1-70B","llama-3.1-70b-instruct"),
    "minimaxm3": ("MiniMaxAI/MiniMax-M3", "minimax-m3"),
    "dsv4": ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-v4-pro"),
    "qwen3.5": ("Qwen/Qwen3.5-397B-A17B", "qwen3.5-397b-a17b"),
    "minimaxm2.5": ("MiniMaxAI/MiniMax-M2.5", "minimax-m2.5"),
    "kimik2.5": ("moonshotai/Kimi-K2.5", "kimi-k2.5"),
    "glm5": ("zai-org/GLM-5-FP8", "glm-5"),
    "dsv41flash": ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-v4.1-flash"),
    "kimik3": ("moonshotai/Kimi-K3", "kimi-k3"),
    "glm5.2": ("zai-org/GLM-5.2", "glm-5.2"),
    "glm5.1": ("zai-org/GLM-5.1", None),
    "qwen3.8next": ("Qwen/Qwen3.8-Next", None),
    "kimik2.6": ("moonshotai/Kimi-K2.6", "kimi-k2.6"),
    "glm5.3": ("zai-org/GLM-5.3", "glm-5.3"),
    "dsv4flash": ("deepseek-ai/DeepSeek-V4-Flash", None),
    "musespark": (None, None),
}
HW = {"h100": "h100", "h200": "h200", "b200": "b200", "b300": "b300", "gb200": "gb200-nvl72",
      "gb300": "gb300", "rtx6000pro": None, "vr200": None}
VLLM = ("vllm", "dynamo-vllm", "llmd-vllm", "vllm-disagg")


def h(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def read_logs(paths):
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in open(p, errors="replace"):
            f = ln.rstrip("\n").split("\t")
            if len(f) < 9:
                f += [r"\N"] * (9 - len(f))
            nul = lambda x: None if x in (r"\N", "") else x  # noqa: E731
            sid = f[0]
            if len(f) >= 9 and p.endswith("logs2.tsv"):
                ver, sgl, rust, snap, spec, conn, v2, nd = f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]
            else:  # logs.tsv: id ver rust snap spec conn nondefault
                ver, sgl, rust, snap, spec, conn, v2, nd = f[1], r"\N", f[2], f[3], f[4], f[5], r"\N", f[6]
            prev = out.get(sid, {})
            rec = {"engine_version": nul(ver) or prev.get("engine_version"),
                   "rust_frontend": (rust == "t") or prev.get("rust_frontend", False),
                   "hf_revision": nul(snap) or prev.get("hf_revision"),
                   "spec_config": nul(spec) or prev.get("spec_config"),
                   "kv_connector": nul(conn) or prev.get("kv_connector"),
                   "v2_runner": (nul(v2) is not None and "1" in (v2 or "")) or prev.get("v2_runner", False),
                   "args": (nul(nd) or prev.get("args") or "")[:400]}
            out[sid] = rec
    return out


def catalog_sets(cat, reg):
    models = set(os.listdir(os.path.join(cat, "models")))
    hw = {f[:-5] for f in os.listdir(os.path.join(cat, "hardware")) if f.endswith(".yaml")}
    return models, hw


UNLOCK = {
    "AMD": "none (out of scope by decision)",
    "d_engine": "none (BLIS models vLLM only)",
    "b_hardware": "catalog hardware entry (vendor spec) + registry coefficients for the part",
    "a_model": "catalog: derive model from HF config.json at a pinned revision (blis-catalog scripts/derive_graph.py)",
    "a_model:modelopt_layout": "catalog: ModelOpt layout reader in derive_graph (draft issue 1)",
    "a_model:hf_quant_config": "catalog+schemas: graph provenance over hf_quant_config.json (draft issue 3)",
    "a_model:dspark3": "catalog: DSpark draft module in handler_deepseek_v4 (draft issue 4)",
    "a_model:qwen4exp": "catalog: Qwen4Exp handler (draft issue 5)",
    "a_model:gated": "user accepts the HF gate, then derive",
    "c_coeff": "registry: fit the model family's operator coefficients on this GPU from public AISimulate/FPM data",
    "e_config": "schema/inference-sim: model the stated connector / layout (see blocker_detail)",
    "f_workload": "schemas+inference-sim: transcribe the client's replay scenario/version",
    "g_nolog": "harness: no server log in the db row; fetch the run's GitHub server_logs_* artifact (public until expiry) or score PARTIAL with vLLM defaults resolved",
    "gated": "user accepts the dataset's HF licence gate (not accepted on the user's behalf), then fetch",
    "ok": "",
}
# GPU recovered from the GitHub job runner name when the artifact's hw field is empty
# (CATALOG_WORK.md: runner h200-dgxc-slurm_00/_02, label cluster:h200-dgxc).
GPU_EVIDENCE = {
    "38336419249": ("h200", "job runner h200-dgxc-slurm_00 (cluster:h200-dgxc)"),
    "38972117807": ("h200", "job runners h200-dgxc-slurm_00/_02 (cluster:h200-dgxc)"),
}
# Checkpoints the catalog cannot yet price correctly (CATALOG_WORK.md, parked), keyed by a
# substring of the served/HF repo. Named blockers, checked before the generic model test.
PARKED = [
    ("GLM-5.2-NVFP4", "a_model:modelopt_layout", "ModelOpt NVFP4 per-module quant (ignore list) not read by derive_graph"),
    ("Qwen3.5-397B-A17B-NVFP4", "a_model:modelopt_layout", "ModelOpt NVFP4 per-module / mixed quant not read by derive_graph"),
    ("MiniMax-M3-NVFP4", "a_model:modelopt_layout", "ModelOpt MIXED_PRECISION (NVFP4 experts + MXFP8 rest); schema has no mxfp8 dtype"),
    ("DeepSeek-R1-0528-NVFP4", "a_model:hf_quant_config", "quant layout only in hf_quant_config.json; graph provenance records one file"),
    ("DeepSeek-R1-0528-FP4", "a_model:hf_quant_config", "quant layout only in hf_quant_config.json; graph provenance records one file"),
    ("DeepSeek-V4-Pro-0813", "a_model:dspark3", "3-layer DSpark draft; handler_deepseek_v4 supports a 1-layer MTP"),
    ("Qwen3.8-Flash-Next", "a_model:qwen4exp", "Qwen4ExpForConditionalGeneration: no catalog handler"),
    ("Muse-Spark", "a_model:gated", "HF gated (manual); config not fetched"),
]
# Known model-side pricing errors of entries that ARE scored (CATALOG_WORK.md); reported, not blocking.
MODEL_CAVEATS = [
    ("MiniMax-M3-MXFP8", "MXFP8 resident scales (1 B / 32 elements) not priced: weight bytes about -3%"),
    ("Qwen3.5-397B-A17B-FP8", "lm_head excluded from fp8 in the checkpoint but priced at fp8"),
    ("GLM-5.2-FP8", "lm_head excluded from fp8 in the checkpoint but priced at fp8"),
]
SPARSE = {"deepseek-v4-pro", "minimax-m3", "glm-5", "glm-5.2", "glm-5.3", "deepseek-v4.1-flash", "kimi-k3"}


def classify(r, cat_models, cat_hw, v41_hw):
    hwk = r["gpu"]
    if hwk in AMD:
        return "AMD", "out of scope (AMD)"
    if r["engine"] not in VLLM:
        return "d_engine", f"engine {r['engine']} (BLIS models vLLM)"
    if not hwk:
        return "b_hardware", "GPU not recorded in the artifact (hw field empty)"
    if hwk == "vr200":
        return "b_hardware", "Vera Rubin (vr200): no public spec/driver figures gathered (future work)"
    served = f"{r.get('hf_repo_served') or ''} {r.get('hf_repo') or ''}"
    for key, cls, why in PARKED:
        if key in served and r["engine"] in VLLM:
            return cls, why
    if HW.get(hwk) is None or HW[hwk] not in cat_hw:
        return "b_hardware", f"no catalog hardware for {hwk}"
    if r["catalog_model"] is None or r["catalog_model"] not in cat_models:
        return "a_model", f"model {r['model_slug']} not in blis-catalog"
    if r["catalog_model"] == "deepseek-v4.1-flash" and HW[hwk] not in v41_hw:
        return "c_coeff", f"DSV4.1 sparse-MLA/indexer coefficients scoped to {sorted(v41_hw)} only"
    if r["is_multinode"] and r["engine"] == "dynamo-vllm":
        return "e_config", "Dynamo multinode serving (frontend/router not llm-d; P/D over Dynamo)"
    conn = r.get("kv_connector")
    be = r.get("offload_backend")
    if r["offload"] == "on":
        if be in ("mooncake", "lmcache") or conn in ("MooncakeConnector", "MooncakeStoreConnector", "LMCacheMPConnector"):
            return "e_config", f"offload backend {be or conn}: third-party store not modelled (unlock: connector model)"
        if be == "vllm-simple" or conn == "SimpleCPUOffloadConnector":
            return "e_config", "SimpleCPUOffloadConnector refused by inference-sim cmd/kv_offload_scenario.go (unlock: port v0.31.0 simple_kv_offload)"
        if conn not in (None, "OffloadingConnector") and be != "vllm-native":
            return "e_config", f"KV connector {conn or be} not modelled"
    if r["parallel"].get("disagg") and r["engine"] in ("vllm", "vllm-disagg") and r["is_multinode"]:
        return "e_config", "multinode vLLM P/D outside llm-d (unlock: harness mapping of the run's router)"
    if not r.get("has_server_log"):
        return "g_nolog", "no server log: engine settings unknown"
    if r["scenario"] == "agentic_traces" and r.get("aiperf_client") == "agentx-1.x":
        return "f_workload", "agentx 1.x client: harness version not transcribed (0.12.0 only)"
    return "ok", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--logs", action="append", default=[])
    ap.add_argument("--bmk-dir")
    ap.add_argument("--binding", action="append", default=[])
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--recipe-recon", default=None, help="recipe_recon.jsonl (validation/coverage/recipe/reconstruct.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logs = read_logs(a.logs)
    cat_models, cat_hw = catalog_sets(a.catalog, a.registry)
    v41_hw = {"h100", "h200"}
    rows_sha = hashlib.sha256(open(a.rows, "rb").read()).hexdigest()
    runs = collections.OrderedDict()
    for ln in open(a.rows, errors="replace"):
        f = ln.rstrip("\n").split("\t")
        (rid, wfid, ghid, att, head, url, cid, model, hw, fw, prec, spec, disagg, mn, ptp, pep, pdpa, pnw,
         dtp, dep, ddpa, dnw, npg, ndg, btype, date, isl, osl, conc, image, err, off, slid, rfp, offb,
         tpot, ttft, haskv, trid) = f
        nul = lambda x: None if x == r"\N" else x  # noqa: E731
        key = (ghid, att, cid, btype, nul(isl), nul(osl), nul(image), off)
        run_id = "ixdb-" + h(*key)
        lg = logs.get(slid) if nul(slid) else None
        r = runs.get(run_id)
        if r is None:
            repo, cm = MODELS.get(model, (None, None))
            r = runs[run_id] = {
                "run_id": run_id, "source": "inferencex-db", "model_slug": model, "hf_repo": repo,
                "hf_revision": None, "catalog_model": cm, "gpu": hw, "engine": fw,
                "engine_image": nul(image), "engine_version": None, "precision": prec,
                "parallel": {"disagg": disagg == "t", "prefill": {"tp": int(ptp), "ep": int(pep), "dp_attention": pdpa == "t",
                             "workers": int(pnw), "gpus": int(npg)},
                             "decode": {"tp": int(dtp), "ep": int(dep), "dp_attention": ddpa == "t",
                                        "workers": int(dnw), "gpus": int(ndg)}},
                "is_multinode": mn == "t", "spec_method": spec, "spec_config": None,
                "offload": off, "offload_backend": nul(offb), "kv_connector": None,
                "frontend": None, "v2_runner": None,
                "scenario": btype, "isl": int(isl) if nul(isl) else None, "osl": int(osl) if nul(osl) else None,
                "date": date, "config_id": int(cid), "recipe_fingerprint": nul(rfp),
                "levels": [], "rows": {}, "has_server_log": False, "errors": 0,
                "granularity": "aggregate (db metrics jsonb)",
                "source_url": f"{GH}{ghid}" + (f"/attempts/{att}" if att not in ("1", r"\N") else ""),
                "source_sha256": {"db_rows_tsv": rows_sha, "db_dump": DB_URL}}
            if btype == "agentic_traces":
                r["aiperf_client"] = "aiperf-fork-0.12.x"
        c = int(conc)
        if err == "t":
            r["errors"] += 1
        else:
            if c not in r["levels"]:
                r["levels"].append(c)
            r["rows"].setdefault(str(c), []).append(int(rid))
        if nul(slid):
            r["has_server_log"] = True
            r.setdefault("server_log_ids", set()).add(slid)
        if lg:
            for k_src, k_dst in (("engine_version", "engine_version"), ("hf_revision", "hf_revision"),
                                 ("spec_config", "spec_config"), ("kv_connector", "kv_connector")):
                if lg.get(k_src) and not r.get(k_dst):
                    r[k_dst] = lg[k_src]
            m = re.search(r"'model': '([^']+)'", lg.get("args") or "")
            if m and not r.get("hf_repo_served"):
                r["hf_repo_served"] = m.group(1)
            if fw in VLLM:
                r["frontend"] = "rust" if lg.get("rust_frontend") else (r["frontend"] or "python")
                r["v2_runner"] = bool(lg.get("v2_runner")) or bool(r["v2_runner"])
    # request-level upgrades from the bound DSV4.1 anchors
    arm_rows = {}
    for bpath in a.binding:
        if not os.path.exists(bpath):
            continue
        bind = json.load(open(bpath))
        for arm, lv in bind.items():
            for c, x in lv.items():
                b = x.get("bound")
                if b:
                    arm_rows[b["row_id"]] = (arm, c, b["dir"], b["jsonl_sha256"])
    if arm_rows:
        for r in runs.values():
            for c, ids in r["rows"].items():
                for i in ids:
                    if i in arm_rows:
                        r["granularity"] = "request-level (profile_export.jsonl bound to db row) + aggregate"
                        r.setdefault("request_level", {})[c] = {"arm": arm_rows[i][0], "dir": arm_rows[i][2],
                                                               "jsonl_sha256": arm_rows[i][3]}
    # post-dump GitHub artifacts (aggregate bmk json; request-level agentic_ artifacts exist alongside)
    if a.bmk_dir and os.path.isdir(a.bmk_dir):
        idx = {}
        ip = os.path.join(a.bmk_dir, "index.tsv")
        if os.path.exists(ip):
            for ln in open(ip):
                run, d, aid, name, zsha = ln.rstrip("\n").split("\t")
                idx[aid] = (run, d, name, zsha)
        for p in sorted(glob.glob(os.path.join(a.bmk_dir, "json", "*.json"))):
            aid = os.path.basename(p)[:-5]
            try:
                d = json.load(open(p))
            except Exception:
                continue
            run, date, name, zsha = idx.get(aid, (None, None, None, None))
            slug = d.get("infmax_model_prefix")
            hwlab = str(d.get("hw", "")).replace("cluster:", "").split("-")[0]
            agentic = name and name.startswith("bmk_agentic")
            off = "on" if str(d.get("kv_offloading", "none")) not in ("none", "None", "") else "off"
            key = (run, slug, hwlab, d.get("framework"), d.get("image"), d.get("tp"), d.get("ep"),
                   d.get("dp_attention"), d.get("disagg"), d.get("prefill_tp"), d.get("decode_tp"),
                   d.get("prefill_num_workers"), d.get("decode_num_workers"), d.get("spec_decoding"), off,
                   d.get("isl"), d.get("osl"), d.get("precision"), d.get("recipe_fingerprint") if not agentic else None)
            run_id = "ixgh-" + h(*key)
            r = runs.get(run_id)
            if r is None:
                repo, cm = MODELS.get(slug, (d.get("model"), None))
                r = runs[run_id] = {
                    "run_id": run_id, "source": "github-artifact", "model_slug": slug, "hf_repo": d.get("model") or repo,
                    "hf_revision": None, "catalog_model": cm, "gpu": hwlab, "engine": d.get("framework"),
                    "engine_image": d.get("image"), "engine_version": None, "precision": d.get("precision"),
                    "parallel": {"disagg": str(d.get("disagg")) in ("True", "true"),
                                 "tp": d.get("tp"), "ep": d.get("ep"), "dp_attention": str(d.get("dp_attention")) == "true",
                                 "prefill": {"tp": d.get("prefill_tp"), "ep": d.get("prefill_ep"), "workers": d.get("prefill_num_workers")},
                                 "decode": {"tp": d.get("decode_tp"), "ep": d.get("decode_ep"), "workers": d.get("decode_num_workers")},
                                 "num_gpus": d.get("num_gpus")},
                    "is_multinode": str(d.get("is_multinode")) in ("True", "true"),
                    "spec_method": d.get("spec_decoding"), "spec_config": None, "offload": off,
                    "offload_backend": (d.get("kv_offload_backend") or {}).get("name") if isinstance(d.get("kv_offload_backend"), dict) else d.get("kv_offload_backend"), "kv_connector": None,
                    "kv_p2p_transfer": d.get("kv_p2p_transfer"), "frontend": None, "v2_runner": None,
                    "scenario": "agentic_traces" if agentic else "single_turn",
                    "isl": d.get("isl"), "osl": d.get("osl"), "date": date, "levels": [], "rows": {},
                    "has_server_log": True, "errors": 0,
                    "granularity": ("aggregate (bmk json); request-level profile_export.jsonl in the run's agentic_ artifact"
                                    if agentic else "aggregate (bmk json)"),
                    "source_url": f"{GH}{run}", "source_sha256": {}}
                if agentic:
                    r["aiperf_client"] = "agentx-1.x" if str(d.get("scenario_type")) == "agentic-coding" else "unknown"
            c = int(d.get("conc"))
            if c not in r["levels"]:
                r["levels"].append(c)
            r["rows"].setdefault(str(c), []).append(f"artifact:{aid}")
            r["source_sha256"][f"artifact:{aid}"] = zsha
    if a.extra and os.path.exists(a.extra):
        for ln in open(a.extra):
            if ln.strip():
                x = json.loads(ln)
                x.setdefault("source", "inventory_extra")
                src = str(x["source"])
                if not (src.startswith("inferencex") or x.get("blocker") == "gated"):
                    continue
                if x.get("blocker") == "gated":
                    # Third-party end-to-end data behind a licence gate the user has not accepted:
                    # recorded, re-keyed to this inventory's point-id scheme, never partitioned or scored.
                    x["points"] = [{"point_id": h(x["run_id"], p["conc"]), "conc": p["conc"]} for p in x.get("points", [])]
                    x["levels"] = sorted({p["conc"] for p in x["points"]})
                runs.setdefault(x["run_id"], x)
    recon = {}
    if a.recipe_recon and os.path.exists(a.recipe_recon):
        for ln in open(a.recipe_recon):
            x = json.loads(ln)
            recon[x["run_id"]] = x
    counts = collections.Counter()
    with open(a.out, "w") as fo:
        for r in runs.values():
            r["levels"].sort()
            if "server_log_ids" in r:
                r["server_log_ids"] = sorted(r["server_log_ids"])[:50]
                r["log_parsed"] = bool(r.get("engine_version") or r.get("hf_repo_served"))
            ghid = (r.get("source_url") or "").split("/runs/")[-1].split("/")[0]
            if not r.get("gpu") and ghid in GPU_EVIDENCE:
                r["gpu"], r["gpu_evidence"] = GPU_EVIDENCE[ghid]
            served = f"{r.get('hf_repo_served') or ''} {r.get('hf_repo') or ''}"
            cav = [why for key, why in MODEL_CAVEATS if key in served]
            if cav:
                r["model_caveats"] = cav
            if "blocker" not in r:
                r["blocker"], r["blocker_detail"] = classify(r, cat_models, cat_hw, v41_hw)
                rc = recon.get(r["run_id"])
                if r["blocker"] == "g_nolog" and rc and rc["status"] == "EXACT":
                    r["blocker_original"] = "g_nolog"
                    r["blocker"], r["blocker_detail"] = "ok", "no server log; " + rc["provenance"]
                    r["config_provenance"] = rc["provenance"]
                    r["config_from_recipe"] = {"image": rc["image"][0], "launchers": rc["launchers"]}
                    r["config_lost_vs_log"] = ["logged KV pool size (pool check)", "engine-resolved defaults",
                                               "vLLM build inside the image tag", "HF snapshot sha"]
                    r["engine_image_from_recipe"] = rc["image"][0]  # engine_image stays the db column (partition key)
            r["unlock"] = UNLOCK.get(r["blocker"], "")
            if r.get("blocker") != "gated":
                r["points"] = [{"point_id": h(r["run_id"], c), "conc": c} for c in r["levels"]]
            counts[(r["blocker"], r["scenario"])] += len(r["points"])
            fo.write(json.dumps(r, sort_keys=True) + "\n")
    for k, v in sorted(counts.items()):
        print(k, v)
    print("runs", len(runs), "points", sum(counts.values()))


if __name__ == "__main__":
    main()
