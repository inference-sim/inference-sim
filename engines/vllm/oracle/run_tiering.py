# Original to blis-vllm.
"""Direct traces of vLLM's offload tiering with secondary tiers (design FID-5, §7.6; T5).

A sequence (oracle/scenarios/tiering/*.json) drives a TieringOffloadingSpec's manager with one or
more secondary tiers through calls at the OffloadingManager interface, as
oracle/drive_kv_interfaces.py drives the CPU tier, and records two interfaces: OffloadingManager,
as the connector's scheduler side calls it, and SecondaryTierManager (vllm/v1/kv_offload/tiering/
base.py:135-423@4ba787ec), as the tiering manager calls each tier. core/connector/tiering's
replay test makes the same calls on the port and compares both.

The tiers run vLLM's own code over a real directory (the file system is the transport). Nothing
in vLLM is replaced or written to, so this needs no entry in oracle/mocks.json. Each fs tier is
configured with one load-priority I/O thread and no store-priority thread (n_read_threads 1,
n_write_threads 0, tiering/fs/thread_pool.py:60-100): that thread runs a call's loads, then its
stores, each in submission order. After every driver call the driver waits, by reading only,
until the tier's pool is idle (DualQueueThreadPool.wait_idle) and until its lookup worker has
posted the batch the call flushed. So every job a call submits has finished, in that order,
before the next call, and every lookup a step flushed has resolved before the next step: the
port's replay runs the same transport rule (FSTier.RunAll, then FSTier.ResolveLookups). A lookup
slower than a step is not covered here.

A sequence:
    {"description": "...",
     "spec": {as drive_kv_interfaces.py, with "extra": {"secondary_tiers": [{"type": "fs", ...}]}},
     "calls": [manager calls as drive_kv_interfaces.py,
               {"m": "on_new_request", "ctx": "a", "filter": [{"medium": "cpu"}]},   (a load filter)
               {"m": "on_reset_flushed"},
               {"m": "rm", "tier": 0, "key": "k1"},    another process deletes the key's block file
               {"m": "put", "tier": 0, "key": "k1"}]}  another process stores it
A tier's "root_dir" that starts "@root" is under a fresh directory for the sequence.

An obj tier ("type": "obj", vllm/v1/kv_offload/tiering/obj/manager.py@4ba787ec) reaches its bucket
through NIXL's OBJ backend, which is neither installed nor reachable here: oracle/fake_nixl is an
in-memory NIXL agent and bucket in its place (the byte transport, listed in oracle/mocks.json). Its
transport rule matches the fs tier's: after each call every transfer the call submitted finishes,
in submission order (unless held), and every flushed lookup batch resolves. For an obj tier "rm" and "put" delete
or store the key's object in the tier's bucket, and the sequence may also call
               {"m": "deny", "tier": 0}       the bucket refuses writes (each WRITE ends "ERR")
               {"m": "allow", "tier": 0}      ... and accepts them again
               {"m": "hold", "tier": 0}       the tier's transfers stay in flight across calls
               {"m": "release", "tier": 0}    ... and finish after each call again
               {"m": "refuse_next", "tier": 0}  the tier's next transfer fails at submission (transfer() "ERR")

Usage:
    python -I oracle/run_tiering.py <vllm checkout> <sequence dir> <out dir> [--check <golden dir>]
"""

import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = ".tiering.json"
WAIT_S = 30.0
TIER_METHODS = ("lookup", "submit_store", "submit_load", "get_finished_jobs", "has_pending_work", "take_events",
                "touch", "on_new_request", "on_request_finished", "serve_external_requests", "on_schedule_end",
                "drain_jobs")


def tier_monitor():
    sys.path.insert(0, HERE)
    from kv_interfaces import Monitor  # noqa: E402

    class TierMonitor(Monitor):
        """Monitor, with the SecondaryTierManager interface beside the two it records."""

        def __init__(self):
            super().__init__()
            from vllm.v1.kv_offload.tiering.base import SecondaryTierManager
            from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
            from vllm.v1.kv_offload.tiering.obj.manager import ObjectStoreSecondaryTierManager
            self.trace["tier"] = []
            for cls in (SecondaryTierManager, FileSystemTierManager, ObjectStoreSecondaryTierManager):
                for name in TIER_METHODS:
                    f = cls.__dict__.get(name)
                    if callable(f) and hasattr(f, "__code__"):
                        self.codes[f.__code__] = ("tier", name)

        def use_tiers(self, tiers) -> None:
            self.tiers = list(tiers)

        def tier_index(self, t) -> int:
            return self.tiers.index(t)

        def job(self, j) -> dict:
            return {"id": int(j.job_id), "keys": self.keys(j.keys), "chunk_ids": [int(c) for c in j.chunk_ids],
                    "promotion": bool(j.is_promotion), **self.context(j.req_context)}

        def args(self, iface, m, f):
            if iface != "tier":
                return super().args(iface, m, f)
            a = f.f_locals
            out = {"tier": self.tier_index(a["self"])}
            if "req_context" in a:
                out.update(self.context(a["req_context"]))
            if m == "lookup":
                out["key"] = self.key(a["key"])
            elif m in ("submit_store", "submit_load"):
                out["job"] = self.job(a["job_metadata"])
            elif m == "touch":
                out["keys"] = self.keys(a["keys"])
            elif m == "on_schedule_end":
                ctx = a["context"]
                out["new"], out["preempted"] = sorted(ctx.new_req_ids), sorted(ctx.preempted_req_ids)
            return out

        def ret(self, iface, m, v, f):
            if iface != "tier":
                return super().ret(iface, m, v, f)
            if v is None:
                return None
            if m == "lookup":
                return v.name
            if m == "on_new_request":
                return {"policy": v.policy.name}
            if m == "has_pending_work":
                return bool(v)
            if m == "get_finished_jobs":
                return [{"id": int(r.job_id), "success": bool(r.success),
                         "successful_keys": None if r.successful_keys is None else self.keys(r.successful_keys),
                         "transfer_bytes": r.transfer_bytes} for r in v]
            return repr(v)

        def event(self, e) -> dict:
            out = super().event(e)
            out["locality"] = None if e.locality is None else e.locality.name
            return out

    return TierMonitor()


def mapper(tier):
    """The tier's FileMapper: fs names it file_mapper, obj _file_mapper."""
    return tier.file_mapper if tier.tier_type == "fs" else tier._file_mapper


def wait_transport(tier) -> None:
    """Read-only waits: the tier's I/O is done (fs: the pool is idle; obj: the fake NIXL agent
    finishes the transfers in flight), and its lookup worker posted what was flushed."""
    from vllm.v1.kv_offload.tiering.async_lookup import LookupPhase
    if tier.tier_type == "obj":
        import nixl._api as fake_nixl
        fake_nixl.complete_all()
    else:
        tier._pool.wait_idle()
    lm = tier._lookup_manager
    deadline = time.monotonic() + WAIT_S
    while any(s.phase is LookupPhase.IN_FLIGHT for s in list(lm._lookup_state.values())) and \
            lm._pending_results.qsize() == 0:
        if time.monotonic() > deadline:
            raise RuntimeError("run_tiering: the lookup worker did not post within %.0fs" % WAIT_S)
        time.sleep(0.001)


def run_sequence(vllm_root: str, path: str) -> dict:
    if vllm_root not in sys.path:
        sys.path.append(vllm_root)
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "fake_nixl"))
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    import nixl._api as fake_nixl  # noqa: E402
    fake_nixl.reset()
    from drive_kv_interfaces import call_manager, key_of, offloading_config  # noqa: E402
    from run_scheduler import provenance  # noqa: E402
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import _parse_tier_filter  # noqa: E402
    from vllm.v1.kv_offload.base import ReqContext, ScheduleEndContext  # noqa: E402
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory  # noqa: E402

    with open(path) as f:
        sq = json.load(f)
    name = os.path.basename(path)[:-5]
    root = tempfile.mkdtemp(prefix="blis-tiering-")
    spec_in = json.loads(json.dumps(sq["spec"]))
    for t in spec_in.get("extra", {}).get("secondary_tiers", []):
        if str(t.get("root_dir", "")).startswith("@root"):
            t["root_dir"] = root + t["root_dir"][len("@root"):]
    ctxs: dict = {}

    def ctx(c: dict):
        if c["ctx"] not in ctxs:
            kw = {}
            if "filter" in c:
                kw["load_tier_filter"] = _parse_tier_filter(c["filter"])
            ctxs[c["ctx"]] = ReqContext(req_id=c["ctx"], **kw)
        return ctxs[c["ctx"]]

    def ks(c: dict) -> list:
        return [key_of(k) for k in c["keys"]]

    mon = tier_monitor()
    files: dict = {}
    try:
        with mon:
            spec = OffloadingSpecFactory.create_spec(offloading_config(spec_in, name))
            mgr = spec.get_manager()
            mon.use_tiers(mgr.secondary_tiers)
            try:
                for c in sq["calls"]:
                    m = c["m"]
                    if m in ("rm", "put") and mon.tiers[c["tier"]].tier_type == "obj":
                        tier = mon.tiers[c["tier"]]
                        p = tier._file_mapper.get_file_name(key_of(c["key"]))
                        if m == "rm":
                            fake_nixl.remove(tier._agent, p)
                        else:
                            fake_nixl.put(tier._agent, p, b"\0" * tier.block_size_bytes)
                    elif m in ("hold", "release"):
                        fake_nixl.hold(mon.tiers[c["tier"]]._agent, m == "hold")
                    elif m in ("deny", "allow"):
                        fake_nixl.deny(mon.tiers[c["tier"]]._agent, m == "deny")
                    elif m == "refuse_next":
                        fake_nixl.refuse_next(mon.tiers[c["tier"]]._agent)
                    elif m in ("rm", "put"):
                        tier = mon.tiers[c["tier"]]
                        p = tier.file_mapper.get_file_name(key_of(c["key"]))
                        if m == "rm":
                            os.remove(p)
                        else:
                            os.makedirs(os.path.dirname(p), exist_ok=True)
                            with open(p, "wb") as fh:
                                fh.write(b"\0" * tier.block_size_bytes)
                    elif m == "on_reset_flushed":
                        mgr.on_reset_flushed()
                    else:
                        if m == "on_new_request":
                            ctx(c)
                        call_manager(mgr, m, c, ctx, ks, ScheduleEndContext)
                    for tier in mon.tiers:
                        wait_transport(tier)
                    # Each key's block file in each tier, relative to the sequence's directory.
                    for k in c.get("keys", []) + ([c["key"]] if "key" in c else []):
                        files.setdefault(k, [os.path.relpath(t.file_mapper.get_file_name(key_of(k)), root)
                                             if t.tier_type == "fs" else t._file_mapper.get_file_name(key_of(k))
                                             for t in mon.tiers])
            finally:
                mgr.shutdown()
        tiers = [{"type": t.tier_type, "block_size_bytes": int(t.block_size_bytes),
                  "run_config": mapper(t).get_run_config(), "rank": int(mapper(t).rank),
                  "locality": None if t.locality is None else t.locality.name,
                  "events": t.events is not None,
                  **({"bucket": t._agent.bucket_name} if t.tier_type == "obj" else {})} for t in mon.tiers]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    prov = provenance(vllm_root, path)
    prov["kind"] = "direct"
    return {"provenance": prov, "tiers": tiers, "files": files, **mon.trace}


def compare(out_dir: str, golden_dir: str, names: list) -> int:
    bad = 0
    for n in names:
        with open(os.path.join(out_dir, n)) as a:
            got = json.load(a)
        gp = os.path.join(golden_dir, n)
        if not os.path.exists(gp):
            print("missing golden %s" % gp)
            bad += 1
            continue
        with open(gp) as b:
            want = json.load(b)
        got.pop("provenance", None)
        want.pop("provenance", None)
        if got != want:
            print("%s differs from its golden" % n)
            bad += 1
    return bad


def main(argv) -> int:
    check = None
    if "--check" in argv:
        i = argv.index("--check")
        check, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    vllm_root, seq_dir, out_dir = argv
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for f in sorted(x for x in os.listdir(seq_dir) if x.endswith(".json")):
        trace = run_sequence(vllm_root, os.path.join(seq_dir, f))
        n = f[:-5] + SUFFIX
        with open(os.path.join(out_dir, n), "w") as fh:
            json.dump(trace, fh, indent=1, sort_keys=True)
        names.append(n)
    if check:
        bad = compare(out_dir, check, names)
        if bad:
            print("%d tiering trace(s) differ" % bad)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
