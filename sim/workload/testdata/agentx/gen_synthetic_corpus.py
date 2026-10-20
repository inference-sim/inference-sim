"""Write weka_v7_synthetic.jsonl: hand-built Weka v7 traces for the shapes the real corpus
does not exercise. (`uv run --no-project python -I gen_synthetic_corpus.py > weka_v7_synthetic.jsonl`)

  release2   at the root's turn-0 return, a worker turn held by the barrier on that turn is
             released while a flat chain spawns with offset 0 and is itself barrier-held on the
             released turn, and a subagent's next turn sleeps on a long scheduler timer: aiperf
             issues the released turn before its idle-cap check only because the spawn's
             asyncio.gather yields (replay_dependencies.py:233-238)
  geometry   partial last blocks (in % 64 != 0), a hashed partial block (len(hash_ids) =
             in // 64 + 1), truncated hash lists (len(hash_ids) < in // 64), a compaction
             pull-back, a block-aligned pull-back (assistant caps), zero outputs
  release    a flattened worker turn held by the replay barrier until the root's turn 0
             returns, at the same instant a subagent spawns with offset 0 (immediate)
  quick      three short turns, so one lane recycles many times and reaches the window end
  background a subagent outliving the root's last turn (a background branch), two
             subagents sharing a join group, and a subagent whose inner requests fork
"""

import json


def req(t, api, n_in, out, hashes):
    return {"type": "s", "t": t, "api_time": api, "in": n_in, "out": out, "model": "m", "hash_ids": hashes}


def sub(t, agent, dur_ms, inner):
    return {"type": "subagent", "t": t, "agent_id": agent, "subagent_type": "Explore", "duration_ms": dur_ms,
            "status": "completed", "models": ["m"], "requests": inner}


def trace(tid, reqs):
    return {"id": tid, "models": ["m"], "block_size": 64, "hash_id_scope": "local", "requests": reqs}


def req_m(t, api, n_in, out, hashes, model):
    r = req(t, api, n_in, out, hashes)
    r["model"] = model
    return r


traces = [
    trace("release2", [
        req(0.0, 1.0, 64 * 3, 200, [1, 2, 3]),                    # turn 0: 2.5 s of service
        sub(0.05, "s", 120000.0, [req(0.05, 0.1, 64 * 2, 1, [70, 71]),
                                  req(60.0, 0.1, 64 * 3, 1, [70, 71, 72])]),   # 60 s timer
        req(0.1, 0.1, 64 * 4, 1, [1, 2, 3, 50]),                  # worker forks during turn 0
        req(1.05, 0.5, 64 * 5, 1, [1, 2, 3, 50, 51]),             # waits on turn 0 (barrier)
        req_m(1.7, 9.0, 64 * 4, 1, [1, 2, 3, 60], "m2"),          # background branch: spawns at
                                                                  # turn 0's return with offset 0
                                                                  # and waits on the worker's turn 1
        req(10.0, 1.0, 64 * 4, 5, [1, 2, 3, 4]),
    ]),
    trace("geometry", [
        req(0.0, 1.0, 64 * 3 + 10, 70, [1, 2, 3, 4]),             # hashed partial block
        req(4.0, 1.0, 64 * 5 + 33, 130, [1, 2, 3, 4, 5, 6]),      # lcp 4 > previous covered 3
        req(9.0, 2.0, 64 * 8, 0, [1, 2, 3, 4, 5]),                # truncated: 5 of 8 blocks
        req(15.0, 1.0, 64 * 4 + 5, 200, [1, 2, 3, 9]),            # compaction pull-back
        req(20.0, 1.0, 64 * 6, 300, [1, 2, 3, 9, 10, 11]),
        req(26.0, 1.0, 64 * 4, 64, [1, 2, 3, 9]),                 # block-aligned pull-back
        req(31.0, 1.0, 64 * 7 + 1, 1, [1, 2, 3, 9, 12, 13, 14]),
    ]),
    trace("release", [
        req(0.0, 1.0, 64 * 4, 40, [21, 22, 23, 24]),
        req(0.2, 0.5, 64 * 5, 20, [21, 22, 23, 24, 50]),          # forks while turn 0 runs: flat chain
        sub(1.5, "s1", 2500.0, [req(1.5, 1.0, 64 * 3, 30, [80, 81, 82]),
                                req(3.0, 0.5, 64 * 4, 20, [80, 81, 82, 83])]),
        req(2.0, 1.0, 64 * 6, 10, [21, 22, 23, 24, 50, 51]),      # extends the worker; waits on turn 0
        req(6.0, 1.0, 64 * 5, 30, [21, 22, 23, 24, 25]),
    ]),
    trace("quick", [
        req(0.0, 0.5, 64 * 2, 5, [31, 32]),
        req(1.0, 0.5, 64 * 3, 5, [31, 32, 33]),
        req(2.0, 0.5, 64 * 4, 5, [31, 32, 33, 34]),
    ]),
    trace("background", [
        req(0.0, 1.0, 64 * 3, 20, [41, 42, 43]),
        sub(1.2, "a", 3000.0, [req(1.2, 1.0, 64 * 2, 20, [60, 61]), req(3.0, 1.0, 64 * 3, 20, [60, 61, 62])]),
        sub(1.4, "b", 2600.0, [req(1.4, 2.0, 64 * 2, 20, [70, 71])]),
        req(5.0, 1.0, 64 * 4, 20, [41, 42, 43, 44]),
        sub(5.5, "c", 60000.0, [req(5.5, 3.0, 64 * 2, 20, [90, 91]),
                                req(5.6, 3.0, 64 * 2, 20, [90, 95]),      # forks inside the subagent
                                req(9.0, 3.0, 64 * 3, 20, [90, 91, 92])]),
        req(8.0, 1.0, 64 * 5, 20, [41, 42, 43, 44, 45]),
    ]),
]
for t in traces:
    print(json.dumps(t, separators=(",", ":")))
