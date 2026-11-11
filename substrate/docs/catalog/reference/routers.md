# Router table

[[catalog.count.routers]] router modes BLIS does not host, each mapped to the policy of llm-d's router that stands closest to it, in [`routers/`](https://github.com/inference-sim/blis-substrate/tree/[[catalog.ref]]/catalog/routers).

BLIS hosts one router, llm-d's, and never simulates another (X4). A benchmark run measured behind another router, such as Dynamo's, is scored by simulating the llm-d policy this table names, and the point says so: it names the router the run used, the policy BLIS used and the evidence below, and it is never graded G1 and never presented as if llm-d had routed the run (point-record kind `router_differs`). This is design §11.3 and §11.4. A router mode with no entry, or an entry whose policy is `none`, stays not scored (`router_unsupported`). An entry whose policy is `none` records that the mode was examined and no policy is close enough, and says why and what would change that.

An entry is added only with evidence: for each field it states, a citation of the router's own code or documentation at a pinned commit, and a citation of llm-d's router code that the policy rests on. A mapping is bound to the builds of the router the entry lists (`revisions`): a run on another build of the same router and mode has no mapping and stays `router_unsupported`. The table is keyed by the router, its mode and its session-affinity setting, not by the framework's label, because a router's behaviour is set by its mode. An entry's file is named for its key, `<router>-<mode>`, with `-affinity-<seconds>` when the setting is set, so one key cannot be stated twice. Design §11.4 keys it the same way: by the mode and session-affinity setting a run used.

[[catalog.routers.table]]

## Entries

[[catalog.routers.entries]]

## The policies llm-d's router offers

A policy is a composition of the router's scheduling plugins that it runs without code of BLIS's own. The list is closed: a policy joins it with the router code that defines it.

| Policy | Plugins | Defined in llm-d-router |
|---|---|---|
| `round-robin` | max-score-picker; no scorer; one counter per runtime, drawn at every pick of every profile | `pkg/epp/framework/plugins/scheduling/picker/maxscore/picker.go` and `picker/common.go` (`RotateScoredEndpoints`) |
| `least-active-requests` | inflight-load-producer; active-request-scorer; max-score-picker | `pkg/epp/framework/plugins/scheduling/scorer/activerequest/active_request.go` |
| `default` | queue-scorer (2); kv-cache-utilization-scorer (2); prefix-cache-scorer (3); max-score-picker | `pkg/epp/config/loader/defaults.go` (`loadDefaultConfig`) |
| `none` | no policy is close enough | |

The file format, and the rules a validator refuses an entry by, are in the [router table reference](../../contracts/reference/router-table.md).
