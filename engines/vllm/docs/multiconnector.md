# The MultiConnector model

vLLM's `MultiConnector` puts several KV connectors behind the one connector the scheduler calls.
A deployment that wants a prefill/decode transfer and a CPU tier in the same engine uses it:
NIXL moves KV between engines, the offloading connector keeps recently used KV in host memory,
and `MultiConnector` makes them look like one connector to the scheduler.

This page covers the scheduler side that `core/connector/multi` models: the choice rule, what
each child is told, how the model is tested, and what it refuses. The worker side, which moves
the bytes, is a transport in the simulator and in the oracle. Source:
`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py@5df63a72` (vLLM v0.32.0).

## Configuration

An engine block names the wrapper and lists its children, first to last:

```yaml
engine:
  kv_transfer:
    connector: MultiConnector
    kv_role: kv_both             # the wrapper's role: kv_producer, kv_consumer or kv_both
    connectors: [NixlConnector, OffloadingConnector]
  kv_offload:                    # the offloading child's CPU tier, as without the wrapper
    connector: OffloadingConnector
    cpu_bytes: 17179869184
    bytes_per_block: 1048576
```

The order matters: the first child that can serve a load serves it (next section). The NIXL
child takes `kv_transfer.kv_role`. The offloading child is `kv_both`. vLLM takes each child's
settings from the child's own entry in `kv_connector_extra_config.connectors`; here the offloading
child's settings are the `kv_offload` block's, so they are written once.

## What the wrapper does

Every hook reaches every child, in the listed order. The wrapper makes one decision of its own.

**Choice.** `get_num_new_matched_tokens` (`multi_connector.py:434-456`) asks each child how many
tokens beyond the local cache hit it can supply. The first child that answers more than zero is
chosen for that request, and the wrapper remembers the choice. The later children are still
asked, so each one's state for the request stays current, but their answers are not used. If any
child answers that it is still looking up its hit, the whole request waits for the next step; the
scheduler asks again.

**Allocation.** After the scheduler allocates blocks, `update_state_after_alloc` (:467-477) gives
the chosen child the number of external tokens and gives every other child the same blocks with
zero external tokens. A request no child matched has no chosen child, and every child sees zero.

**Saving.** `build_connector_meta` (:483-494) asks every child for its metadata from the step's
scheduler output, and the worker returns each child its own slice of the worker metadata
(`update_connector_output`, :496-517). So the offloading child stores prompt chunks of every
request that passes through, whichever child loaded it. Nothing is routed to one child.

**Finishing.** `request_finished` and `request_finished_all_groups` (:573-616) call every child and
hold the request's blocks if any child holds them. A prefiller's request (`do_remote_decode`) is
held by NIXL until the decoder has pulled it; the offloading child never holds blocks, so its
answer is "free". The wrapper forgets the choice when the request finishes.

**Capabilities.** `requires_kv_delivery` and `has_pending_push_work` are true if any child's is.
The hybrid KV cache manager stays on only when every child supports it. Async scheduling,
pipeline parallelism and drafters are accepted when both children are tested beside them.

### A worked example (token counts are illustrative)

Take `connectors: [NixlConnector, OffloadingConnector]`.

| Request | NIXL answers | Offloading answers | Chosen | Loaded |
|---|---|---|---|---|
| decode side, `do_remote_prefill`, prompt not in the CPU tier | 32 | 0 | NIXL (child 0) | 32 tokens pulled |
| local request whose prompt prefix is in the CPU tier | 0 | 16 | offloading (child 1) | 16 tokens from the CPU tier |
| decode side whose prefix is also in the CPU tier | 32 | 16 | NIXL (child 0) | 32 tokens pulled; offloading is told 0 external tokens |
| local request, nothing cached | 0 | 0 | none | nothing |

With `connectors: [OffloadingConnector, NixlConnector]` the third row flips: offloading answers
first, wins, and loads 16 tokens; NIXL is told zero external tokens and loads nothing.

On a prefiller (`do_remote_decode`), NIXL answers 0 for the load, so a prefix in the CPU tier is
loaded by the offloading child; once the request finishes, NIXL holds its blocks until the send is
reported, and `delay_freed` lists the request until then.

## How it is tested

Three layers, each against something other than the code under test.

1. **Behaviour with stub children** (`core/connector/multi/multi_test.go`). Children that record
   the calls they receive and answer from a script. Each rule above has a test, and each test was
   run against a deliberately broken wrapper (choosing the last matching child, stopping at the
   first match, giving every child the external tokens, short-circuiting the finish calls, and so
   on) and fails on it. Two more tests run the wrapper over the real NIXL and offloading
   schedulers on a `sched.Scheduler`: a prefiller loaded from the CPU tier and then held by NIXL,
   and a decoder loaded by NIXL when NIXL is listed first.
2. **Differential test against vLLM** (`core/connector/multi/diff_test.go`). The oracle runner,
   `oracle/run_multiconnector.py`, builds vLLM's real `MultiConnector` over its real
   `NixlConnector` and `OffloadingConnector` schedulers on vLLM's real `Scheduler`, mocks only the
   GPU, the clock and the transports, and records a trace for each scenario in
   `oracle/scenarios/multiconnector/`. A step of the trace is the offloading step record (jobs,
   flushes, completions, the CPU tier), the transport NIXL sees (received, sent, failed), the
   scheduler's state, and `chosen`: the child chosen for each request the wrapper still knows.
   `TestMultiConnectorAgainstVLLM` replays each scenario through `core/sched` with the Go wrapper
   and compares every step with the trace in `core/connector/multi/testdata/golden/`. A trace is
   refused unless its provenance names this scenario's hash, the pinned vLLM commit and the
   current mock list (`oracle/mocks.json`).
3. **Seeded fuzz.** The oracle's generator writes random scenarios and runs vLLM on each;
   `TestMultiConnectorFuzzCorpusAgainstVLLM` replays a directory of them and requires zero
   mismatches. Set `BLIS_VLLM_MULTI_FUZZ_DIR` to the directory; unset, the test is skipped and
   says so. `oracle/README.md` lists the generator's seed ranges and the commands CI runs.

To run the first and second layers locally: `go test ./core/connector/multi/`. The third needs a
corpus from the oracle (`oracle/README.md`, Linux x86_64 with the locked Python environment).

The replay driver was also checked on its own: running every committed NIXL and offloading
scenario through the wrapper with that connector as its only child reproduces the connector's own
golden trace, the batch-queue scenarios included.

## What the engine adds

The engine (`core/vllm/multiconnector.go`) builds both children, installs the wrapper, and keeps
one sequence of transfer IDs for the NIXL side and the offload side, so `TransferDone` reaches the
side that started the transfer. A decode leg whose KV the offloading child loads (the offloading
child listed first and holding a hit) waits for that CPU load. NIXL, given no external tokens,
only tells the prefiller to release its blocks (`update_state_after_alloc` with no external tokens,
`nixl/pull_scheduler.py:157-168`): the engine reports it as a pull of no tokens, which the cost
model prices at zero bytes, so that the simulator can end the prefiller's handoff. Its end lands no
KV and reports no `finished_recving`; the offload's load does. Every transfer the engine starts
validates and names a device of the pool's plan, which `TestMultiConnectorTransfersPriceAgainstThePlan`
holds.

## What the oracle leaves out

The fuzz generator never puts two client aborts in one step. vLLM's `finish_requests` builds a set
of the ids and frees the requests in its iteration order (`scheduler.py:2662-2668`), which depends
on the interpreter's hash seed; the free order decides which block a later request gets, so such a
scenario has no single answer from vLLM and none is generated.

## What it refuses

- Any child other than NIXL and the offloading connector: Mooncake, MoRIIO, NIXL push,
  `SimpleCPUOffloadConnector`, LMCache. Their scheduler sides are not modelled.
- A child list other than the NIXL and offloading pair, in either order.
- Two children that both hold a finished request's blocks. vLLM then makes the worker report the
  request's `finished_sending` once per holder (`_extra_async_saves`, :542-571); that is not
  modelled. With this pair only NIXL holds blocks, so the case does not arise.
- Data parallelism, as the NIXL and offloading connectors are each refused there.
- An MTP drafter beside the offloading child, as the offloading connector is refused there.

The registry entry (`registry/connectors.yaml`) serves `MultiConnector` in the schema's
`composite` role only. The wrapper moves no KV itself, so it is neither a `pd_transfer` nor an
`offload` connector.
