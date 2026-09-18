// Package kvtransfer models vLLM's KV-offload I/O thread pool as a bounded
// priority queueing station with N servers per storage tier.
//
// It fills hole H2 (`kv_transfer_station`) of the multi-tier KV-offload design
// (issue #1585, sub-issue #1588). The station is the mechanism by which KV
// blocks move between the CPU tier and a secondary tier (disk / object store):
// reads (secondary→CPU) and writes (CPU→secondary) are each served by a fixed
// set of I/O servers, exactly as vLLM's DualQueueThreadPool serves them with two
// queues and two thread groups (vllm/v1/kv_offload/tiering/fs/thread_pool.py).
//
// # The right shape: N servers, not divided bandwidth
//
// A disk or object store saturates on concurrency, not on divided bandwidth.
// This station therefore models a bounded queue with N = NRead+NWrite servers
// per tier and priority classes — NOT the fair-share bandwidth split of
// sim/cluster's PDTransferContention. The two models predict opposite signs for
// "add write threads, watch read latency" (see BC-S2), which is what makes the
// difference testable rather than cosmetic.
//
// # Priority with fallback (BC-S2)
//
// Read-priority servers drain the read queue first, then fall back to the write
// queue; write-priority servers do the reverse. A server idles only when BOTH
// queues are empty. This mirrors DualQueueThreadPool._worker (thread_pool.py:165)
// where each thread pops from its primary queue if non-empty, else the secondary.
//
// # Jobs, not blocks (BC-S5)
//
// One TransferJob is one unit of service covering many blocks, matching vLLM's
// enqueue_store(job_id, 1, [task]) (fs/manager.py:229): a single batched pool
// task carries the whole block set. Bytes is the total payload; the station
// never splits a job. Modeling one event per block would be both slower and less
// faithful.
//
// # Service time (BC-S3)
//
// service_time = base + bytes/bandwidth, configured per (tier, direction). Read
// and write bandwidth are independent because measured NVMe KV-offload work shows
// read/write asymmetry too large for a single number.
//
// # Determinism (BC-S4, BC-S6, INV-6)
//
// vLLM's pool is genuinely nondeterministic (real OS threads, GIL scheduling).
// BLIS must produce byte-identical output for a fixed seed. Both cannot hold
// literally, so this package matches the DISTRIBUTION while fixing the SEQUENCE:
// completion order is a total deterministic function of
// (completeAt, submitTick, tier, direction, jobID) — the same discipline the
// engine already uses for event ties (timestamp, priority, seqID). The model
// introduces NO randomness; it does not draw from any RNG. This is stronger than
// seeding the randomness would be, and it is why the package imports no sim/RNG.
//
// # Layering
//
// This is a leaf module: it depends only on the standard library. Nothing in the
// simulator calls it yet — H1 (kv_tier_chain, #1590) is the consumer. Until then
// the station has no effect on any simulation, so blis run/replay/observe output
// is byte-identical to a pre-feature build (INV-6) and run/replay parity (INV-13)
// is unaffected.
package kvtransfer
