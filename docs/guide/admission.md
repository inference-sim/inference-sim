# Admission Control

Admission control is the first gate in the cluster pipeline. It decides whether to accept or reject incoming requests before they reach the routing stage. Admission only applies in cluster mode (`--num-instances` > 1) -- single-instance simulations skip directly to the wait queue.

```bash
# Rate-limit a 4-instance cluster with token bucket admission
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 500 --num-requests 2000 \
  --admission-policy token-bucket \
  --token-bucket-capacity 10000 --token-bucket-refill-rate 1000
```

## Available Policies

| Policy | Flag Value | Behavior |
|--------|-----------|----------|
| **Always-admit** | `--admission-policy always-admit` (default) | Accepts all requests unconditionally. No filtering. |
| **Token-bucket** | `--admission-policy token-bucket` | Rate-limiting. Each request consumes tokens equal to its input token count. Tokens refill at a constant rate. Rejects when the bucket is empty. |
| **Reject-all** | `--admission-policy reject-all` | Rejects all requests unconditionally. Pathological template for testing. |

## Token Bucket Mechanics

The token bucket policy controls throughput by treating each request's input token count as a cost:

| Flag | Description | Default |
|------|-------------|---------|
| `--token-bucket-capacity` | Maximum number of tokens the bucket can hold | 10000 |
| `--token-bucket-refill-rate` | Tokens added per second of simulation time | 1000 |

How it works:

1. **Bucket starts full.** At initialization, the bucket holds `capacity` tokens.
2. **Refill is continuous.** On each admission decision, the bucket refills proportionally to elapsed simulation time: `refill = elapsed_microseconds * refill_rate / 1,000,000`.
3. **Cost per request = number of input tokens.** A request with 512 input tokens costs 512 tokens from the bucket.
4. **Admission check.** If `current_tokens >= cost`, the request is admitted and the cost is subtracted. Otherwise the request is rejected with reason `"insufficient tokens"`.
5. **Capacity cap.** Tokens never accumulate beyond `capacity`, even after long idle periods.

!!! example "Sizing the bucket"
    With `--token-bucket-capacity 10000 --token-bucket-refill-rate 1000` and requests averaging 512 input tokens, the sustained admission rate is roughly `1000 / 512 ~ 1.95 req/s`. The bucket's capacity of 10000 tokens allows a burst of up to `10000 / 512 ~ 19` requests before rate-limiting kicks in.

Rejected requests are counted in the output anomaly counters (`Rejected Requests`) and in the full pipeline conservation formula (`num_requests == injected_requests + rejected_requests`), but they never enter the routing stage or any instance queue.

## When to Use Admission Control

- **Overload protection.** When the arrival rate significantly exceeds service capacity, unbounded queues grow without limit. Admission shedding keeps queue depth manageable.
- **Cost control.** Limit total token throughput to match a token budget or downstream rate limit.
- **Graceful degradation.** Shed excess load to protect SLO compliance for admitted requests. Under extreme overload, routing distributes load and scheduling orders within instances, but neither can reduce total queue depth -- admission is the lever that can.
- **Testing rejection paths.** The `reject-all` policy verifies that rejection counting, trace recording, and conservation invariants hold when no requests are admitted.

!!! tip "Admission is the third lever"
    Routing distributes load across instances. Scheduling orders requests within each instance. But when total arrival rate exceeds total service capacity, neither routing nor scheduling can reduce the queue -- they can only redistribute it. Admission control is the mechanism that actually reduces inbound volume.

## Pipeline Latency

The `--admission-latency` and `--routing-latency` flags model real network and processing overhead between gateway and backend (gRPC hops, service mesh serialization, queue dispatch). These are pipeline concerns that affect both admission and routing stages. See [Cluster Simulation](cluster.md#admission-and-routing-latency) for details on configuring pipeline latency.

## Further Reading

- [Cluster Simulation](cluster.md) -- full pipeline overview
- [Routing Policies](routing.md) -- the next stage after admission
- [Cluster Architecture](../concepts/architecture.md#admission-pipeline) -- architectural details
