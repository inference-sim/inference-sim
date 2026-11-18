# The admission queue and its acceptance criteria

<p class="lede">A request waits in a queue before an engine admits it. What contends there is a property of the
engine's settings, so the engine states the queue's shape in its plan and the cost model prices a request's demand on
it. The simulator schedules requests on a generic multi-server queue and knows nothing of what the servers are.</p>

## The rule

- The engine states `plan.HostPlan.Admission`: `Servers` independent first-in first-out queues of `WorkersPerServer`
  servers each. Zero workers is no queue.
- The cost model answers `Kernel.AdmissionDemand(promptTokens)` with a `HostDemand`: `Service`, the time a request
  holds one server (the per-token term), and `Delay`, the time it only waits (the per-request term). They sum to
  `AdmissionOverhead`, which is unchanged. A request of no tokens demands nothing.
- The simulator reads the demand through `Environment.HostDemand`, as it reads a transfer's demand.

Before this change all of the admission overhead was a server's service time. A one-server queue then starts at most
1 / 40.4 ms = 24.8 requests a second, and a saturated closed loop sat at that rate.

## The assumption

The per-request term (`host_admission_per_request`, 40 ms, method `assumed`) is a latency of unknown origin. The
registry's rationale says tokenisation, templating and the message hop measured 0.0153 and 0.1083 ms in all. vLLM
v0.32.0 runs the tokenizer and the chat template on the renderer's thread pool, which shows that work uses a server. It
does not show that the 40 ms is a wait after the pool. The change charges it as `Delay`. If it is in fact server work,
the earlier model was right and the saturation it showed is real. If none of it is, the cap is the per-token term's. A
measurement of client time to first token at one and 1024 tokens at one client on the target part separates the two.

## Acceptance criteria

These were fixed before any held-out evaluation, from what the change does and not from development residuals. The
change removes a cap, so the first criterion is about the cap and the rest bound what must not move.

- (a) The cap is gone. For a 1P3D shape on 1k/1k at 1024 clients, the simulated closed-loop request rate,
  clients / (TTFT + (output tokens - 1) x TPOT), is at least 2 times the old cap of one API server,
  1 / (admission overhead) = 24.9 requests a second. Before the change the rate equalled the cap.
- (b) The knee error shrinks. On 1k/1k points with more than 128 clients, TTFT MAPE falls by at least 50 points and the
  median signed error moves toward zero. The development results (284% to 39%) had been seen when this was written; 50
  points is a deliberately loose floor, far below what the development points showed, and not a prediction.
- (c) Decode is untouched. TPOT MAPE over the run does not worsen by more than 1 point.
- (d) Light load is untouched. A one-client point is identical. A point at 8 clients or fewer changes its mean TTFT by
  no more than (clients - 1) / 2 x 40.4 ms in magnitude, the queueing a burst of clients used to do at the thread.

All four must hold. If one does not, the change is wrong or incomplete, and it is revised, or the delay assumption is
rejected and admission stays server time, before it is merged.

## The held-out evaluation

- Unit: the withheld test-split points of P/D run B, which the accuracy dashboard's harness holds.
- Who and when: the harness's gate job, once, after the consumers are pinned to their final heads. Nobody reads,
  scores or prints a test point by hand, and nothing is tuned after the verdict.
- What is reported: for each of (a) to (d), the aggregate figure the criterion names and whether it holds. No
  per-point value.
- Where it is recorded: one row of the dashboard's evaluation log, which makes a repeated evaluation visible.

## What the development points can and cannot show

The 60 train and validation points of run B are in-sample. They bound the assumption (all demand on the server, 78.1%
TTFT MAPE; split as here, 24.7%; no queue, 24.7%) but cannot tell the split from no contention: at those loads the
rate stays far below the per-token cap, about 2700 requests a second, so the queue never binds.
