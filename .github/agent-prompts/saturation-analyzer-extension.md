# Saturation detector

The post-hoc detector (`sim/saturation`, `--detectors settle`) is the one steady-state detector
of design DEV-10. It replaced the roster of composite, threshold, backlog-drift and peak-rate
detectors (inference-sim#1616, #1617, #1618, #1693, #1927), which disagreed on the same run.

## What it decides

A window is `SETTLED` (no trend in the backlog), `OVERLOADED` (the backlog grows) or
`INCOMPLETE` (too few completions, or a latency tail so heavy that the window cannot settle).
Only a settled run gives a steady-state latency figure.

## The statistic

1. The requests in the system are swept from every arrival and departure. A request still in the
   system at the end of the run stays in it, so an overloaded run's backlog does not fall at its
   end.
2. The window is cut into 20 equal batches; each batch's time-average backlog is its batch mean.
3. The trend is the one-sided Mann-Kendall test on the batch means after trend-free
   pre-whitening of lag-1 autocorrelation (Yue and Pilon 2002), at alpha 0.01. The run is
   overloaded when the test rejects and the growth Sen's slope implies across the window is at
   least 2 requests and a quarter of the mean backlog.
4. Unless the backlog visibly grows (a server that completes few requests while its backlog
   grows is overloaded): fewer than 10 completions per batch is incomplete, and so is a
   latency tail whose Hill index is below 2.2 (largest tenth) or 2.0 (largest twentieth), which
   means an infinite variance.

A closed system holds its backlog at the concurrency, so it reads settled.

## What it promises

`sim/saturation/settle_test.go` simulates single-server queues whose behaviour is known exactly,
over 200 seeds per row, discarding the first 40% of arrivals as warm-up. "Reads as" is the
fraction of runs given the label in the last column. A failing row fails the test.

| System | Arrivals | Reads as | Fraction |
|---|---|---|---|
| M/M/1, load 0.5 | 2000 | SETTLED | 0.98 |
| M/M/1, load 0.7 | 4000 | SETTLED | 0.98 |
| M/M/1, load 0.9 | 8000 | SETTLED | 0.98 |
| M/M/1, load 0.97 | 8000 | SETTLED | 0.77 |
| M/M/1, load 1.0 | 8000 | OVERLOADED | 0.51 |
| M/M/1, load 1.1 | 8000 | OVERLOADED | 1.00 |
| M/M/1, load 1.2 | 4000 | OVERLOADED | 1.00 |
| M/M/1, load 1.5 | 1000 | OVERLOADED | 1.00 |
| M/G/1, Pareto 3.5 service, load 0.6 | 4000 | SETTLED | 0.98 |
| M/G/1, Pareto 1.5 service, load 0.6 | 4000 | INCOMPLETE | 0.82 |
| M/G/1, Pareto 1.5 service, load 0.6 | 16000 | INCOMPLETE | 0.93 |
| Closed queue, 8 clients | 4000 | SETTLED | 1.00 |

So the false-alarm rate (a settled system not read as settled) is about 2% at loads to 0.9. At
load 1 the verdict is decided by the horizon, as theory says: the backlog grows like the square
root of time, and the detector calls it overload about half the time at 8000 arrivals. A load
of 0.97 reads settled about three runs in four. A Pareto tail of index 1.5 is caught 82% of the
time at 4000 arrivals and 93% at 16000, because the tail index is estimated from the data.
