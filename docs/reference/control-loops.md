# Control loops

This page is for agents and maintainers: it lists the control loops BLIS runs beside the router and
where each is defined. The design rule is SIM-14 in the target design.

A control loop acts on the cluster while requests are in flight, as an autoscaler adds and removes
instances. Each loop is a component: it reads only values the cluster publishes to it, copied at the
time it runs, and never the instances themselves. Each is listed below with the upstream controller
it models and how far its behaviour has been checked (design section 8.9). A loop that is verified
against written rules and upstream code, but compared with no measured cluster run, is *verified, not
validated*.

<!-- BEGIN GENERATED from sim/cluster/control_loops.go -->
| Control loop | Configured by | Components | Reads | Models | Validation status | Evidence |
|---|---|---|---|---|---|---|
| `model-autoscaler` | `PolicyBundle.Autoscaler` | Collector, Analyzer, Engine and Actuator, run by the scaling tick (autoscaler.go) | per-instance status copied into sim.ClusterState at each tick (queue depth, batch size, KV use, in-flight count), not the instances themselves | the llm-d workload-variant-autoscaler (its saturation model); its README (https://github.com/llm-d/llm-d-workload-variant-autoscaler#readme, read for this list in November 2026) says it is deprecated in favour of a target-value rule on a router metric | verified, not validated | the pipeline is tested against its written contracts (autoscaler_test.go, autoscaler_invariant_test.go); no measured autoscaling run has been compared with it, and the upstream it follows is deprecated |
<!-- END GENERATED -->

The page is generated: `BLIS_UPDATE_CONTROL_LOOPS=1 go test ./sim/cluster -run TestControlLoopsPage`
rewrites the table from the list in `sim/cluster/control_loops.go`, and the test fails when the page
and the list disagree. `TestControlLoops_EveryConfigurablePolicyIsListedOrDeclaredNotALoop` fails when
a field is added to the policy bundle (`sim/bundle.go`) without being listed or declared not to be a
loop.

LoRA adapter placement is not a control loop in this repository: adapter residency is decided
inside each engine, and routing to a holder is the llm-d router's `lora-affinity-scorer`, which reads
the adapter sets the engines publish. A placement loop that acts beside the router would be added to
the list when it exists.
