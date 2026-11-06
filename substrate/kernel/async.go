package latencykernel

import (
	"fmt"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"

	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// resolveAsyncScheduling settles whether the engine schedules asynchronously, as vLLM
// v0.31.0 does (vllm/config/vllm.py:1544-1635).
//
// A stated value stands, and a stated true the engine would refuse is refused here: with a
// draft method outside the async-compatible set vLLM raises "Currently, async scheduling is
// only supported with EAGLE/MTP/Draft Model/NGram GPU/DSpark/DFlash kind of speculative
// decoding" (:1553-1567). Left to the engine, it turns on unless an incompatible option is
// present (:1576-1635). Of those conditions the kernel can see one, the draft method; the
// others cannot occur in a priced deployment: pooling runners are not priced, pipeline
// parallelism turns it off only under the V1 model runner (vllm/config/vllm.py:1634-1641@
// v0.32.0) and CUDA defaults to the V2 one, which the kernel takes (maxInFlightTokens records
// the V1 case as a coverage limit), disable_padded_drafter_batch has no deployment field,
// the DeepEP DBO case is ROCm-only, and CUDA's default executor on one node or many is
// multiprocessing (vllm/config/parallel.py:990-1030), which supports async scheduling
// (vllm/v1/executor/multiproc_executor.py:560-561).
func (k *Kernel) resolveAsyncScheduling(r resolve.EngineRules) error {
	method := ""
	if s := k.engine.Speculative; s != nil && s.NumSpecTokens > 0 {
		method = s.Method
	}
	var compatible, known bool
	if method != "" {
		compatible, known = r.AsyncCompatibleSpeculative(method)
	}
	knownIncompatible := method != "" && known && !compatible
	if method != "" && !known {
		k.assume("async_scheduling_spec", method,
			"the rules value carries no async-compatible draft list, so the draft method is "+
				"taken as compatible with async scheduling")
	}
	if a := k.engine.AsyncScheduling; a != nil {
		if *a && knownIncompatible {
			return fmt.Errorf("async_scheduling is requested with draft method %q, which "+
				"vLLM v0.31.0 refuses: async scheduling is only supported with "+
				"EAGLE/MTP/Draft Model/NGram GPU/DSpark/DFlash speculative decoding "+
				"(vllm/config/vllm.py:1553-1567)", method)
		}
		k.asyncScheduling = *a
		return nil
	}
	k.asyncScheduling = !knownIncompatible
	return nil
}

// schedulerHost is the engine-core CPU work one step carries for its requests --
// Scheduler.schedule(), the SchedulerOutput send and update_from_output, which loop over
// the scheduled requests -- and how much of it the step exposes.
//
// Without async scheduling EngineCore.step runs schedule, execute and update_from_output
// in series (vllm/v1/engine/core.py:630-668), so all of it adds to the step. With it,
// step_with_batch_queue schedules the next batch and issues it with non_block=True while
// the current one runs on the device (core.py:670-760), so the host work hides behind the
// device and only the part longer than the step shows: max(host, device). The one-step
// admission and release lag async scheduling also brings is the simulator's to price; the
// kernel reports the resolved mode (Resolved().AsyncScheduling) for it.
func (k *Kernel) schedulerHost(b kernel.Batch, device time.Duration) time.Duration {
	host := time.Duration(float64(k.schedPerRequest) * float64(len(b.Reqs)))
	if !k.asyncScheduling {
		return host
	}
	if host > device {
		return host - device
	}
	return 0
}
