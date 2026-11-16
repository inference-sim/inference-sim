package latencykernel

import (
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

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
