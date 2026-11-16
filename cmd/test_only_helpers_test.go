package cmd

import (
	"github.com/inference-sim/blis-vllm/compat/build"
	"github.com/inference-sim/blis-vllm/compat/kernelmodel"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// Views of the build site's state that only this package's tests read. No binary calls them, so
// they live in a test file (design LIF-7).

// rankInPool is an instance's rank within its pool (build.RankInPool).
func rankInPool(id cluster.InstanceID, firstDecode int) int {
	return build.RankInPool(string(id), firstDecode)
}

// currentRoutingOverrides is the routing layer the pools will open with.
func currentRoutingOverrides() routingOverrides { return kernelmodel.CurrentRoutingOverrides() }

// batchConfigFromCLI is the run's batch configuration, the part of engineConfigFor's the
// prefix-caching toggle reaches (build.BatchConfigOf).
func batchConfigFromCLI() batchConfig {
	return build.BatchConfigOf(engineSettings(modelConfig{}, 1, kvOffloadConfig{}, loraConfig{}, speculativeConfig{}))
}
