package cmd

import "github.com/sirupsen/logrus"

// The P/D pools are opened, checked and priced by the engine through the build site
// (engine_build.go), which also builds each pool's engine settings as the cluster takes them
// (poolOverridesOf); this file re-sizes a kernel run's KV pool for the LoRA reservation.

// applyKernelLoRAReservation re-sizes a kernel run's KV pool with the static LoRA adapter
// reservation set aside, once the LoRA config is known (it is resolved after the deployment
// is adopted). vLLM takes the reservation beside the weights before profiling the KV pool, so
// the kernel's budget shrinks by it. No-op when no kernel is open or with no reservation.
func applyKernelLoRAReservation() {
	if kernelOpened == nil || loraReservedBytesForKV <= 0 {
		return
	}
	s, err := kernelOpened.SettingsReserving(loraReservedBytesForKV)
	if err != nil {
		logrus.Fatalf("scenario %q with a %d-byte LoRA adapter reservation: %v",
			kernelScenario, loraReservedBytesForKV, err)
	}
	requireWindowFits("pool with the LoRA adapter reservation set aside", s)
	totalKVBlocks = s.KVBlocks
}
