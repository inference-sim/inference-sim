package cmd

import (
	"github.com/sirupsen/logrus"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/cluster"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// The P/D pools are opened, checked and priced by the engine through the build site
// (engine_build.go); this file holds each pool's engine settings as the cluster takes them.

// overrides is each role's engine, from its own pool's kernel.
func (p *kernelPools) overrides() (prefill, decode cluster.PoolOverrides) {
	return poolOverridesOf(p.prefill), poolOverridesOf(p.decode)
}

func poolOverridesOf(m *kernelmodel.Model) cluster.PoolOverrides {
	s, err := m.SettingsReserving(loraReservedBytesForKV)
	if err != nil {
		logrus.Fatalf("%s pool: %v", m.Role(), err)
	}
	requireWindowFits(string(m.Role())+" pool", s)
	d := m.Deployment()
	tp := d.TP
	blocks := s.KVBlocks
	seqs, toks := int64(s.MaxNumSeqs), int64(s.MaxNumBatchedTokens)
	noCache := s.PrefixCachingDisabled
	ml := int64(s.MaxModelLen)
	state := s.SequenceState
	hit := prefixHitOf(string(m.Role())+" pool", m)
	logSequenceFixedCharge(string(m.Role())+" pool", s)
	o := cluster.PoolOverrides{
		TP: &tp, GPU: d.Hardware, LatencyModel: m, TotalKVBlocks: &blocks, MaxModelLen: &ml,
		MaxNumSeqs: &seqs, MaxNumBatchedTokens: &toks, PrefixCachingDisabled: &noCache,
		SequenceState: &state, PrefixHit: &hit,
	}
	poolEngineKnobs(m, &o)
	// Each pool runs its own engine's scheduling policy, unless a flag or policy bundle chose
	// one for the run, which then applies to every pool and is recorded where it departs.
	stated := enginePolicyOf(m)
	if schedulerSource != "" {
		recordPolicyOverride("scheduler", string(m.Role()), stated.scheduler, scheduler, schedulerSource)
	} else {
		o.Scheduler = stated.scheduler
	}
	if preemptionSource != "" {
		recordPolicyOverride("preemption_policy", string(m.Role()), stated.preemption, preemptionPolicy, preemptionSource)
	} else {
		o.PreemptionPolicy = stated.preemption
	}
	return o
}

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
