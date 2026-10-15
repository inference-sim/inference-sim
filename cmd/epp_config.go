package cmd

import (
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
)

// --epp-config (#1930): an llm-d EndpointPickerConfig, ingested verbatim. It owns
// routing and flow control, so the flags it replaces are refused alongside it.
var (
	eppConfigPath string
	eppVersionArg string
	eppCPUTierArg string
	eppIngest     *sim.EPPIngest // nil unless --epp-config is set
)

// eppConflictingFlags are the routing and flow-control flags an EndpointPickerConfig
// replaces; setting any of them with --epp-config is refused rather than
// silently overridden in either direction.
var eppConflictingFlags = []string{
	"routing-policy", "routing-scorers", "lora-scorer-weight",
	"flow-control", "saturation-detector", "dispatch-order", "slo-targets",
	"max-gateway-queue-depth", "queue-depth-threshold", "kv-cache-util-threshold",
	"max-concurrency", "per-band-capacity", "usage-limit-threshold", "fairness-policy",
	"request-ttl", "queue-shedding", "in-flight-eviction",
	"prefill-routing-scorers", "decode-routing-scorers", "pd-decider", "pd-prefix-threshold",
}

func registerEPPConfigFlag(cmd *cobra.Command) {
	cmd.Flags().StringVar(&eppConfigPath, "epp-config", "", "llm-d EndpointPickerConfig YAML, ingested verbatim: plugins, schedulingProfiles (single or disagg-profile-handler P/D) and flowControl replace --routing-policy/--routing-scorers, the P/D routing flags and the flow-control flags. Unsupported scheduling plugins are refused by name; infrastructure-only plugins are accepted as inert.")
	cmd.Flags().StringVar(&eppCPUTierArg, "epp-cpu-tier-view", "", "BLIS opt-in (deviation from llm-d): name an approx-prefix-cache-producer of --epp-config that reports each instance's CPU-offload-tier-resident prefix instead of llm-d's router-side LRU estimate (#1912). Recorded in the epp block's deviations.")
	cmd.Flags().StringVar(&eppVersionArg, "epp-version", string(sim.EPPVersionMain), "llm-d-router release whose plugin semantics --epp-config follows: "+string(sim.EPPVersionMain)+" (latest main; alias main) or "+string(sim.EPPVersionV0_10_0)+" (the EPP image our calibration deployments run). See docs/guide/routing.md for the deltas.")
}

// applyEPPConfig loads --epp-config and writes the routing and flow-control
// package-level settings it determines. Precondition: the scenario is adopted
// (blockSizeTokens and the pool counts are known).
func applyEPPConfig(cmd *cobra.Command, bundle *sim.PolicyBundle) {
	eppIngest = nil
	if eppConfigPath == "" {
		return
	}
	var conflicts []string
	for _, name := range eppConflictingFlags {
		if f := cmd.Flags().Lookup(name); f != nil && f.Changed {
			conflicts = append(conflicts, "--"+name)
		}
	}
	if len(conflicts) > 0 {
		logrus.Fatalf("--epp-config owns routing and flow control; remove %s", strings.Join(conflicts, ", "))
	}
	if bundle != nil && (bundle.Routing.Policy != "" || len(bundle.Routing.Scorers) > 0) {
		logrus.Fatalf("--epp-config owns routing; remove the routing section from --policy-config")
	}
	version, err := sim.ParseEPPVersion(eppVersionArg)
	if err != nil {
		logrus.Fatalf("--epp-version: %v", err)
	}
	var opts []sim.EPPOption
	if eppCPUTierArg != "" {
		opts = append(opts, sim.WithCPUTierView(eppCPUTierArg))
	}
	ingest, err := sim.LoadEndpointPickerConfig(eppConfigPath, int(blockSizeTokens), version, opts...)
	if err != nil {
		logrus.Fatalf("--epp-config %s (llm-d %s): %v", eppConfigPath, version, err)
	}
	pools := prefillInstances > 0 || decodeInstances > 0 || prefillDecodeInstances > 0
	switch {
	case encodeInstances > 0:
		logrus.Fatalf("--epp-config with an encode pool is not supported (E/P/D is not part of the EPP ingestion)")
	case ingest.Disaggregated && !pools:
		logrus.Fatalf("--epp-config %s uses a disagg-profile-handler but the scenario defines no prefill/decode pools", eppConfigPath)
	case !ingest.Disaggregated && pools:
		logrus.Fatalf("--epp-config %s is a single-profile config but the scenario defines prefill/decode pools; use a disagg-profile-handler config", eppConfigPath)
	}
	if iv := ingest.MetricsRefreshIntervalUs; iv > 0 {
		if f := cmd.Flags().Lookup("snapshot-refresh-interval"); f != nil && f.Changed && snapshotRefreshInterval != iv {
			logrus.Fatalf("--snapshot-refresh-interval %d conflicts with the config's metrics-data-source interval (%dµs)", snapshotRefreshInterval, iv)
		}
		snapshotRefreshInterval = iv
	}
	eppIngest = ingest
	routingPolicy = "weighted"
	if ingest.Disaggregated {
		// prefix-based-pd-decider is built from PDPrefixBased; the named decider is
		// the fallback the cluster never consults when PDPrefixBased is set.
		pdDecider = ingest.PDDecider
		if pdDecider == "prefix-based" {
			pdDecider = "never"
		}
	}
	if fc := ingest.FlowControl; fc != nil {
		flowControlEnabled = true
		flowControlDetector = "concurrency"
		flowControlMaxConcurrency = int(fc.Detector.MaxConcurrency)
		flowControlMaxQueueDepth = fc.MaxRequests
		flowControlPerBandCapacity = fc.PerBandMaxRequests
		flowControlRequestTTL = fc.RequestTTLUs
	}
	for _, line := range ingest.Provenance.Inert {
		logrus.Infof("[epp] inert: %s", line)
	}
	for _, line := range ingest.Provenance.Notes {
		logrus.Infof("[epp] note: %s", line)
	}
	logrus.Infof("[epp] ingested %s (sha256 %s, llm-d-router %s): handler=%s, filters=%v, pd-decider=%q",
		eppConfigPath, ingest.Provenance.ConfigSHA256, version, ingest.Provenance.ProfileHandler, ingest.Provenance.Profile.Filters, ingest.Provenance.PDDecider)
}

// eppProvenance returns the `epp` output block, or nil without --epp-config.
func eppProvenance() *sim.EPPProvenance {
	if eppIngest == nil {
		return nil
	}
	p := eppIngest.Provenance
	return &p
}
