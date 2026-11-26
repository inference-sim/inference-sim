package cmd

import (
	"fmt"
	"slices"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	specmodel "github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// The catalog's per-entity projections. `catalog list` and `catalog resolve` report the same
// entities, so each entity is projected once and both commands read that projection. Two
// hand-maintained copies drift: they already disagreed on how a chip's unstated packaging was
// encoded (a null on one, a zero on the other), and a client trusts whichever it read.

// chipFacts is one chip's datasheet figures, packaging and provenance, as both commands report
// them. Every number is nominal: an achievable rate is a coefficient, not a chip fact, so nothing
// here is derated.
type chipFacts struct {
	Provenance string `json:"provenance,omitempty"`
	// FieldProvenance names the fields whose source differs from Provenance, keyed by this
	// record's own field names ("memory_gib" for the catalog's MemoryGiB).
	FieldProvenance map[string]string `json:"field_provenance,omitempty"`

	BF16TFlops  float64 `json:"bf16_tflops"`
	FP8TFlops   float64 `json:"fp8_tflops"`
	NVFP4TFlops float64 `json:"nvfp4_tflops,omitempty"`
	// TDPWatts is the board power the vendor specifies for the part as shipped, watts per GPU,
	// omitted where the entry states none. It is a packaging maximum a client prices electricity
	// with, not a measured draw, so a client must read the omission as unavailable rather than as
	// a zero-watt part and must not substitute another configuration's figure.
	TDPWatts float64 `json:"tdp_watts,omitempty"`
	// MemoryGiB is the device's memory in GiB, the unit the catalog states it in; a client showing
	// bytes multiplies by 2^30.
	MemoryGiB          float64 `json:"memory_gib"`
	MemoryBandwidthTBs float64 `json:"memory_bandwidth_tbs"`
	// IntraNodeBwGBps is per-GPU unidirectional on-node bandwidth; IntraRackBwGBps is the
	// within-rack tier where it differs from the inter-rack fabric, zero where the part has none.
	IntraNodeBwGBps float64 `json:"intra_node_bw_gbps"`
	IntraRackBwGBps float64 `json:"intra_rack_bw_gbps,omitempty"`
	SMCount         int     `json:"sm_count"`
	// ComputeCapability is the CUDA capability, "major.minor", and HostCPU the architecture of the
	// CPU that drives the GPUs; each empty where the entry states none.
	ComputeCapability string `json:"compute_capability,omitempty"`
	HostCPU           string `json:"host_cpu,omitempty"`
	// Packaging is the GPUs per node and per rack the product fixes. Both are null where the entry
	// leaves the count to the scenario, which is a different fact from zero: one command reporting
	// the two as an absent field and the other as 0 is what made the shared projection necessary.
	Packaging chipPackaging `json:"packaging"`
}

// chipFactsOf is one chip's facts. The provenance map is rekeyed to the record's field names, so a
// client reads which of the numbers beside it came from another source than the file's own.
func chipFactsOf(c *hardware.Chip) chipFacts {
	out := chipFacts{
		Provenance:         string(c.Provenance),
		BF16TFlops:         c.BF16Peak,
		FP8TFlops:          c.FP8Peak,
		NVFP4TFlops:        c.NVFP4Peak,
		TDPWatts:           c.TDPWatts,
		MemoryGiB:          c.MemoryGiB,
		MemoryBandwidthTBs: c.MemoryBandwidthTBs,
		IntraNodeBwGBps:    c.IntraNodeBwGBps,
		IntraRackBwGBps:    c.IntraRackBwGBps,
		SMCount:            c.SMCount,
		ComputeCapability:  c.ComputeCapability,
		HostCPU:            c.HostCPU,
		Packaging:          packagingOf(c),
	}
	if len(c.FieldProvenance) > 0 {
		out.FieldProvenance = make(map[string]string, len(c.FieldProvenance))
		for k, v := range c.FieldProvenance {
			out.FieldProvenance[chipFieldName(k)] = v
		}
	}
	return out
}

// chipPackaging is what a chip's catalog entry states about its packaging: the GPUs per node and
// per rack the product fixes, each null where the entry leaves it to the scenario.
type chipPackaging struct {
	GPUsPerNode *int `json:"gpus_per_node"`
	GPUsPerRack *int `json:"gpus_per_rack"`
}

// packagingOf is a chip's packaging, with an unstated count left null rather than encoded as zero:
// the scenario fills a null, and a zero would claim the product fixes none.
func packagingOf(c *hardware.Chip) chipPackaging {
	var p chipPackaging
	if c.GPUsPerNode > 0 {
		n := c.GPUsPerNode
		p.GPUsPerNode = &n
	}
	if c.GPUsPerRack > 0 {
		n := c.GPUsPerRack
		p.GPUsPerRack = &n
	}
	return p
}

// chipFieldName maps a catalog chip field's wire name to the name this record reports it under.
// A field this record does not report keeps its own name, so an unmapped entry is still readable.
func chipFieldName(field string) string {
	if name, ok := chipFieldNames[field]; ok {
		return name
	}
	return field
}

var chipFieldNames = map[string]string{
	"TFlopsPeak":        "bf16_tflops",
	"TFlopsFP8":         "fp8_tflops",
	"TFlopsNVFP4":       "nvfp4_tflops",
	"TDPWatts":          "tdp_watts",
	"MemoryGiB":         "memory_gib",
	"BwPeakTBs":         "memory_bandwidth_tbs",
	"IntraNodeBwGBps":   "intra_node_bw_gbps",
	"IntraRackBwGBps":   "intra_rack_bw_gbps",
	"SMCount":           "sm_count",
	"ComputeCapability": "compute_capability",
	"HostCPU":           "host_cpu",
	"GPUsPerNode":       "gpus_per_node",
	"GPUsPerRack":       "gpus_per_rack",
}

// drafterInfo is one entry of a target's drafters: the word "speculator" for its own in-graph
// speculator, or the catalog name of a separate drafter checkpoint.
//
// Method and NumSpecTokens are the speculator the drafter's graph states (the target's own graph
// for the in-graph speculator) and the draft length its checkpoint defaults to, which is not a
// limit (the substrate model.Speculator.NumSpec). Both are omitted when the catalog ships no graph
// for the name: a separate drafter checkpoint need not have one, and an empty method or a zero
// length would read as a stated value.
type drafterInfo struct {
	Name          string  `json:"name"`
	Method        *string `json:"method,omitempty"`
	NumSpecTokens *int    `json:"num_spec_tokens,omitempty"`
}

// draftersOf is what a model's entry lists as able to draft for it, in the entry's order, with the
// facts the drafter's graph states. It returns nothing, not an error, for a model that lists none.
func draftersOf(data *load.Data, target string) ([]drafterInfo, error) {
	id, err := data.ModelIdentity(target)
	if err != nil {
		return nil, err
	}
	if len(id.Drafters) == 0 {
		return nil, nil
	}
	var out []drafterInfo
	for _, d := range id.Drafters {
		graphOf := d
		if d == specmodel.InGraphSpeculator {
			graphOf = target
		}
		info := drafterInfo{Name: d}
		// A drafter checkpoint with no graph is the documented absence: data.Model refuses a
		// name the catalog has no entry for, and the entry list is the same map that refusal
		// consults, so absence is decided here and a read that still fails is a caller's error
		// to see rather than a fact silently dropped.
		if slices.Contains(data.Models(), graphOf) {
			g, err := data.Model(graphOf)
			if err != nil {
				return nil, fmt.Errorf("%s's drafter %s: %w", target, graphOf, err)
			}
			if g.Speculator != nil {
				m, n := g.Speculator.Method, g.Speculator.NumSpec
				info.Method, info.NumSpecTokens = &m, &n
			}
		}
		out = append(out, info)
	}
	return out, nil
}
