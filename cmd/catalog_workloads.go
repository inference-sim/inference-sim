package cmd

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	schemaworkload "github.com/inference-sim/blis-substrate/contracts/spec/workload"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/inference-sim/sim/workload"
)

const (
	// catalogWorkloadsSubdir is the catalog's workloads namespace; a named preset is
	// workloads/<name>.yaml.
	catalogWorkloadsSubdir = "workloads"
	// presetFileExt is the extension of a catalog preset definition.
	presetFileExt = ".yaml"
)

// presetWorkload is one named workload preset (chatbot, summarization, contentgen,
// multidoc), as the catalog's workloads/<name>.yaml states it.
//
// #1769: this is the ONLY declaration of the preset shape. It used to be duplicated — the
// `workloads:` block of the bundled defaults.yaml (the copy actually read) and the
// catalog's workloads/*.yaml (read by nothing) held identical values with nothing keeping
// them in sync, so the drifting copy was the one that moved output. The catalog is now the
// single source of truth, consistent with NS-6 (#1733): a run already requires the catalog
// for its model, and nothing about a run is inferred from a bundled default.
//
// The FILE format is not declared here. It is the contracts' workload.Shape, decoded and
// checked by blis-substrate's loader with the rest of the catalog. This struct is the value BLIS
// reads out of a Shape, so there is one parser per catalog format.
type presetWorkload struct {
	PrefixTokens      int
	PromptTokensMean  int
	PromptTokensStdev int
	PromptTokensMin   int
	PromptTokensMax   int
	OutputTokensMean  int
	OutputTokensStdev int
	OutputTokensMin   int
	OutputTokensMax   int

	// Acceptance is what the preset states about speculative acceptance -- its category,
	// thinking mode and own measured value (#1939) -- nil when it states none.
	Acceptance *acceptance.Workload
}

// presetFromShape reads the token distributions out of a catalog workload shape.
func presetFromShape(s *schemaworkload.Shape) presetWorkload {
	return presetWorkload{
		PrefixTokens:      s.PrefixTokens,
		PromptTokensMean:  s.Prompt.Mean,
		PromptTokensStdev: s.Prompt.StdDev,
		PromptTokensMin:   s.Prompt.Min,
		PromptTokensMax:   s.Prompt.Max,
		OutputTokensMean:  s.Output.Mean,
		OutputTokensStdev: s.Output.StdDev,
		OutputTokensMin:   s.Output.Min,
		OutputTokensMax:   s.Output.Max,
		Acceptance:        acceptanceTermsOf(s),
	}
}

// acceptanceTermsOf is a shape's acceptance terms, or nil when it states none.
func acceptanceTermsOf(s *schemaworkload.Shape) *acceptance.Workload {
	if t := s.AcceptanceTerms(); t.Category != "" || t.Thinking != "" || t.Value != nil {
		return &t
	}
	return nil
}

// toPresetConfig converts a catalog preset into the library's PresetConfig. It is the ONE
// construction site for workload.PresetConfig in cmd/ (R4): the preset consumers
// (`blis run --workload`, `blis convert preset`) each built it inline, so a new preset field
// had to be threaded through every copy — and a field missed in one of them would make the
// same preset mean different things per command.
func (w presetWorkload) toPresetConfig() workload.PresetConfig {
	return workload.PresetConfig{
		PrefixTokens:      w.PrefixTokens,
		PromptTokensMean:  w.PromptTokensMean,
		PromptTokensStdev: w.PromptTokensStdev,
		PromptTokensMin:   w.PromptTokensMin,
		PromptTokensMax:   w.PromptTokensMax,
		OutputTokensMean:  w.OutputTokensMean,
		OutputTokensStdev: w.OutputTokensStdev,
		OutputTokensMin:   w.OutputTokensMin,
		OutputTokensMax:   w.OutputTokensMax,
	}
}

// readCatalogPresetWorkload reads one named preset from the catalog in use. The catalog was
// read and checked when it was opened (load.Data), so a preset that does not decode or validate
// never reaches here; this reads its token distributions out and holds them to what BLIS's
// sampler needs.
//
// Every failure is a named refusal (R1): an absent preset names the catalog's source and the
// presets the catalog does have; a replay workload is refused as a preset. There is no fallback
// to a built-in preset, because a silently substituted token distribution is the drift #1769
// removed.
func readCatalogPresetWorkload(name string, data *load.Data) (*presetWorkload, error) {
	if name == "" {
		return nil, fmt.Errorf("no workload preset name was supplied")
	}
	preset, err := catalogPreset(data, name)
	if err != nil {
		return nil, fmt.Errorf("workload preset %q: %w", name, err)
	}
	if preset.Shape == nil {
		return nil, fmt.Errorf("workload %q in the catalog is a replay workload, not a preset; "+
			"replay it with blis replay --agentic-scenario", name)
	}
	wl := presetFromShape(preset.Shape)
	if err := presetBoundsError(name, wl); err != nil {
		return nil, err
	}
	return &wl, nil
}

// presetSource is the part of a catalog workload entry a preset run reads: its shape, nil for a
// replay workload.
type presetSource struct{ Shape *schemaworkload.Shape }

// catalogPreset reads the catalog's workload entry called name through blis-substrate's reader.
func catalogPreset(data *load.Data, name string) (*presetSource, error) {
	entry, err := data.Workload(name)
	if err != nil {
		return nil, err
	}
	return &presetSource{Shape: entry.Shape}, nil
}

// presetBoundsError refuses a preset whose token distributions a run cannot sample from: no
// mean, or bounds that are missing or inconsistent.
func presetBoundsError(name string, wl presetWorkload) error {
	if wl.PromptTokensMean <= 0 || wl.OutputTokensMean <= 0 {
		return fmt.Errorf(
			"workload preset %q declares no token distribution (prompt.tokens=%d, output.tokens=%d); both must be > 0",
			name, wl.PromptTokensMean, wl.OutputTokensMean)
	}

	// #1793: BLIS samples from a clamped Gaussian, so it requires the min/max/stdev bounds a
	// preset may leave out: an omitted bound would clamp at zero rather than at "unbounded".
	// The same bound validation as the CLI distribution path (validateDistributionParams), so a
	// preset and the equivalent --prompt-tokens-* / --output-tokens-* flags are refused alike.
	if msg := validateDistributionParams(
		wl.PromptTokensMin, wl.PromptTokensMax,
		wl.OutputTokensMin, wl.OutputTokensMax,
		wl.PromptTokensStdev, wl.OutputTokensStdev,
		wl.PromptTokensMean, wl.OutputTokensMean,
	); msg != "" {
		return fmt.Errorf(
			"workload preset %q has invalid token distribution bounds: %s\n"+
				"  (a preset's YAML keys mirror the CLI flags: e.g. prompt.tokens_min is --prompt-tokens-min)",
			name, msg)
	}
	return nil
}

// loadPresetWorkload reads the named preset from the run's catalog (runData), used by
// `blis run --workload` and `blis convert preset`. The one funnel both share is
// readCatalogPresetWorkload -> presetWorkload.toPresetConfig (R23), which keeps the two from
// resolving one preset name to different distributions.
func loadPresetWorkload(name string) (*presetWorkload, error) {
	data, err := runData()
	if err != nil {
		return nil, err
	}
	return readCatalogPresetWorkload(name, data)
}
