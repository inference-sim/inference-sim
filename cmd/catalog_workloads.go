package cmd

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	blisschemas "github.com/inference-sim/blis-schemas"
	schemaworkload "github.com/inference-sim/blis-schemas/spec/workload"

	"github.com/inference-sim/inference-sim/sim/workload"
)

const (
	// catalogWorkloadsSubdir is the workloads namespace inside a catalog CLONE ROOT — the
	// sibling of models/ established by #1774. A named preset lives at
	// <catalog>/workloads/<name>.yaml.
	catalogWorkloadsSubdir = "workloads"
	// presetFileExt is the extension of a catalog preset definition.
	presetFileExt = ".yaml"
)

// presetWorkload is one named workload preset (chatbot, summarization, contentgen,
// multidoc), as stored in the catalog at <catalog>/workloads/<name>.yaml.
//
// #1769: this is the ONLY declaration of the preset shape. It used to be duplicated — the
// `workloads:` block of the bundled defaults.yaml (the copy actually read) and the
// catalog's workloads/*.yaml (read by nothing) held identical values with nothing keeping
// them in sync, so the drifting copy was the one that moved output. The catalog is now the
// single source of truth, consistent with NS-6 (#1733): a run already requires the catalog
// for its model, and nothing about a run is inferred from a bundled default.
//
// The FILE format is not declared here. It is blis-schemas' workload.Shape, which owns the
// catalog's workloads namespace (blis-catalog#18 moved it to nested prompt:/output:
// distributions), and it is decoded by blis-schemas' own strict LoadWorkload. This struct
// is the value BLIS reads out of a Shape, so there is one parser per catalog format.
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
	}
}

// toPresetConfig converts a catalog preset into the library's PresetConfig. It is the ONE
// construction site for workload.PresetConfig in cmd/ (R4): the three preset consumers
// (`blis run --workload`, `blis convert preset`, `blis observe --workload`) each built it
// inline, so a new preset field had to be threaded through three copies — and a field
// missed in one of them would make the same preset mean different things per command.
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

// catalogWorkloadPath returns the catalog file a named preset is read from:
//
//	<catalog>/workloads/<name>.yaml
//
// This function is PURE — it derives a path and touches no filesystem, so the path law is
// table-testable. There is no flat fallback: the workloads namespace is introduced at its
// settled #1774 location only, the sibling of models/ under the catalog clone root. (The
// pre-#1771 catalogModelDirs carried a flat fallback for the in-repo model catalogs that
// predated the models/ level; #1771 deleted that tree and the fallback, so model and
// workload resolution now share the single clone-root layout.)
//
// The preset name comes from a user flag (--workload / --name), so it is validated against
// path traversal and separators before being joined onto the catalog root.
func catalogWorkloadPath(name, catalog string) (string, error) {
	// Defensive: production callers get a non-empty root from resolveCatalogRoot, which
	// refuses when neither --catalog nor BLIS_CATALOG is set. An empty root here would
	// silently resolve presets relative to the process working directory.
	if catalog == "" {
		return "", fmt.Errorf("model catalog root is empty; pass --catalog or set %s", catalogEnvVar)
	}
	if name == "" {
		return "", fmt.Errorf("no workload preset name was supplied")
	}
	if strings.ContainsAny(name, `/\`) || strings.Contains(name, "..") || filepath.IsAbs(name) {
		return "", fmt.Errorf("workload preset name %q contains invalid path components", name)
	}
	return filepath.Join(catalog, catalogWorkloadsSubdir, name+presetFileExt), nil
}

// availablePresetNames lists the preset names a catalog defines, sorted. Used to make an
// unknown-preset refusal actionable: the valid set is a property of the catalog in use, not
// a hard-coded list that would go stale the moment a catalog adds a preset.
//
// Sorted, so the diagnostic is byte-identical across runs (INV-6). An unreadable or absent
// workloads/ directory yields no names — the caller reports the path either way, so a
// missing namespace does not need a second error channel.
func availablePresetNames(catalog string) []string {
	entries, err := os.ReadDir(filepath.Join(catalog, catalogWorkloadsSubdir))
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), presetFileExt) {
			continue
		}
		names = append(names, strings.TrimSuffix(entry.Name(), presetFileExt))
	}
	sort.Strings(names)
	return names
}

// readCatalogPresetWorkload reads one named preset from the catalog rooted at catalog. It
// is the injectable core of the preset reader — production callers use loadPresetWorkload,
// which resolves the root from --catalog / BLIS_CATALOG.
//
// Every failure is a named refusal (R1): an absent preset names the path looked at and the
// presets the catalog does have; an unreadable or malformed file names the file. There is no
// fallback to a built-in preset, because a silently-substituted token distribution is
// exactly the drift #1769 removes.
func readCatalogPresetWorkload(name, catalog string) (*presetWorkload, error) {
	path, err := catalogWorkloadPath(name, catalog)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(path); err != nil {
		// ENOENT and ENOTDIR both mean "the catalog does not define this preset"; any
		// other failure (EACCES, EIO) is reported as such, so a permission problem
		// is never reported as an unknown preset name.
		if errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
			available := availablePresetNames(catalog)
			have := "none — the catalog defines no workload presets"
			if len(available) > 0 {
				have = strings.Join(available, ", ")
			}
			return nil, fmt.Errorf(
				"workload preset %q is not in the catalog: no %s\n"+
					"  presets defined by this catalog: %s\n"+
					"  add the preset there, or point --catalog / %s at a catalog that has it",
				name, path, have, catalogEnvVar)
		}
		return nil, fmt.Errorf("workload preset %q at %s is not readable: %w", name, path, err)
	}

	// blis-schemas' strict loader (R10): an unrecognized key is a hard error rather than a
	// silently dropped field, which for a token distribution would mean a
	// plausible-but-wrong zero. Its field validation runs too, so a preset BLIS accepts is
	// one the catalog's own gate accepts.
	shape, err := blisschemas.LoadWorkload(path)
	if errors.Is(err, io.EOF) {
		// An empty file has no document; the token-distribution check below reports it.
		shape, err = &schemaworkload.Shape{Name: name}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("workload preset %q at %s is not a valid preset definition: %w", name, path, err)
	}
	wl := presetFromShape(shape)
	// The schema's field validation runs once the means are present, so a preset that
	// states no distribution at all gets the actionable message below rather than the
	// schema's per-field list.
	if wl.PromptTokensMean > 0 && wl.OutputTokensMean > 0 {
		if problems := shape.Validate(); !problems.OK() {
			return nil, fmt.Errorf("workload preset %q at %s is not a valid preset definition: %s",
				name, path, problems.Error())
		}
	}
	if wl.PromptTokensMean <= 0 || wl.OutputTokensMean <= 0 {
		return nil, fmt.Errorf(
			"workload preset %q at %s declares no token distribution (prompt.tokens=%d, output.tokens=%d); both must be > 0",
			name, path, wl.PromptTokensMean, wl.OutputTokensMean)
	}

	// #1793: the mean check above leaves the min/max/stdev bounds unvalidated, so a
	// malformed preset — min > max, a mean outside [min, max], a negative stdev, or an
	// omitted-and-therefore-zero min/max — would reach the Gaussian sampler
	// (sim/workload/distribution.go) and be silently clamped to a wrong distribution. Apply
	// the SAME bound validation the CLI distribution path applies (validateDistributionParams
	// in root.go, shared by the concurrency and rate-mode synthesis paths), so a preset and
	// the equivalent --prompt-tokens-* / --output-tokens-* flags are refused identically. A
	// preset with valid bounds is unaffected (INV-6): the bundled presets all pass.
	//
	// This is stricter than blis-schemas, where tokens_min/tokens_max are optional. The
	// difference is deliberate: BLIS samples from a clamped Gaussian, and an omitted bound
	// would clamp at zero rather than at "unbounded", so BLIS requires what its sampler needs.
	if msg := validateDistributionParams(
		wl.PromptTokensMin, wl.PromptTokensMax,
		wl.OutputTokensMin, wl.OutputTokensMax,
		wl.PromptTokensStdev, wl.OutputTokensStdev,
		wl.PromptTokensMean, wl.OutputTokensMean,
	); msg != "" {
		return nil, fmt.Errorf(
			"workload preset %q at %s has invalid token distribution bounds: %s\n"+
				"  (a preset's YAML keys mirror the CLI flags: e.g. prompt.tokens_min is --prompt-tokens-min)",
			name, path, msg)
	}
	return &wl, nil
}

// loadPresetWorkload locates the catalog and reads the named preset from it. It is the
// catalog-locating wrapper, used by `blis run --workload` and `blis convert preset`;
// `blis observe --workload` resolves the catalog root itself (runObserve does it once, then
// passes it down) and so calls readCatalogPresetWorkload directly.
//
// The funnel shared by all three consumers is therefore readCatalogPresetWorkload →
// presetWorkload.toPresetConfig, not this function (R23) — that is what keeps the three from
// resolving one preset name to different distributions.
//
// (`blis replay` resolves no preset: it replays recorded requests, so INV-13 does not reach
// preset resolution.)
func loadPresetWorkload(name string) (*presetWorkload, error) {
	catalog, err := resolveCatalogRoot()
	if err != nil {
		return nil, err
	}
	return readCatalogPresetWorkload(name, catalog)
}
