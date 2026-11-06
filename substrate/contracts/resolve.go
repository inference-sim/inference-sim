package contracts

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// ValidateAgainstCatalog resolves every by-name reference a bundle's scenario makes
// against a catalog and registry on disk. It is separate from Validate on purpose:
// Validate is pure, and this repository deliberately does not vendor the catalog, so a
// caller without one checked out gets the pure checks and one with a catalog gets the
// references resolved too. The README's "two layers" split rests on that separation —
// keeping the shapes apart from the contents is what lets either side be validated in CI
// without the other being present — so resolution lives behind its own entry point rather
// than inside Validate.
//
// It checks presence, not validity: that model names a directory the catalog has, that
// hardware, fabric and a workload shape name files it has, and that every storage tier,
// coefficient set and acceptance-profile set a scenario names exists. A replay arm that
// references a catalog replay workload is loaded and merged with its run choices
// (ResolveReplay), since a reference that resolves to an invalid replay is a miss too. The
// Trace arm and an inline Replay are excluded on purpose — they reference external
// captures by path or repository, not catalog identities. Validating the
// resolved documents is Validate's job, reached by loading them into a Bundle; this answers
// the prior question of whether the names resolve at all, which nothing else does —
// `hardware: nvidia-h200` validates clean today and fails later, elsewhere, as a missing file
// rather than an invalid scenario.
//
// Every failure names the path it looked for and enumerates what the catalog actually has,
// so the error teaches the naming convention in one turn rather than leaving a generator to
// guess: `hardware: nvidia-h200` → "looked for hardware/nvidia-h200.yaml; the catalog has
// [a100-80 a100-sxm b200 h100 h200 ...]".
//
// A nil scenario resolves nothing and is reported as such rather than passing silently: a
// bundle with no scenario names no references, so there is nothing to resolve, but a caller
// that meant to pass one should not read an empty green as success. An empty catalogRoot or
// registryRoot is only an error when a reference of the corresponding kind is actually made.
//
// Problems are recorded under dotted paths below "references." — references.hardware,
// references.coefficients[1] — matching how spec/ validators report, so a reader of a mixed
// report sees resolution findings in the same shape as field findings.
func ValidateAgainstCatalog(b Bundle, catalogRoot, registryRoot string) Report {
	field := &validate.Problems{}

	if b.Scenario == nil {
		field.Field("references",
			"no scenario in the bundle: there are no by-name references to resolve")
		return Report{Field: field, Rule: &validate.Problems{}}
	}
	s := b.Scenario

	// model → models/<name>/ (a directory, since a model is a tree: graph.yaml, model.yaml,
	// config.json). The reference is the directory name.
	if s.Model != "" {
		resolveDir(field, "references.model", catalogRoot, "models", s.Model)
	}

	// hardware → hardware/<name>.yaml, fabric → networks/<name>.yaml. Both are single files
	// named by stem (#27: the stem is the identity).
	//
	// The chip's entry also owns the cluster's packaging where its product fixes it
	// (scenario.Cluster.Packaging), so a hardware reference that resolves is loaded and the
	// cluster's packaging resolved against it: a gpus_per_node or gpus_per_rack that
	// contradicts the entry is refused here, at scenario.cluster.
	if resolveFile(field, "references.hardware", catalogRoot, "hardware", s.Cluster.Hardware) {
		chip, err := LoadChip(filepath.Join(catalogRoot, "hardware", s.Cluster.Hardware+".yaml"))
		if err != nil {
			field.Field("references.hardware", "%q did not load, so the cluster's packaging cannot be resolved: %v",
				s.Cluster.Hardware, err)
		} else {
			field.Merge("scenario.cluster", s.Cluster.ValidatePackaging(chip))
		}
	}
	if s.Cluster.Fabric != "" {
		resolveFile(field, "references.fabric", catalogRoot, "networks", s.Cluster.Fabric)
	}

	// workload shape → workloads/<name>.yaml. A Scenario's workload is a sum type: the Shape
	// arm names a catalog workload by identity (resolve it), while the Trace arm references
	// an external capture by file PATH, not a catalog identity — so only the shape arm is a
	// by-name catalog reference and the trace arm is deliberately not resolved here.
	if s.Workload != nil && s.Workload.Shape != "" {
		if resolveFile(field, "references.workload.shape", catalogRoot, "workloads", s.Workload.Shape) {
			w, err := LoadCatalogWorkload(filepath.Join(catalogRoot, "workloads", s.Workload.Shape+".yaml"))
			if err == nil && w.Shape == nil {
				field.Field("references.workload.shape",
					"%q is a replay workload, not a shape; bind it as replay: {workload: %s, concurrency: ...}",
					s.Workload.Shape, s.Workload.Shape)
			}
		}
	}
	// A replay reference names a catalog replay workload, which must exist, be a replay, and
	// resolve with the binding's run choices to a valid replay.
	if s.Workload != nil && s.Workload.Replay != nil && s.Workload.Replay.Workload != "" {
		name := s.Workload.Replay.Workload
		if resolveFile(field, "references.workload.replay.workload", catalogRoot, "workloads", name) {
			if _, err := ResolveReplay(catalogRoot, s.Workload.Replay); err != nil {
				field.Field("references.workload.replay.workload", "%v", err)
			}
		}
	}

	// storage → a key in devices/storage.yaml. Unlike the others this is a membership check
	// inside one mapping file, so it loads the file once and checks every named tier against
	// it, naming the inventory in the error exactly as the offload-tier check does.
	if len(s.Cluster.Storage) > 0 {
		resolveStorage(field, catalogRoot, s.Cluster.Storage)
	}

	// coefficients → coefficients/<name>.yaml under the registry, which is a separate root:
	// a scenario is fitted against registry sets, and the registry is not the catalog.
	for i, name := range s.Coefficients {
		if name == "" {
			continue
		}
		resolveFile(field, fmt.Sprintf("references.coefficients[%d]", i),
			registryRoot, "coefficients", name)
	}

	// acceptance.profiles -> acceptance/<name>.yaml under the registry, the acceptance
	// counterpart of a coefficient set.
	if s.Acceptance != nil && s.Acceptance.Profiles != "" {
		resolveFile(field, "references.acceptance.profiles", registryRoot, "acceptance",
			s.Acceptance.Profiles)
	}

	// routing.profiles -> routing/<name>.yaml under the registry.
	if s.Routing != nil && s.Routing.Profiles != "" {
		resolveFile(field, "references.routing.profiles", registryRoot, "routing",
			s.Routing.Profiles)
	}

	return Report{Field: field, Rule: &validate.Problems{}}
}

// resolveDir reports a problem at path unless <root>/<dir>/<name> is a directory the catalog
// has. The error lists the catalog's real subdirectories in the namespace.
func resolveDir(p *validate.Problems, path, root, dir, name string) {
	if root == "" {
		p.Field(path, "%q cannot be resolved: no catalog root was given", name)
		return
	}
	if !validStem(p, path, name, dir) {
		return
	}
	full := filepath.Join(root, dir, name)
	info, err := os.Stat(full)
	if err != nil || !info.IsDir() {
		p.Field(path, "%q: looked for %s; the catalog has %v",
			name, filepath.Join(dir, name)+string(filepath.Separator),
			dirEntries(root, dir, true))
	}
}

// resolveFile reports a problem at path unless <root>/<dir>/<name>.yaml exists. The error
// names the looked-for path and lists the real entries (as stems), so it teaches the naming
// convention; it names the registry rather than the catalog for a coefficients reference.
// It returns whether the file was found.
func resolveFile(p *validate.Problems, path, root, dir, name string) bool {
	if name == "" {
		p.Field(path, "required: a scenario must name a %s", dir)
		return false
	}
	if root == "" {
		p.Field(path, "%q cannot be resolved: no %s root was given", name, rootKind(dir))
		return false
	}
	if !validStem(p, path, name, dir) {
		return false
	}
	info, err := os.Stat(filepath.Join(root, dir, name+".yaml"))
	if err != nil || !info.Mode().IsRegular() {
		// A regular file, not merely a path that stats: a directory named `h200.yaml/`
		// would otherwise satisfy an error-only check and resolve as if it were the chip
		// file, which it is not — the same file-vs-directory distinction cmd/validate-catalog
		// makes when it loads an artifact.
		p.Field(path, "%q: looked for %s; %s has %v",
			name, filepath.Join(dir, name+".yaml"), rootKind(dir),
			dirEntries(root, dir, false))
		return false
	}
	return true
}

// validStem reports whether name is a bare catalog identity — one ordinary path segment that
// names a sibling inside the namespace. A reference is a filename stem (#27: identity IS the
// filename), so a name carrying a separator, a parent-dir hop, or a lone/trailing dot is not a
// catalog miss to list alternatives for; it is a malformed reference that would otherwise
// resolve something OUTSIDE the entry it names and read as valid — "../models/x/graph" statting
// models/x/graph.yaml, or "." statting the models/ directory itself (filepath.Join cleans a
// "." segment to the parent, which os.Stat then reports as an existing dir).
//
// The test is positional, not a blocklist: a valid stem must survive filepath.Clean unchanged
// AND contain no separator, so the single segment the caller joins is exactly the one it typed.
// That rejects "", ".", "..", "a/b", "a..b"-free-but-cleanable forms, and trailing-dot names in
// one check rather than enumerating each. On a bad name it records the problem and returns false
// so the caller stops before statting.
func validStem(p *validate.Problems, path, name, dir string) bool {
	if name == "" || name == "." || name == ".." ||
		strings.ContainsAny(name, `/\`) || filepath.Clean(name) != name {
		p.Field(path,
			"%q is not a valid %s identity: a reference is a bare filename stem, with no path separator, %q, or %q",
			name, dir, ".", "..")
		return false
	}
	return true
}

// resolveStorage checks every named tier against the keys in devices/storage.yaml, loading
// the file once. A tier not in it is named alongside the tiers that are, so the error lists
// the inventory the scenario could have drawn from.
func resolveStorage(p *validate.Problems, root string, tiers []string) {
	if root == "" {
		p.Field("references.storage", "storage tiers cannot be resolved: no catalog root was given")
		return
	}
	devices, err := LoadStorageDevices(filepath.Join(root, "devices", "storage.yaml"))
	if err != nil {
		p.Field("references.storage",
			"devices/storage.yaml did not load, so no tier can be resolved: %v", err)
		return
	}
	have := make(map[string]bool, len(devices))
	names := make([]string, 0, len(devices))
	for _, d := range devices {
		have[d.Name] = true
		names = append(names, d.Name)
	}
	sort.Strings(names)
	for i, tier := range tiers {
		if tier == "" {
			continue
		}
		if !have[tier] {
			p.Field(fmt.Sprintf("references.storage[%d]", i),
				"%q is not a tier in devices/storage.yaml; it has %v", tier, names)
		}
	}
}

// dirEntries lists the catalog's entries in a namespace, for an error that teaches the
// naming convention. For a file namespace it returns the stems (h200, not h200.yaml); for a
// directory namespace (models) it returns the subdirectory names. A root that cannot be read
// yields an empty list rather than an error of its own: the caller is already reporting the
// miss, and "the catalog has []" is a truthful answer that also surfaces an empty or wrong
// root.
func dirEntries(root, dir string, wantDirs bool) []string {
	entries, err := os.ReadDir(filepath.Join(root, dir))
	if err != nil {
		return []string{}
	}
	out := []string{}
	for _, e := range entries {
		if e.IsDir() != wantDirs {
			continue
		}
		name := e.Name()
		if !wantDirs {
			if !strings.HasSuffix(name, ".yaml") {
				continue
			}
			name = strings.TrimSuffix(name, ".yaml")
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// rootKind names which root a namespace lives under, so an error says "the catalog has" for
// a chip and "the registry has" for a coefficient set rather than a generic "it has".
func rootKind(dir string) string {
	if dir == "coefficients" || dir == "acceptance" {
		return "the registry"
	}
	return "the catalog"
}

// ResolveReplay returns the inline replay a binding's replay arm describes: an inline
// replay as it is, or a reference merged with the catalog replay workload it names, read
// from <catalogRoot>/workloads/<name>.yaml. It refuses a missing entry, an entry that is a
// shape, and a merge that is not a valid replay.
func ResolveReplay(catalogRoot string, r *workload.Replay) (*workload.Replay, error) {
	if r.Workload == "" {
		return r, nil
	}
	if catalogRoot == "" {
		return nil, fmt.Errorf("replay workload %q cannot be resolved: no catalog root was given", r.Workload)
	}
	p := &validate.Problems{}
	if !validStem(p, "workload", r.Workload, "workloads") {
		return nil, fmt.Errorf("%s", p.Error())
	}
	w, err := LoadCatalogWorkload(filepath.Join(catalogRoot, "workloads", r.Workload+".yaml"))
	if err != nil {
		return nil, err
	}
	if w.Replay == nil {
		return nil, fmt.Errorf("workloads/%s.yaml is a shape, not a replay workload", r.Workload)
	}
	if p := w.Replay.Validate(); !p.OK() {
		return nil, fmt.Errorf("workloads/%s.yaml is not a valid replay workload:\n%s", r.Workload, p.Error())
	}
	return r.Resolve(w.Replay)
}
