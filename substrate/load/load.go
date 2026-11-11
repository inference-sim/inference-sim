// Package load reads the catalog and the registry a run uses, from the copy embedded in
// this substrate release or from directories a caller names, and checks them before any
// value is used (design §3.3).
//
// Both sources go through one path: the same file patterns (package substrate), the same
// refusals, the same validators CI runs (package validate/dataset), and the same content
// hash (SUB-9). A path replaces the embedded copy; it never overlays it (SIM-6). Overlays
// are named separately and add or replace entries by identity.
//
// Open is the one constructor. A Data it returns is complete and checked: every data file
// parsed and validated, the registry checked against the catalog, and the combined data
// checked again after overlays. Its accessors return documents already parsed; the
// selected-entry report (Report) records which registry entries lookups returned and which
// registry files no reader loaded.
package load

import (
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"
	"sync"

	substrate "github.com/inference-sim/blis-substrate"
	"github.com/inference-sim/blis-substrate/validate/dataset"
)

// Source says where a catalog or registry came from.
type Source string

const (
	// SourceEmbedded is the copy compiled into this substrate release.
	SourceEmbedded Source = "embedded"
	// SourcePath is a directory the caller named.
	SourcePath Source = "path"
)

// Options name the data a run uses. An empty Catalog or Registry means the embedded copy.
// Each overlay is a directory with the catalog's or the registry's layout, applied in the
// order given.
type Options struct {
	Catalog          string
	Registry         string
	CatalogOverlays  []string
	RegistryOverlays []string
}

// Provenance is what a run records about one of its catalog and registry (design §3.3, X8):
// its source, its content hash, each overlay's hash, and the hash of the combined data.
type Provenance struct {
	Source Source
	// Path is the directory read, empty for the embedded copy. It is recorded for a
	// reader; the byte gate leaves it out with the source (X6).
	Path string
	// Hash is the content hash of the catalog or registry before overlays.
	Hash     string
	Overlays []Overlay
	// CombinedHash is Hash when no overlay is used; otherwise the SHA-256 of
	// "<Hash>  base\n" followed by "<overlay hash>  overlay\n" for each overlay in order.
	CombinedHash string
}

// Overlay is one overlay directory and its content hash.
type Overlay struct {
	Path string
	Hash string
}

// Data is a checked catalog and registry.
type Data struct {
	Catalog  Provenance
	Registry Provenance

	catalog  *dataset.Catalog
	registry *dataset.Registry
	// setFile maps each coefficient set of the combined registry to its file.
	setFile map[string]string

	mu       sync.Mutex
	loaded   map[string]bool // registry files a reader loaded, by path relative to registry/
	selected map[string]Selected
	// observers are asked at report time whether a coefficient was read outside its fit range.
	observers []ExtrapolationObserver
}

// Embedded opens the embedded catalog and registry with no overlay.
func Embedded() (*Data, error) { return Open(Options{}) }

// Open reads, checks and returns the data o names.
func Open(o Options) (*Data, error) {
	cat, catProv, err := readBase("catalog", o.Catalog, substrate.Catalog, catalogRules())
	if err != nil {
		return nil, err
	}
	reg, regProv, err := readBase("registry", o.Registry, substrate.Registry, registryRules())
	if err != nil {
		return nil, err
	}

	// The base data are checked before overlays, so a fault in the released data is
	// reported as the base's, not as an overlay's.
	c, problems := dataset.CheckCatalog(cat.files)
	if err := refuse(cat.label, problems); err != nil {
		return nil, err
	}
	r, problems := dataset.CheckRegistry(reg.files)
	if err := refuse(reg.label, problems); err != nil {
		return nil, err
	}

	d := &Data{Catalog: catProv, Registry: regProv,
		loaded: map[string]bool{}, selected: map[string]Selected{}, setFile: map[string]string{}}
	for stem := range r.Sets {
		d.setFile[stem] = "coefficients/" + stem + ".yaml"
	}

	if len(o.CatalogOverlays) > 0 {
		files := cloneFiles(cat.files)
		for _, dir := range o.CatalogOverlays {
			ov, err := readOverlay(dir, catalogOverlayRules())
			if err != nil {
				return nil, err
			}
			if err := applyCatalogOverlay(files, ov.files); err != nil {
				return nil, err
			}
			d.Catalog.Overlays = append(d.Catalog.Overlays, Overlay{Path: dir, Hash: Hash(ov.files)})
		}
		if c, problems = dataset.CheckCatalog(files); len(problems) > 0 {
			return nil, refuse("the catalog with overlays "+strings.Join(o.CatalogOverlays, ", "), problems)
		}
	}
	for _, dir := range o.RegistryOverlays {
		ov, err := readOverlay(dir, registryOverlayRules())
		if err != nil {
			return nil, err
		}
		or, problems := dataset.CheckRegistry(ov.files)
		if err := refuse(dir, problems); err != nil {
			return nil, err
		}
		if err := checkSetLevelFields(r, or); err != nil {
			return nil, err
		}
		applyRegistryOverlay(r, or)
		for stem := range or.Sets {
			if _, ok := d.setFile[stem]; !ok {
				d.setFile[stem] = "coefficients/" + stem + ".yaml"
			}
		}
		d.Registry.Overlays = append(d.Registry.Overlays, Overlay{Path: dir, Hash: Hash(ov.files)})
	}
	if len(o.RegistryOverlays) > 0 {
		// The combined registry is checked in full: each merged set's own rules, and the
		// rules across sets (VAL-4).
		var problems []dataset.Problem
		for _, stem := range sortedKeys(r.Sets) {
			for _, e := range r.Sets[stem].Validate().Errors() {
				problems = append(problems, dataset.Problem{File: "coefficients/" + stem + ".yaml", Message: e.String()})
			}
		}
		problems = append(problems, dataset.CheckOrdered(r.Sets)...)
		if err := refuse("the registry with overlays "+strings.Join(o.RegistryOverlays, ", "), problems); err != nil {
			return nil, err
		}
	}
	// The registry must fit the catalog it runs with, whatever the source of either.
	if err := refuse(regProv.label()+" against "+catProv.label(), dataset.CheckFits(c, r)); err != nil {
		return nil, err
	}
	d.catalog, d.registry = c, r
	d.Catalog.CombinedHash = combinedHash(d.Catalog.Hash, d.Catalog.Overlays)
	d.Registry.CombinedHash = combinedHash(d.Registry.Hash, d.Registry.Overlays)
	return d, nil
}

func (p Provenance) label() string {
	if p.Source == SourceEmbedded {
		return "the embedded data"
	}
	return p.Path
}

// readBase reads the catalog or registry from dir, or from the embedded copy when dir is
// empty.
func readBase(kind, dir string, embedded func() fs.FS, r rules) (*tree, Provenance, error) {
	label, fsys, prov := "the embedded "+kind, embedded(), Provenance{Source: SourceEmbedded}
	if dir != "" {
		var err error
		if fsys, err = dirFS(dir); err != nil {
			return nil, Provenance{}, fmt.Errorf("the %s path: %w", kind, err)
		}
		label, prov = dir, Provenance{Source: SourcePath, Path: dir}
	}
	t, err := readTree(label, fsys, r)
	if err != nil {
		return nil, Provenance{}, err
	}
	if len(t.files) == 0 {
		return nil, Provenance{}, fmt.Errorf("%s holds no %s data: none of %v matched", label, kind, r.patterns)
	}
	prov.Hash = Hash(t.files)
	return t, prov, nil
}

func readOverlay(dir string, r rules) (*tree, error) {
	fsys, err := dirFS(dir)
	if err != nil {
		return nil, fmt.Errorf("an overlay: %w", err)
	}
	t, err := readTree(dir, fsys, r)
	if err != nil {
		return nil, err
	}
	if len(t.files) == 0 {
		return nil, fmt.Errorf("the overlay %s holds no data file: none of %v matched", dir, r.patterns)
	}
	return t, nil
}

func refuse(source string, problems []dataset.Problem) error {
	if len(problems) == 0 {
		return nil
	}
	msgs := make([]string, len(problems))
	for i, p := range problems {
		msgs[i] = p.String()
	}
	return &Error{Source: source, Problems: msgs}
}

// catalogRules are the catalog's data patterns. Inside the data directories, each model's
// config.json is ignored by name, and so is everything under ledger/ except published/
// (design §3.3).
func catalogRules() rules {
	return rules{patterns: substrate.CatalogPatterns, extraDirs: []string{"ledger/published"}, ignored: func(rel string) bool {
		if path.Base(rel) == "config.json" && path.Dir(path.Dir(rel)) == "models" {
			return true
		}
		return strings.HasPrefix(rel, "ledger/") && rel != "ledger/published" &&
			!strings.HasPrefix(rel, "ledger/published/")
	}}
}

func registryRules() rules { return rules{patterns: substrate.RegistryPatterns} }

// catalogOverlayRules are the files a catalog overlay may hold: a model directory, or a
// hardware, device, network or workload file. Ledger files, and the measurement-source
// registry, are refused by name.
func catalogOverlayRules() rules {
	var ps []string
	for _, p := range substrate.CatalogPatterns {
		if !strings.HasPrefix(p, "measurements/") && !strings.HasPrefix(p, "ledger/") {
			ps = append(ps, p)
		}
	}
	return rules{patterns: ps, refused: map[string]string{
		"ledger":       "an overlay may not contain ledger files",
		"measurements": "an overlay adds models, hardware, devices, networks or workloads; the measurement-source registry is not overlaid",
	}, ignored: catalogRules().ignored}
}

func registryOverlayRules() rules {
	return rules{patterns: substrate.RegistryPatterns, refused: map[string]string{
		"ledger": "an overlay may not contain ledger files",
	}}
}

func cloneFiles(m map[string][]byte) map[string][]byte {
	out := make(map[string][]byte, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
