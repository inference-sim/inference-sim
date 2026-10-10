package cmd

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"

	"gopkg.in/yaml.v3"

	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/hardware"
	schemamodel "github.com/inference-sim/blis-schemas/spec/model"
)

// This file implements the strict WHOLE-CATALOG load behind the R1/C6 acceptance gate
// (#1750). C6 states: "There is no validate command: the simulator validates whatever it
// reads and fails with a message naming the file and the problem. CI runs a load over every
// catalog entry, which is the same code path."
//
// Two consequences shape the design:
//
//  1. NO validate command, and no bespoke validator. Every namespace a run consumes is
//     loaded through the very function the run uses — model graphs through blis-schemas'
//     LoadModelGraph (what blis-latency-kernel opens), workload presets through
//     readCatalogPresetWorkload, storage devices through loadCatalogStorageDevices. The
//     model identity manifest (models/<name>/model.yaml) and the chip descriptors
//     (hardware/<gpu>.yaml) are read through blis-schemas' own LoadModelIdentity and LoadChip
//     rather than a re-derivation of their formats. The vendor config.json beside each graph
//     is not read by BLIS at all — the graph is derived from it in blis-catalog, whose own
//     CI checks the pair — so it is not part of this gate.
//
//  2. The GATE is a test, not a subcommand: cmd/catalog_load_test.go drives loadCatalog
//     against the committed fixture catalog (testdata/catalog) unconditionally, and
//     scripts/catalog-load-gate.sh drives it against a pinned blis-catalog checkout. The
//     workflow job that calls that script is a PENDING HUMAN EDIT to
//     .github/workflows/ci.yml — the delivery loop's token cannot push workflow files — and its
//     body is quoted verbatim in the script's header.
//
// SCOPE BOUNDARY, stated because it is the one place this file could be misread as doing
// less than the issue asks: the completeness rule ("a models/<name>/ dir missing either
// graph.yaml or model.yaml fails") and the deployment-fact rule are properties of the
// CATALOG AS A WHOLE, enforced here. `blis run` is deliberately NOT tightened to require
// model.yaml: it reads only the graph of the model its scenario names, so making a run depend
// on a provenance file it never reads would break working scratch clones for no fidelity
// gain. The graph.yaml half of the rule IS the run path — a graph that is absent or invalid
// fails a run and fails this gate through the same loader.

const (
	// catalogHardwareSubdir is the hardware namespace inside a catalog CLONE ROOT (#1774) —
	// one file per GPU, a SIBLING of models/, workloads/ and devices/.
	catalogHardwareSubdir = "hardware"
	// catalogModelEntryFile is the identity/provenance half of a model entry. The other
	// half is catalogModelGraphFile, the derived graph a run prices from.
	catalogModelEntryFile = "model.yaml"
	// catalogYAMLExt is the extension of every catalog-authored file (as opposed to the
	// vendor JSON configs).
	catalogYAMLExt = ".yaml"
)

// catalogLoadReport is the outcome of one whole-catalog load: how much was loaded per
// namespace, plus every problem found. Counts are reported so the gate can assert
// NON-VACUITY — a loader that silently walked an empty tree would otherwise pass.
//
// Problems are ACCUMULATED rather than fail-fast, and sorted, so one CI run names every
// broken file instead of making an operator fix them one push at a time (and so the
// diagnostic is byte-identical across runs, INV-6).
type catalogLoadReport struct {
	Models        int
	Hardware      int
	Presets       int
	DeviceClasses int
	Problems      []string
}

// Err returns a single error naming every problem found, or nil when the catalog loaded
// clean. The message leads with the count so a CI log line is readable without expanding it.
func (r catalogLoadReport) Err() error {
	if len(r.Problems) == 0 {
		return nil
	}
	return fmt.Errorf("catalog load found %d problem(s):\n  %s",
		len(r.Problems), strings.Join(r.Problems, "\n  "))
}

// loadCatalog loads EVERY entry in the catalog rooted at root, strictly, through the same
// code paths a run uses (see the file header). It returns what it loaded and what it
// rejected; it never writes to the catalog and never fetches anything.
//
// models/ must exist and hold at least one entry — a root with no models is not a catalog,
// and accepting it would make this gate pass on a mistyped path. The other three namespaces
// are loaded when present and skipped when absent, because a catalog is allowed to predate a
// namespace (nothing consumes hardware/ yet) and a hard requirement here would fail a
// perfectly usable catalog. The gate asserts the counts it expects, so "absent" cannot hide
// behind "loaded nothing".
func loadCatalog(root string) (catalogLoadReport, error) {
	if root == "" {
		return catalogLoadReport{}, fmt.Errorf("catalog root is empty; pass --catalog or set %s", catalogEnvVar)
	}
	info, err := os.Stat(root)
	if err != nil {
		return catalogLoadReport{}, fmt.Errorf("catalog root %q cannot be inspected: %w", root, err)
	}
	if !info.IsDir() {
		return catalogLoadReport{}, fmt.Errorf("catalog root %q is not a directory", root)
	}

	var report catalogLoadReport
	var problems []string
	// The namespace list is hardcoded rather than discovered, because each namespace has its
	// own typed reader with its own shape — there is nothing generic to iterate. The cost is
	// that a namespace BLIS cannot read is a namespace this gate does not check, so extending
	// this list is part of adding a reader.
	//
	// R2 adds networks/ + clusters (#1817): add each here when its reader lands, or this gate
	// silently skips it.
	//
	// The networks/ fabric reader is CONSTRAINED before it is written (#1838): a fabric class
	// states its nominal InterNodeBwGBps — which IS the PD-transfer bandwidth, there is no
	// separate PD figure (R2H2, blis-catalog#10) — and carries NO PDTransferBaseLatencyMs.
	// blis-catalog#12 removed that key (the nominal value was always a 0 placeholder) and the
	// closed fabric schema now rejects it, so a reader that reads or requires it reads a key the
	// catalog does not have the moment CATALOG_REVISION advances past that PR. The PD-transfer
	// base latency stays a CLI input owned by blis-registry (--pd-transfer-base-latency, default
	// 0.05 ms, blis-registry#10): the effective value is that number alone, with no catalog 0 to
	// compose with. cmd/catalog_networks_fabric_test.go holds the guards — and its BC-4 tripwire
	// fires on ANY field this report gains, under any name, since a fabric reader could be called
	// anything: it names the three rules and tells you to update its known-field set (or, for the
	// fabric reader itself, to delete the tripwire) once you have read them.
	for _, namespace := range []struct {
		count *int
		load  func(string) (int, []string)
	}{
		{&report.Models, loadCatalogModelEntries},
		{&report.Hardware, loadCatalogHardwareEntries},
		{&report.Presets, loadCatalogWorkloadEntries},
		{&report.DeviceClasses, loadCatalogDeviceEntries},
	} {
		loaded, found := namespace.load(root)
		*namespace.count = loaded
		problems = append(problems, found...)
	}
	sort.Strings(problems)
	report.Problems = problems
	return report, nil
}

// loadCatalogModelEntries loads every models/<name>/ entry: both halves must be present, the
// graph must load and validate through the loader a run uses, and model.yaml must parse
// strictly and name its own directory.
func loadCatalogModelEntries(root string) (int, []string) {
	dir := filepath.Join(root, catalogModelsSubdir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, []string{fmt.Sprintf("%s: models namespace cannot be read: %v (a catalog must have a %s/ directory)",
			dir, err, catalogModelsSubdir)}
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	if len(names) == 0 {
		return 0, []string{fmt.Sprintf("%s: models namespace holds no model entries", dir)}
	}

	var problems []string
	loaded := 0
	for _, name := range names {
		entryProblems := loadCatalogModelEntry(root, name)
		problems = append(problems, entryProblems...)
		if len(entryProblems) == 0 {
			loaded++
		}
	}
	return loaded, problems
}

// loadCatalogModelEntry loads one models/<name>/ entry and returns every problem with it.
// Both halves are reported independently: a broken model.yaml must not hide a broken
// graph.yaml, or an operator fixes one, re-pushes, and learns about the other.
func loadCatalogModelEntry(root, name string) []string {
	var problems []string
	entryDir := filepath.Join(root, catalogModelsSubdir, name)
	if _, err := readCatalogModelGraph(entryDir, name); err != nil {
		problems = append(problems, err.Error())
	}
	// model.yaml — no run reads it, so this is its only validation.
	if _, err := readCatalogModelEntry(entryDir, name); err != nil {
		problems = append(problems, err.Error())
	}
	return problems
}

// readCatalogModelGraph loads and validates <catalog>/models/<name>/graph.yaml through
// blis-schemas — the loader blis-latency-kernel opens a scenario's model with, so a graph
// that fails here fails a run. The name the graph states must match its directory, because
// the directory is what a scenario's model resolves against.
func readCatalogModelGraph(entryDir, name string) (*schemamodel.Graph, error) {
	path := filepath.Join(entryDir, catalogModelGraphFile)
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
			return nil, fmt.Errorf("%s: model entry is incomplete — no %s (the model graph a run "+
				"prices from; blis-catalog derives it from the vendor config)", path, catalogModelGraphFile)
		}
		return nil, fmt.Errorf("%s: %s cannot be inspected: %w", path, catalogModelGraphFile, err)
	}
	graph, err := blisschemas.LoadModelGraph(path)
	if err != nil {
		return nil, fmt.Errorf("%s: %s is not a valid model graph: %w", path, catalogModelGraphFile, err)
	}
	if problems := graph.Validate(); !problems.OK() {
		return nil, fmt.Errorf("%s: %s is not a valid model graph: %s", path, catalogModelGraphFile, problems.Error())
	}
	if graph.Name != name {
		return nil, fmt.Errorf("%s: %s declares name %q but sits in directory %q; the directory name is "+
			"what a scenario's model resolves against, so the two must agree", path, catalogModelGraphFile, graph.Name, name)
	}
	return graph, nil
}

// readCatalogModelEntry reads and validates <catalog>/models/<name>/model.yaml through
// blis-schemas, which owns the identity format: LoadModelIdentity decodes it strictly and
// Identity.Validate requires the name and the source provenance.
//
// Every failure names the file (C6): absent, unreadable, an unknown key, a missing identity
// or provenance field, or a name that disagrees with the directory it sits in. The
// name/directory agreement check is the caller's half of the contract blis-schemas states:
// the DIRECTORY is what a scenario's model resolves against, so a model.yaml naming a
// different model makes the entry's provenance describe something other than the graph
// beside it.
func readCatalogModelEntry(entryDir, name string) (*schemamodel.Identity, error) {
	path := filepath.Join(entryDir, catalogModelEntryFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) || errors.Is(err, syscall.ENOTDIR) {
			return nil, fmt.Errorf("%s: model entry is incomplete — no %s (a catalog model entry is "+
				"the derived %s plus %s, which records the name and the source repo/revision its "+
				"vendor config was fetched from)", path, catalogModelEntryFile, catalogModelGraphFile, catalogModelEntryFile)
		}
		return nil, fmt.Errorf("%s: %s is not readable: %w", path, catalogModelEntryFile, err)
	}

	// The catalog-wide YAML rules first, so the specific "not catalog data" / "not one
	// document" message wins over the generic unknown-key one.
	if err := checkCatalogAuthoredYAML(path, data); err != nil {
		return nil, err
	}

	identity, err := blisschemas.LoadModelIdentity(path)
	if err != nil {
		return nil, fmt.Errorf("%s: %s is not a valid model entry: %w", path, catalogModelEntryFile, err)
	}
	if problems := identity.Validate(); !problems.OK() {
		return nil, fmt.Errorf("%s: %s is not a valid model entry: %s", path, catalogModelEntryFile, problems.Error())
	}
	if identity.Name != name {
		return nil, fmt.Errorf("%s: %s declares name %q but sits in directory %q; the directory name is "+
			"what a scenario's model resolves against, so the two must agree", path, catalogModelEntryFile, identity.Name, name)
	}
	return identity, nil
}

// loadCatalogHardwareEntries loads every hardware/<gpu>.yaml file. Absent namespace = not a
// problem (nothing consumes it yet); present namespace = every file in it must load.
func loadCatalogHardwareEntries(root string) (int, []string) {
	dir := filepath.Join(root, catalogHardwareSubdir)
	paths, err := catalogYAMLFiles(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, []string{fmt.Sprintf("%s: hardware namespace cannot be read: %v", dir, err)}
	}
	var problems []string
	loaded := 0
	for _, path := range paths {
		if _, err := readCatalogHardwareEntry(path); err != nil {
			problems = append(problems, err.Error())
			continue
		}
		loaded++
	}
	return loaded, problems
}

// readCatalogHardwareEntry reads and validates one <catalog>/hardware/<gpu>.yaml through
// blis-schemas, which owns that format: it is a chip descriptor (peak rates, memory, link
// bandwidth), decoded strictly by LoadChip and checked by the chip's own field validation.
// A run reads the scenario's chip through blis-latency-kernel, which uses the same loader, so
// this checks every chip the catalog publishes, not only the ones a scenario happens to name.
func readCatalogHardwareEntry(path string) (*hardware.Chip, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("%s: hardware entry is not readable: %w", path, err)
	}
	if err := checkCatalogAuthoredYAML(path, data); err != nil {
		return nil, err
	}
	chip, err := blisschemas.LoadChip(path)
	if err != nil {
		return nil, fmt.Errorf("%s: hardware entry is not a valid chip descriptor: %w", path, err)
	}
	if problems := chip.Validate(); !problems.OK() {
		return nil, fmt.Errorf("%s: hardware entry is not a valid chip descriptor: %s",
			path, problems.Error())
	}
	return chip, nil
}

// loadCatalogWorkloadEntries loads every workloads/<name>.yaml preset through
// readCatalogPresetWorkload — the production reader shared by `blis run --workload`,
// `blis observe --workload` and `blis convert preset --name` (#1769).
func loadCatalogWorkloadEntries(root string) (int, []string) {
	dir := filepath.Join(root, catalogWorkloadsSubdir)
	paths, err := catalogYAMLFiles(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, []string{fmt.Sprintf("%s: workloads namespace cannot be read: %v", dir, err)}
	}
	var problems []string
	loaded := 0
	for _, path := range paths {
		name := strings.TrimSuffix(filepath.Base(path), presetFileExt)
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			problems = append(problems, fmt.Sprintf("%s: workload preset is not readable: %v", path, readErr))
			continue
		}
		if yamlErr := checkCatalogAuthoredYAML(path, data); yamlErr != nil {
			problems = append(problems, yamlErr.Error())
			continue
		}
		if _, err := readCatalogPresetWorkload(name, root); err != nil {
			problems = append(problems, err.Error())
			continue
		}
		loaded++
	}
	return loaded, problems
}

// loadCatalogDeviceEntries loads devices/storage.yaml through loadCatalogStorageDevices —
// the production reader a run uses when a --kv-offload-config secondary tier names a
// device_class (#1770). Absent file = not a problem (the run-time dependency is lazy).
func loadCatalogDeviceEntries(root string) (int, []string) {
	path := catalogStorageDevicesPath(root)
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// An absent devices/ namespace is fine — nothing reads it until a tier names a
			// device_class. But a namespace that EXISTS and holds ANYTHING while missing the
			// one file BLIS reads is reported: that is what a misnamed storage table looks
			// like (storages.yaml), what a table filed one level too deep looks like
			// (devices/storage/storage.yaml), and what a namespace holding only prose looks
			// like — each would otherwise pass this gate as "no devices to load" and then
			// fail a real run with "device_class is not defined".
			//
			// Every entry is listed, not just the .yaml ones: filtering to YAML is what made
			// the subdirectory and non-YAML shapes indistinguishable from an absent
			// namespace, which is exactly the silence this diagnostic exists to break.
			if contents := catalogDirContents(filepath.Join(root, catalogDevicesSubdir)); len(contents) > 0 {
				return 0, []string{fmt.Sprintf("%s: the devices namespace holds %v but no %s, the only "+
					"storage-device table BLIS reads", path, contents, catalogStorageDevicesFile)}
			}
			return 0, nil
		}
		return 0, []string{fmt.Sprintf("%s: storage-device table is not readable: %v", path, err)}
	}
	if yamlErr := checkCatalogAuthoredYAML(path, data); yamlErr != nil {
		return 0, []string{yamlErr.Error()}
	}
	devices, err := loadCatalogStorageDevices(root)
	if err != nil {
		return 0, []string{err.Error()}
	}
	return len(devices), nil
}

// catalogYAMLFiles lists the .yaml files directly inside dir, sorted so a load is
// deterministic (INV-6). A missing directory is returned as os.ErrNotExist for the caller to
// treat as "namespace absent" rather than "namespace broken".
func catalogYAMLFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), catalogYAMLExt) {
			continue
		}
		paths = append(paths, filepath.Join(dir, e.Name()))
	}
	sort.Strings(paths)
	return paths, nil
}

// catalogDirContents lists everything dir holds, sorted (INV-6), with a directory marked by a
// trailing separator so "unexpected/" reads as one. Unlike catalogYAMLFiles it filters nothing:
// its caller is diagnosing a namespace that is PRESENT but has nothing BLIS can read, and a
// filter there would report that namespace as absent. An absent or unreadable dir yields
// nothing, which is the caller's "namespace absent" case.
func catalogDirContents(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			name += string(filepath.Separator)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// catalogDeploymentFactKeys are the key spellings a catalog file may NOT use, in the
// normalized form normalizeCatalogKey produces. Deployment choices — which GPU, how many
// tensor-parallel ranks — are stated on the command line and required there
// (requireDeploymentFlags, NS-6 / #1733); a catalog that also stated them would give a run
// two disagreeing sources for one fact, and the catalog's copy would be the one no operator
// chose.
//
// KEY-based, deliberately: a VALUE scan would have to guess which strings name GPUs
// ("h100" is also a plausible nickname in a provenance comment), while a key is a
// declaration. The set covers the spellings a person would actually reach for, including the
// vLLM/BLIS flag names.
var catalogDeploymentFactKeys = map[string]string{
	"gpu":                     "GPU type is a deployment choice (--hardware)",
	"gpus":                    "GPU type is a deployment choice (--hardware)",
	"gputype":                 "GPU type is a deployment choice (--hardware)",
	"hardware":                "GPU type is a deployment choice (--hardware)",
	"tp":                      "tensor-parallel degree is a deployment choice (--tp)",
	"tpsize":                  "tensor-parallel degree is a deployment choice (--tp)",
	"tpdegree":                "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallel":          "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelism":       "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelsize":      "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparalleldegree":    "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelismsize":   "tensor-parallel degree is a deployment choice (--tp)",
	"tensorparallelismdegree": "tensor-parallel degree is a deployment choice (--tp)",
}

// normalizeCatalogKey folds a YAML key to the form catalogDeploymentFactKeys is written in:
// lowercased with separators removed, so tensor_parallel_size, tensor-parallel-size,
// tensorParallelSize and TENSOR PARALLEL SIZE are one key.
func normalizeCatalogKey(key string) string {
	var b strings.Builder
	b.Grow(len(key))
	for _, r := range strings.ToLower(key) {
		switch r {
		case '_', '-', ' ', '.':
			continue
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// checkCatalogAuthoredYAML applies the two rules that hold for EVERY catalog-authored YAML
// file whatever namespace it sits in, and is the single gate each namespace reader calls before
// its typed parse (R23 — four readers each calling two guards would be free to drift):
//
//  1. it states no deployment fact, at any nesting depth, in ANY document of the file;
//  2. it holds exactly one document.
//
// Rule 2 is not tidiness. EVERY typed reader in BLIS decodes exactly ONE document — this file's
// model and hardware readers, readCatalogPresetWorkload (#1769), parseCatalogStorageDevices
// (#1770) — so content after a `---` separator is read by nothing: not the strict-key check,
// not a required-field check, not a run. That is the silent-drop defect strict parsing exists
// to prevent (R10) one level up from a single key, so a stream is refused rather than
// half-read. An EMPTY trailing document (a file ending in `---`) carries nothing and does not
// count: yaml.v3 yields a nil document for it, and refusing that would reject a harmless
// separator.
//
// Rule 1 is still checked over every document, so the specific "this states a deployment fact"
// diagnostic wins over the generic stream one.
func checkCatalogAuthoredYAML(path string, data []byte) error {
	docs, err := catalogYAMLDocuments(data)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if err := rejectCatalogDeploymentFacts(path, docs); err != nil {
		return err
	}
	if len(docs) > 1 {
		return fmt.Errorf("%s: catalog file is a multi-document YAML stream (%d documents). BLIS reads "+
			"exactly ONE document per catalog file, so everything after the first `---` separator is read "+
			"by nothing — not the strict-key check, not a run. Split it into one file per document",
			path, len(docs))
	}
	return nil
}

// catalogYAMLDocuments decodes every content-bearing document of a YAML stream.
//
// A failure on the FIRST document yields no documents and no error: the caller defers that
// diagnostic to the typed parse, which produces the real error for its namespace. A failure on a LATER document
// IS returned, because no typed parse ever reaches it.
func catalogYAMLDocuments(data []byte) ([]any, error) {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var docs []any
	for {
		var doc any
		err := decoder.Decode(&doc)
		if errors.Is(err, io.EOF) {
			return docs, nil
		}
		if err != nil {
			if len(docs) == 0 {
				return nil, nil //nolint:nilerr // defer the diagnostic to the typed parse
			}
			return nil, fmt.Errorf("document %d of this multi-document YAML stream does not parse: %w",
				len(docs)+1, err)
		}
		if doc == nil {
			continue // an empty document (a bare `---`) carries nothing to check
		}
		docs = append(docs, doc)
	}
}

// rejectCatalogDeploymentFacts refuses a catalog-authored YAML file that states a GPU or a
// tensor-parallel degree, at ANY nesting depth of ANY of its documents, naming the file and the
// offending key path.
//
// SCOPE: the catalog's own YAML (models/*/model.yaml, hardware/*.yaml, workloads/*.yaml,
// devices/*.yaml) — NOT the vendor config.json, which is committed VERBATIM and never
// edited. That boundary is load-bearing rather than tidy: 10 of the 23 committed vendor
// configs carry `pretraining_tp`, the TP degree a model was PRETRAINED with — an
// architectural fact of the checkpoint, not a choice about how to serve it. Scanning the
// vendor file would reject most of the catalog for stating something it is right to state.
func rejectCatalogDeploymentFacts(path string, docs []any) error {
	var offenders []string
	for i, doc := range docs {
		// A single-document file — every catalog file today — is qualified by key path alone.
		// A stream qualifies by document too, so the key is findable: the file is refused for
		// being a stream as well, but THIS diagnostic is the one that wins.
		prefix := ""
		if len(docs) > 1 {
			prefix = fmt.Sprintf("document[%d]", i+1)
		}
		offenders = append(offenders, collectCatalogDeploymentFacts(doc, prefix)...)
	}
	if len(offenders) == 0 {
		return nil
	}
	// Sorted so a multi-offender diagnostic is byte-identical across runs (INV-6).
	sort.Strings(offenders)
	return fmt.Errorf("%s: catalog file states deployment fact(s): %s. A catalog says what a model, "+
		"a chip, a workload or a storage tier IS; the deployment — which GPU, how many tensor-parallel "+
		"ranks — is stated on the command line and required there (--hardware / --tp). Remove the key(s)",
		path, strings.Join(offenders, "; "))
}

// collectCatalogDeploymentFacts walks a decoded YAML document and returns one entry per
// deployment-fact key, qualified by its path in the document.
func collectCatalogDeploymentFacts(node any, prefix string) []string {
	var offenders []string
	switch n := node.(type) {
	case map[string]any:
		for key, value := range n {
			qualified := key
			if prefix != "" {
				qualified = prefix + "." + key
			}
			if reason, bad := catalogDeploymentFactKeys[normalizeCatalogKey(key)]; bad {
				offenders = append(offenders, fmt.Sprintf("%q (%s)", qualified, reason))
			}
			offenders = append(offenders, collectCatalogDeploymentFacts(value, qualified)...)
		}
	case map[any]any:
		// yaml.v3 yields map[string]any for string keys; this arm covers a document with
		// non-string keys, which must still be walked rather than silently skipped.
		for key, value := range n {
			qualified := fmt.Sprintf("%v", key)
			if prefix != "" {
				qualified = prefix + "." + qualified
			}
			if reason, bad := catalogDeploymentFactKeys[normalizeCatalogKey(fmt.Sprintf("%v", key))]; bad {
				offenders = append(offenders, fmt.Sprintf("%q (%s)", qualified, reason))
			}
			offenders = append(offenders, collectCatalogDeploymentFacts(value, qualified)...)
		}
	case []any:
		for i, value := range n {
			offenders = append(offenders, collectCatalogDeploymentFacts(value, fmt.Sprintf("%s[%d]", prefix, i))...)
		}
	}
	return offenders
}
